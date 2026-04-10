from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from assistant.models import ValidationIssue, ValidationResult


CODE_FENCE_RE = re.compile(r"```(?:lua|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
LUA_WRAPPER_RE = re.compile(r"lua\{(.*?)\}lua", re.IGNORECASE | re.DOTALL)
CODE_START_RE = re.compile(
    r"^\s*(?:\{|return\b|local\b|function\b|if\b|for\b|while\b|repeat\b|[A-Za-z_][A-Za-z0-9_]*\s*=|lua\{)"
)
RAW_ARRAY_RE = re.compile(r"\blocal\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{\s*\}")

LAST_MARKERS = ("последн", "last")
INCREMENT_MARKERS = ("увелич", "increment", "счетчик", "counter")
ARRAY_MARKERS = ("отфильт", "filter", "array", "массив")
JSON_RESULT_PHRASES = (
    "json payload",
    "json-payload",
    "верни json",
    "return json",
    "json объект",
    "json-объект",
)
MARK_AS_ARRAY_MARKERS = ("markasarray", "помет")
PLACEHOLDER_MARKERS = ("todo", "placeholder", "your_code", "epoch_seconds")


class LocalScriptValidator:
    def validate(self, task: str, candidate: str) -> ValidationResult:
        normalized_code = self.normalize(candidate)
        issues: list[ValidationIssue] = []
        checks: list[str] = []

        if not normalized_code:
            issues.append(ValidationIssue(rule="non_empty", message="The model returned an empty code block."))
            return ValidationResult(is_valid=False, normalized_code="", issues=issues, checks=checks)

        checks.append("normalized_output")

        if "```" in candidate:
            issues.append(ValidationIssue(rule="no_markdown", message="Markdown code fences must not be returned."))
        else:
            checks.append("no_markdown")

        if "$." in normalized_code or "$[" in normalized_code:
            issues.append(ValidationIssue(rule="no_jsonpath", message="JsonPath access is forbidden in LocalScript."))
        else:
            checks.append("no_jsonpath")

        lower_task = task.lower()
        task_context = self._extract_json_context(task)
        task_wf_vars = self._wf_vars(task_context)
        task_init_vars = self._wf_init_variables(task_context)
        expects_workflow = bool(task_wf_vars or task_init_vars) or "wf." in lower_task
        expects_json_result = any(phrase in lower_task for phrase in JSON_RESULT_PHRASES)

        if expects_workflow and "wf." not in normalized_code:
            issues.append(
                ValidationIssue(
                    rule="direct_wf_access",
                    message="The task contains workflow context, but the result does not access wf.vars or wf.initVariables.",
                )
            )
        else:
            checks.append("wf_access")

        if task_init_vars and "wf.initVariables" not in normalized_code:
            issues.append(
                ValidationIssue(
                    rule="init_variables",
                    message="Startup variables must be accessed through wf.initVariables.",
                )
            )
        elif task_init_vars:
            checks.append("init_variables")

        if task_wf_vars and "wf.vars" not in normalized_code and "wf.initVariables" not in normalized_code:
            issues.append(
                ValidationIssue(
                    rule="wf_vars",
                    message="Workflow variables from wf.vars are available, but the result does not use them.",
                )
            )

        if self._looks_hardcoded(task_context, normalized_code):
            issues.append(
                ValidationIssue(
                    rule="no_hardcoded_samples",
                    message="The result appears to hardcode sample values instead of using wf.vars or wf.initVariables.",
                )
            )
        else:
            checks.append("no_hardcoded_samples")

        if self._needs_array_constructor(lower_task, normalized_code):
            issues.append(
                ValidationIssue(
                    rule="array_constructor",
                    message="When building a new array result, use _utils.array.new().",
                )
            )
        elif any(marker in lower_task for marker in ARRAY_MARKERS):
            checks.append("array_constructor")

        if self._requires_mark_as_array(lower_task, normalized_code):
            issues.append(
                ValidationIssue(
                    rule="mark_as_array",
                    message="When the task asks to mark an existing table as an array, use _utils.array.markAsArray(arr).",
                )
            )
        elif "markasarray" in lower_task or "markasarray" in normalized_code:
            checks.append("mark_as_array")

        if any(marker in lower_task for marker in LAST_MARKERS) and "table.insert" in normalized_code:
            issues.append(
                ValidationIssue(
                    rule="last_element_semantics",
                    message="A task about the last element should not use table.insert().",
                )
            )

        if any(marker in lower_task for marker in INCREMENT_MARKERS) and "+ 1" not in normalized_code:
            issues.append(
                ValidationIssue(
                    rule="increment_semantics",
                    message="An increment task should increase the variable by one.",
                )
            )

        if expects_json_result and not normalized_code.lstrip().startswith("{"):
            issues.append(
                ValidationIssue(
                    rule="json_payload_shape",
                    message="A JSON-oriented task should return a JSON object with LocalScript values wrapped as lua{...}lua.",
                )
            )
        elif expects_json_result and "lua{" not in normalized_code:
            issues.append(
                ValidationIssue(
                    rule="json_lua_wrappers",
                    message="JSON payloads must wrap Lua values as lua{...}lua strings.",
                )
            )
        elif normalized_code.lstrip().startswith("{") and "lua{" in normalized_code:
            checks.append("json_wrappers")

        stripped_code = normalized_code.strip()
        if not expects_json_result and stripped_code in {"{}", "[]"}:
            issues.append(
                ValidationIssue(
                    rule="non_trivial_code",
                    message="The result contains an empty container instead of executable LocalScript code.",
                )
            )
        elif (
            not expects_json_result
            and stripped_code.startswith("{")
            and "lua{" not in normalized_code
            and not stripped_code.startswith("{\"")
        ):
            issues.append(
                ValidationIssue(
                    rule="standalone_table_literal",
                    message="A standalone table literal is not a valid top-level LocalScript chunk for this task.",
                )
            )

        placeholder_hits = [marker for marker in PLACEHOLDER_MARKERS if marker in normalized_code.lower()]
        if placeholder_hits:
            issues.append(
                ValidationIssue(
                    rule="no_placeholders",
                    message=f"Result still contains unresolved placeholder content: {', '.join(placeholder_hits)}.",
                )
            )
        else:
            checks.append("no_placeholders")

        lua_blocks = self._extract_lua_blocks(normalized_code)
        if not lua_blocks:
            issues.append(ValidationIssue(rule="code_shape", message="No Lua code could be extracted from the result."))
            return ValidationResult(is_valid=False, normalized_code=normalized_code, issues=issues, checks=checks)

        for block in lua_blocks:
            issues.extend(self._check_balanced_symbols(block))
            syntax_issue = self._try_luac_check(block)
            if syntax_issue is not None:
                issues.append(syntax_issue)
            else:
                checks.append("luac_or_heuristic")

        return ValidationResult(
            is_valid=not any(issue.severity == "error" for issue in issues),
            normalized_code=normalized_code,
            issues=issues,
            checks=checks,
        )

    def score(self, validation: ValidationResult, normalized_code: str) -> int:
        score = 100 if validation.is_valid else 0
        score += len(validation.checks) * 4
        score -= sum(18 if issue.severity == "error" else 5 for issue in validation.issues)
        if normalized_code.lstrip().startswith(("return", "local", "{")):
            score += 3
        score -= min(len(normalized_code) // 500, 5)
        return score

    def normalize(self, candidate: str) -> str:
        text = candidate.strip()
        if not text:
            return ""

        fenced = CODE_FENCE_RE.findall(text)
        if fenced:
            text = fenced[0].strip()

        lines = [line.rstrip() for line in text.splitlines()]
        start_index = 0
        for index, line in enumerate(lines):
            if CODE_START_RE.match(line):
                start_index = index
                break
        text = "\n".join(lines[start_index:]).strip()

        wrapped_matches = LUA_WRAPPER_RE.findall(text)
        if wrapped_matches and not text.lstrip().startswith("{"):
            if not text.lower().startswith("lua{") or not text.lower().endswith("}lua"):
                text = f"lua{{{wrapped_matches[-1].strip()}}}lua"

        if text.startswith('"') and text.endswith('"'):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, str):
                    text = decoded
            except json.JSONDecodeError:
                pass

        return text.strip()

    def _extract_lua_blocks(self, normalized_code: str) -> list[str]:
        if normalized_code.lstrip().startswith("{") and "lua{" in normalized_code:
            return [item.strip() for item in LUA_WRAPPER_RE.findall(normalized_code)]
        wrapped = LUA_WRAPPER_RE.findall(normalized_code)
        if wrapped:
            return [item.strip() for item in wrapped]
        return [normalized_code]

    def _check_balanced_symbols(self, code: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if code.count("(") != code.count(")"):
            issues.append(ValidationIssue(rule="balanced_parentheses", message="Parentheses are unbalanced."))
        if code.count("{") != code.count("}"):
            issues.append(ValidationIssue(rule="balanced_braces", message="Braces are unbalanced."))
        if code.count("function") > code.count("end"):
            issues.append(ValidationIssue(rule="function_end", message="The number of 'end' keywords is too small."))
        return issues

    def _try_luac_check(self, code: str) -> ValidationIssue | None:
        luac_path = shutil.which("luac")
        if luac_path is None:
            return None

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".lua", delete=False) as tmp_stream:
            tmp_stream.write(code)
            tmp_path = Path(tmp_stream.name)

        try:
            result = subprocess.run(
                [luac_path, "-p", str(tmp_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        finally:
            tmp_path.unlink(missing_ok=True)

        if result.returncode == 0:
            return None

        error_text = result.stderr.strip() or result.stdout.strip() or "luac reported a syntax error."
        return ValidationIssue(rule="luac_parse", message=error_text)

    def _extract_json_context(self, task: str) -> dict[str, Any] | None:
        start = task.find("{")
        end = task.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        raw_context = task[start : end + 1]
        try:
            payload = json.loads(raw_context)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _wf_vars(self, context: dict[str, Any] | None) -> dict[str, Any]:
        if not context:
            return {}
        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return {}
        vars_payload = wf_payload.get("vars")
        return vars_payload if isinstance(vars_payload, dict) else {}

    def _wf_init_variables(self, context: dict[str, Any] | None) -> dict[str, Any]:
        if not context:
            return {}
        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return {}
        init_payload = wf_payload.get("initVariables")
        return init_payload if isinstance(init_payload, dict) else {}

    def _needs_array_constructor(self, lower_task: str, normalized_code: str) -> bool:
        if not any(marker in lower_task for marker in ARRAY_MARKERS):
            return False
        if "_utils.array.new()" in normalized_code:
            return False
        return "table.insert" in normalized_code or RAW_ARRAY_RE.search(normalized_code) is not None

    def _requires_mark_as_array(self, lower_task: str, normalized_code: str) -> bool:
        if "markasarray" in normalized_code:
            return False
        if "markasarray" in lower_task:
            return True
        if "помет" in lower_task and ("массив" in lower_task or "array" in lower_task):
            return "_utils.array.markAsArray" not in normalized_code
        return False

    def _looks_hardcoded(self, task_context: dict[str, Any] | None, normalized_code: str) -> bool:
        prompt_literals = self._extract_context_literals(task_context)
        if not prompt_literals:
            return False
        matches = sum(1 for literal in prompt_literals if literal in normalized_code)
        return matches >= 2 and "wf." not in normalized_code

    def _extract_context_literals(self, task_context: dict[str, Any] | None) -> set[str]:
        literals: set[str] = set()
        if not task_context:
            return literals

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
                return
            if isinstance(value, list):
                for item in value:
                    walk(item)
                return
            if isinstance(value, str) and len(value) >= 3:
                literals.add(value)
                return
            if isinstance(value, (int, float)):
                literals.add(str(value))

        walk(task_context)
        return literals
