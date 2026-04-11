from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assistant.models import ValidationCheckResult, ValidationIssue, ValidationResult


CODE_FENCE_RE = re.compile(r"```(?:lua|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
LUA_WRAPPER_RE = re.compile(r"lua\{(.*?)\}lua", re.IGNORECASE | re.DOTALL)
CODE_START_RE = re.compile(
    r"^\s*(?:\{|return\b|local\b|function\b|if\b|for\b|while\b|repeat\b|[A-Za-z_][A-Za-z0-9_]*\s*=|lua\{)"
)
RAW_ARRAY_RE = re.compile(r"\blocal\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{\s*\}")
WF_REF_RE = re.compile(r"\bwf\.(vars|initVariables)\.([A-Za-z_][A-Za-z0-9_]*)")
JSON_EMBEDDED_CODE_RE = re.compile(r"\b(return|local|function|wf\.)\b")

LAST_MARKERS = ("последн", "last")
INCREMENT_MARKERS = ("увелич", "increment", "счетчик", "counter")
ARRAY_MARKERS = ("отфильтр", "filter", "array", "массив")
JSON_RESULT_PHRASES = (
    "json payload",
    "json-payload",
    "верни json",
    "return json",
    "json объект",
    "json-объект",
)
MARK_AS_ARRAY_MARKERS = ("markasarray", "помет")
PLACEHOLDER_MARKERS = (
    "todo",
    "placeholder",
    "your_code",
    "epoch_seconds",
    "insert code here",
    "write code here",
    "boilerplate",
    "your logic",
)
TEMPLATE_MARKER_RE = re.compile(
    r"(?i)(insert(?: your)? code here|write(?: the)? code here|example code|sample code|boilerplate)"
)
NON_DETERMINISTIC_RE = re.compile(r"\bmath\.random(?:seed)?\b|\bRandom\.new\b")


@dataclass(slots=True)
class LuacCheckOutcome:
    status: str
    detail: str
    issue: ValidationIssue | None = None


class LocalScriptValidator:
    def validate(self, task: str, candidate: str) -> ValidationResult:
        raw_candidate = candidate or ""
        normalized_code = self.normalize(candidate)
        issues: list[ValidationIssue] = []
        checks: list[str] = []
        check_results: list[ValidationCheckResult] = []
        luac_status = "skipped_with_reason"
        luac_detail = "luac not executed."

        def pass_check(name: str, detail: str = "") -> None:
            checks.append(name)
            check_results.append(ValidationCheckResult(name=name, status="passed", detail=detail))

        def fail_check(rule: str, message: str, *, severity: str = "error") -> None:
            issues.append(ValidationIssue(rule=rule, message=message, severity=severity))
            check_results.append(ValidationCheckResult(name=rule, status="failed", detail=message))

        def skip_check(name: str, detail: str) -> None:
            check_results.append(ValidationCheckResult(name=name, status="skipped_with_reason", detail=detail))

        if not normalized_code:
            fail_check("non_empty", "The model returned an empty code block.")
            return ValidationResult(
                is_valid=False,
                normalized_code="",
                issues=issues,
                checks=checks,
                check_results=check_results,
                luac_status="skipped_with_reason",
                luac_detail="Empty normalized output.",
            )

        pass_check("normalized_output", f"chars={len(normalized_code)}")

        if "```" in candidate:
            fail_check("no_markdown", "Markdown code fences must not be returned.")
        else:
            pass_check("no_markdown")

        if "$." in normalized_code or "$[" in normalized_code:
            fail_check("no_jsonpath", "JsonPath access is forbidden in LocalScript.")
        else:
            pass_check("no_jsonpath")

        lower_task = task.lower()
        task_context = self._extract_json_context(task)
        task_wf_vars = self._wf_vars(task_context)
        task_init_vars = self._wf_init_variables(task_context)
        expects_workflow = bool(task_wf_vars or task_init_vars) or "wf." in lower_task
        expects_json_result = any(phrase in lower_task for phrase in JSON_RESULT_PHRASES)

        if expects_workflow and "wf." not in normalized_code:
            fail_check(
                "direct_wf_access",
                "The task contains workflow context, but the result does not access wf.vars or wf.initVariables.",
            )
        else:
            pass_check("wf_access")

        if task_init_vars and "wf.initVariables" not in normalized_code:
            fail_check("init_variables", "Startup variables must be accessed through wf.initVariables.")
        elif task_init_vars:
            pass_check("init_variables", f"keys={','.join(sorted(task_init_vars)[:8])}")

        if task_wf_vars and "wf.vars" not in normalized_code and "wf.initVariables" not in normalized_code:
            fail_check("wf_vars", "Workflow variables from wf.vars are available, but the result does not use them.")
        elif task_wf_vars:
            pass_check("wf_vars", f"keys={','.join(sorted(task_wf_vars)[:8])}")

        missing_wf_refs = self._find_missing_workflow_references(normalized_code, task_wf_vars, task_init_vars)
        if missing_wf_refs:
            fail_check(
                "unknown_wf_reference",
                "Result references workflow keys absent from provided context: " + ", ".join(missing_wf_refs),
            )
        elif expects_workflow:
            pass_check("wf_reference_consistency")

        if self._looks_hardcoded(task_context, normalized_code):
            fail_check(
                "no_hardcoded_samples",
                "The result appears to hardcode sample values instead of using wf.vars or wf.initVariables.",
            )
        else:
            pass_check("no_hardcoded_samples")

        if self._needs_array_constructor(lower_task, normalized_code):
            fail_check("array_constructor", "When building a new array result, use _utils.array.new().")
        elif any(marker in lower_task for marker in ARRAY_MARKERS):
            pass_check("array_constructor")

        if self._requires_mark_as_array(lower_task, normalized_code):
            fail_check(
                "mark_as_array",
                "When the task asks to mark an existing table as an array, use _utils.array.markAsArray(arr).",
            )
        elif "markasarray" in lower_task or "markasarray" in normalized_code:
            pass_check("mark_as_array")

        if any(marker in lower_task for marker in LAST_MARKERS) and "table.insert" in normalized_code:
            fail_check("last_element_semantics", "A task about the last element should not use table.insert().")
        elif any(marker in lower_task for marker in LAST_MARKERS):
            pass_check("last_element_semantics")

        if any(marker in lower_task for marker in INCREMENT_MARKERS) and "+ 1" not in normalized_code:
            fail_check("increment_semantics", "An increment task should increase the variable by one.")
        elif any(marker in lower_task for marker in INCREMENT_MARKERS):
            pass_check("increment_semantics")

        if expects_json_result:
            json_payload = self._parse_json_payload(normalized_code)
            if json_payload is None:
                fail_check(
                    "json_payload_shape",
                    "A JSON-oriented task should return a valid JSON object with LocalScript values wrapped as lua{...}lua.",
                )
            else:
                pass_check("json_payload_shape")
                json_issue = self._validate_json_payload(json_payload)
                if json_issue is not None:
                    fail_check(json_issue.rule, json_issue.message, severity=json_issue.severity)
                else:
                    pass_check("json_wrappers")
        else:
            pass_check("json_payload_shape", "JSON payload not required for this task.")

        stripped_code = normalized_code.strip()
        if not expects_json_result and stripped_code in {"{}", "[]"}:
            fail_check("non_trivial_code", "The result contains an empty container instead of executable LocalScript code.")
        elif (
            not expects_json_result
            and stripped_code.startswith("{")
            and "lua{" not in normalized_code
            and not stripped_code.startswith("{\"")
        ):
            fail_check(
                "standalone_table_literal",
                "A standalone table literal is not a valid top-level LocalScript chunk for this task.",
            )
        else:
            pass_check("non_trivial_code")

        placeholder_hits = [marker for marker in PLACEHOLDER_MARKERS if marker in raw_candidate.lower()]
        if placeholder_hits:
            fail_check(
                "no_placeholders",
                f"Result still contains unresolved placeholder content: {', '.join(placeholder_hits)}.",
            )
        else:
            pass_check("no_placeholders")

        template_match = TEMPLATE_MARKER_RE.search(raw_candidate)
        if template_match:
            fail_check(
                "no_templates",
                f"Result still contains template or boilerplate marker: {template_match.group(0)!r}.",
            )
        else:
            pass_check("no_templates")

        explicit_randomness = any(marker in lower_task for marker in ("random", "случайн"))
        if NON_DETERMINISTIC_RE.search(normalized_code) and not explicit_randomness:
            fail_check(
                "deterministic_output",
                "Avoid randomness in generated LocalScript unless the task explicitly requests it.",
            )
        else:
            pass_check("deterministic_output")

        lua_blocks = self._extract_lua_blocks(normalized_code)
        if not lua_blocks:
            fail_check("code_shape", "No Lua code could be extracted from the result.")
            return ValidationResult(
                is_valid=False,
                normalized_code=normalized_code,
                issues=issues,
                checks=checks,
                check_results=check_results,
                luac_status="skipped_with_reason",
                luac_detail="No Lua blocks extracted.",
            )

        pass_check("code_shape", f"blocks={len(lua_blocks)}")

        structural_issues = self._collect_structural_issues(lua_blocks)
        for issue in structural_issues:
            fail_check(issue.rule, issue.message, severity=issue.severity)

        failed_structural_rules = {issue.rule for issue in structural_issues}
        for rule in ("balanced_parentheses", "balanced_braces", "function_end"):
            if rule not in failed_structural_rules:
                pass_check(rule)

        luac_outcomes = [self._run_luac_check(block) for block in lua_blocks]
        if any(outcome.status == "failed" for outcome in luac_outcomes):
            first_failed = next(outcome for outcome in luac_outcomes if outcome.status == "failed")
            luac_status = "failed"
            luac_detail = first_failed.detail
            if first_failed.issue is not None:
                fail_check(first_failed.issue.rule, first_failed.issue.message, severity=first_failed.issue.severity)
        elif any(outcome.status == "passed" for outcome in luac_outcomes):
            luac_status = "passed"
            luac_detail = "luac parsed generated code successfully."
            pass_check("luac_parse", luac_detail)
        else:
            luac_status = "skipped_with_reason"
            luac_detail = luac_outcomes[0].detail if luac_outcomes else "luac not executed."
            skip_check("luac_parse", luac_detail)

        is_valid = not any(issue.severity == "error" for issue in issues)
        score, breakdown = self.score_with_breakdown(
            ValidationResult(
                is_valid=is_valid,
                normalized_code=normalized_code,
                issues=issues,
                checks=checks,
                check_results=check_results,
                luac_status=luac_status,
                luac_detail=luac_detail,
            ),
            normalized_code,
        )
        return ValidationResult(
            is_valid=is_valid,
            normalized_code=normalized_code,
            issues=issues,
            checks=checks,
            check_results=check_results,
            luac_status=luac_status,
            luac_detail=luac_detail,
            score_breakdown={**breakdown, "total": score},
        )

    def score(self, validation: ValidationResult, normalized_code: str, *, source: str = "llm", repair_round: int = 0) -> int:
        score, _ = self.score_with_breakdown(validation, normalized_code, source=source, repair_round=repair_round)
        return score

    def score_with_breakdown(
        self,
        validation: ValidationResult,
        normalized_code: str,
        *,
        source: str = "llm",
        repair_round: int = 0,
    ) -> tuple[int, dict[str, int]]:
        breakdown = {
            "validity": 140 if validation.is_valid else 20,
            "passed_checks": len(validation.checks) * 4,
            "error_penalty": -sum(22 for issue in validation.issues if issue.severity == "error"),
            "info_penalty": -sum(6 for issue in validation.issues if issue.severity != "error"),
            "shape_bonus": 6 if normalized_code.lstrip().startswith(("return", "local", "{", "function", "if")) else 0,
            "luac": 12 if validation.luac_status == "passed" else (-16 if validation.luac_status == "failed" else -3),
            "provenance": 4 if source == "repair" else 0,
            "repair_round": max(0, 6 - (repair_round * 2)),
            "length_penalty": -min(len(normalized_code) // 700, 8),
        }
        score = sum(breakdown.values())
        return score, breakdown

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

    def _collect_structural_issues(self, code_blocks: list[str]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if any(block.count("(") != block.count(")") for block in code_blocks):
            issues.append(ValidationIssue(rule="balanced_parentheses", message="Parentheses are unbalanced."))
        if any(block.count("{") != block.count("}") for block in code_blocks):
            issues.append(ValidationIssue(rule="balanced_braces", message="Braces are unbalanced."))
        if any(block.count("function") > block.count("end") for block in code_blocks):
            issues.append(ValidationIssue(rule="function_end", message="The number of 'end' keywords is too small."))
        return issues

    def _run_luac_check(self, code: str) -> LuacCheckOutcome:
        luac_path = shutil.which("luac")
        if luac_path is None:
            return LuacCheckOutcome(status="skipped_with_reason", detail="luac binary is not available in PATH.")

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
        except subprocess.TimeoutExpired:
            return LuacCheckOutcome(status="skipped_with_reason", detail="luac check timed out.")
        except OSError as exc:
            return LuacCheckOutcome(status="skipped_with_reason", detail=f"luac execution failed: {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)

        if result.returncode == 0:
            return LuacCheckOutcome(status="passed", detail="luac parsed generated code successfully.")

        error_text = result.stderr.strip() or result.stdout.strip() or "luac reported a syntax error."
        return LuacCheckOutcome(
            status="failed",
            detail=error_text,
            issue=ValidationIssue(rule="luac_parse", message=error_text),
        )

    def _parse_json_payload(self, normalized_code: str) -> dict[str, Any] | None:
        stripped = normalized_code.strip()
        if not stripped.startswith("{"):
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _validate_json_payload(self, payload: dict[str, Any]) -> ValidationIssue | None:
        executable_fields = 0
        for key, value in payload.items():
            if isinstance(value, dict):
                return ValidationIssue(
                    rule="json_nested_objects",
                    message=f"JSON payload field '{key}' must not contain nested JSON objects for executable LocalScript output.",
                )
            if isinstance(value, list):
                return ValidationIssue(
                    rule="json_arrays_not_wrapped",
                    message=f"JSON payload field '{key}' contains an array; executable Lua values must be wrapped as lua{{...}}lua strings.",
                )
            if isinstance(value, str):
                if value.startswith("lua{") and value.endswith("}lua"):
                    executable_fields += 1
                    continue
                if JSON_EMBEDDED_CODE_RE.search(value):
                    return ValidationIssue(
                        rule="json_lua_wrappers",
                        message=f"JSON payload field '{key}' contains executable Lua and must be wrapped as lua{{...}}lua.",
                    )
                continue
            if value is None or isinstance(value, (int, float, bool)):
                continue
            return ValidationIssue(
                rule="json_scalar_values",
                message=f"JSON payload field '{key}' has unsupported value type for judged output.",
            )

        if executable_fields == 0:
            return ValidationIssue(
                rule="json_lua_wrappers",
                message="JSON payloads must wrap executable Lua values as lua{...}lua strings.",
            )
        return None

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

    def _find_missing_workflow_references(
        self,
        normalized_code: str,
        wf_vars: dict[str, Any],
        init_vars: dict[str, Any],
    ) -> list[str]:
        if not (wf_vars or init_vars):
            return []

        missing: list[str] = []
        for scope, name in WF_REF_RE.findall(normalized_code):
            if scope == "vars" and wf_vars and name not in wf_vars:
                missing.append(f"wf.vars.{name}")
            if scope == "initVariables" and init_vars and name not in init_vars:
                missing.append(f"wf.initVariables.{name}")
        return sorted(set(missing))

    def _needs_array_constructor(self, lower_task: str, normalized_code: str) -> bool:
        if not any(marker in lower_task for marker in ARRAY_MARKERS):
            return False
        if "_utils.array.new()" in normalized_code:
            return False
        return "table.insert" in normalized_code or RAW_ARRAY_RE.search(normalized_code) is not None

    def _requires_mark_as_array(self, lower_task: str, normalized_code: str) -> bool:
        if "_utils.array.markAsArray" in normalized_code:
            return False
        if "markasarray" in lower_task:
            return True
        if "помет" in lower_task and ("массив" in lower_task or "array" in lower_task):
            return True
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
