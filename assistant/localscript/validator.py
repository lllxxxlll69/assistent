from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from assistant.localscript.semantic_checks import (
    code_handles_timezone_offset,
    extract_json_context,
    looks_like_iso8601_builder,
    run_execution_probe,
    task_requires_timezone_aware_unix,
    verify_json_payload_shape,
)
from assistant.localscript.lua_sandbox import run_lua_hidden_task_probes
from assistant.localscript.syntax_gate import run_syntax_gate
from assistant.models import ValidationCheckResult, ValidationIssue, ValidationResult


CODE_FENCE_RE = re.compile(r"```(?:lua|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
LUA_WRAPPER_RE = re.compile(r"lua\{(.*?)\}lua", re.IGNORECASE | re.DOTALL)
CODE_START_RE = re.compile(
    r"^\s*(?:\{|return\b|local\b|function\b|if\b|for\b|while\b|repeat\b|[A-Za-z_][A-Za-z0-9_]*\s*=|lua\{)"
)
RAW_ARRAY_RE = re.compile(r"\blocal\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{\s*\}")
WF_REF_RE = re.compile(r"\bwf\.(vars|initVariables)\.([A-Za-z_][A-Za-z0-9_]*)")
JSON_EMBEDDED_CODE_RE = re.compile(r"\b(return|local|function|wf\.)\b")
DIRECT_STRING_RETURN_RE = re.compile(r'^\s*return\s+([\'"])(.*)\1\s*$', re.DOTALL)
DIRECT_WF_RETURN_RE = re.compile(r"^\s*return\s+(wf\.(?:vars|initVariables)(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*$", re.DOTALL)

LAST_MARKERS = ("последн", "last")
INCREMENT_MARKERS = ("увелич", "increment", "счетчик", "counter")
ARRAY_MARKERS = ("отфильтр", "filter", "array", "массив")
REST_CLEANUP_MARKERS = ("rest", "restbody", "entity_id")
ARRAY_FILTER_MARKERS = ("discount", "markdown", "parsedcsv")
UNIX_TIME_MARKERS = ("recalltime", "unix", "timestamp")
PRINT_ALLOWED_MARKERS = ("print", "log", "вывед", "печать")
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
GENERIC_WORKFLOW_LEAVES = {"result", "value", "data", "item", "items", "payload", "response", "body", "list", "array"}
DIRECT_PASSTHROUGH_MARKERS = ("как есть", "as is", "без изменений", "напрямую", "directly")
TRANSFORMATION_HINTS = (
    "обработ",
    "очист",
    "преобраз",
    "конверт",
    "сформ",
    "собер",
    "отфильтр",
    "увелич",
    "добав",
    "измени",
    "доработ",
    "исправ",
    "format",
    "filter",
    "increment",
    "transform",
    "convert",
    "cleanup",
    "build",
    "normalize",
    "parse",
)


@dataclass(slots=True)
class LuacCheckOutcome:
    status: str
    detail: str
    engine: str = ""
    issue: ValidationIssue | None = None


class LocalScriptValidator:
    def validate(
        self,
        task: str,
        candidate: str,
        *,
        run_sandbox: bool = True,
        sandbox_timeout_ms: int = 900,
        sandbox_case_count: int = 2,
    ) -> ValidationResult:
        raw_candidate = candidate or ""
        normalized_code = self.normalize(candidate)
        issues: list[ValidationIssue] = []
        checks: list[str] = []
        check_results: list[ValidationCheckResult] = []
        luac_status = "skipped_with_reason"
        luac_detail = "luac not executed."
        syntax_engine = ""
        execution_status = "skipped_with_reason"
        execution_detail = "Execution probe not executed."
        sandbox_status = "skipped_with_reason"
        sandbox_detail = "Hidden-task sandbox was not executed."

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
                syntax_engine=syntax_engine,
                execution_status=execution_status,
                execution_detail=execution_detail,
                sandbox_status=sandbox_status,
                sandbox_detail=sandbox_detail,
            )

        pass_check("normalized_output", f"chars={len(normalized_code)}")

        if "```" in candidate:
            if CODE_FENCE_RE.fullmatch(candidate.strip()):
                pass_check("no_markdown", "Markdown fences were stripped during normalization.")
            else:
                fail_check("no_markdown", "Markdown code fences must not be returned.")
        else:
            pass_check("no_markdown")

        if "$." in normalized_code or "$[" in normalized_code:
            fail_check("no_jsonpath", "JsonPath access is forbidden in LocalScript.")
        else:
            pass_check("no_jsonpath")

        lower_task = task.lower()
        task_context = extract_json_context(task)
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

        if self._requires_return(lower_task, expects_json_result) and "return " not in normalized_code:
            fail_check("must_return", "The judged result must return the computed value.")
        elif self._requires_return(lower_task, expects_json_result):
            pass_check("must_return")

        if self._looks_like_explanatory_string_return(lower_task, normalized_code):
            fail_check(
                "natural_language_return",
                "The result returns a natural-language explanation in quotes instead of executable LocalScript logic.",
            )
        else:
            pass_check("natural_language_return")

        if self._looks_like_lazy_generic_passthrough(lower_task, normalized_code):
            fail_check(
                "generic_passthrough",
                "The result only returns a generic workflow field like wf.vars.result, but the task wording implies additional processing.",
            )
        else:
            pass_check("generic_passthrough")

        if "print(" in normalized_code and not any(marker in lower_task for marker in PRINT_ALLOWED_MARKERS):
            fail_check("no_print_debug", "Use return, not print(), in judged LocalScript output.")
        else:
            pass_check("no_print_debug")

        if any(marker in lower_task for marker in REST_CLEANUP_MARKERS):
            if "wf.vars.RESTbody.result" not in normalized_code:
                fail_check("rest_result_source", "REST cleanup must start from wf.vars.RESTbody.result.")
            else:
                pass_check("rest_result_source")
            if "filtered_entry[key] = nil" not in normalized_code and 'key ~=' not in normalized_code:
                fail_check("rest_cleanup_pattern", "REST cleanup must remove keys other than ID, ENTITY_ID, and CALL.")
            else:
                pass_check("rest_cleanup_pattern")

        if "discount" in lower_task or "markdown" in lower_task:
            if "wf.vars.parsedCsv" not in normalized_code:
                fail_check("parsed_csv_source", "Discount/Markdown filtering should use wf.vars.parsedCsv.")
            else:
                pass_check("parsed_csv_source")
            if "table.insert" not in normalized_code:
                fail_check("array_insert_pattern", "Filtered rows should be appended with table.insert.")
            else:
                pass_check("array_insert_pattern")

        if any(marker in lower_task for marker in UNIX_TIME_MARKERS):
            if "os.time" not in normalized_code:
                fail_check("unix_time", "Unix-time conversion must use os.time.")
            else:
                pass_check("unix_time")
            if task_requires_timezone_aware_unix(task):
                if not code_handles_timezone_offset(normalized_code):
                    fail_check(
                        "timezone_offset",
                        "Timezone offset from recallTime must be parsed and applied instead of ignored.",
                    )
                else:
                    pass_check("timezone_offset")

        if any(marker in lower_task for marker in ("iso 8601", "yyyymmdd", "hhmmss", "datum")):
            if not looks_like_iso8601_builder(normalized_code):
                fail_check(
                    "iso_8601_shape",
                    "ISO 8601 conversion must build YYYY-MM-DDTHH:MM:SS(.fraction)Z from DATUM/TIME.",
                )
            else:
                pass_check("iso_8601_shape")

        if expects_json_result:
            payload_check = verify_json_payload_shape(task, normalized_code)
            json_payload = payload_check.payload
            if json_payload is None or not payload_check.ok:
                fail_check(
                    "json_payload_shape",
                    payload_check.detail,
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
                syntax_engine=syntax_engine,
                execution_status=execution_status,
                execution_detail=execution_detail,
                sandbox_status=sandbox_status,
                sandbox_detail=sandbox_detail,
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
        syntax_engine = luac_outcomes[0].engine if luac_outcomes else ""
        if any(outcome.status == "failed" for outcome in luac_outcomes):
            first_failed = next(outcome for outcome in luac_outcomes if outcome.status == "failed")
            luac_status = "failed"
            luac_detail = first_failed.detail
            if first_failed.issue is not None:
                fail_check(first_failed.issue.rule, first_failed.issue.message, severity=first_failed.issue.severity)
        elif any(outcome.status == "passed" for outcome in luac_outcomes):
            luac_status = "passed"
            luac_detail = luac_outcomes[0].detail
            pass_check("luac_parse", f"{syntax_engine}: {luac_detail}" if syntax_engine else luac_detail)
        else:
            luac_status = "skipped_with_reason"
            luac_detail = luac_outcomes[0].detail if luac_outcomes else "luac not executed."
            skip_check("luac_parse", luac_detail)

        execution_probe = run_execution_probe(task, normalized_code)
        execution_status = execution_probe.status
        execution_detail = execution_probe.detail
        if execution_probe.status == "failed":
            fail_check("execution_probe", execution_probe.detail)
        elif execution_probe.status == "passed":
            pass_check("execution_probe", execution_probe.detail)
        else:
            skip_check("execution_probe", execution_probe.detail)

        if run_sandbox and luac_status == "passed":
            sandbox_probe = run_lua_hidden_task_probes(
                task,
                normalized_code,
                timeout_ms=sandbox_timeout_ms,
                max_cases=sandbox_case_count,
            )
            sandbox_status = sandbox_probe.status
            sandbox_detail = sandbox_probe.detail
            if sandbox_probe.status == "failed":
                fail_check("sandbox_hidden_tasks", sandbox_probe.detail)
            elif sandbox_probe.status == "passed":
                pass_check("sandbox_hidden_tasks", sandbox_probe.detail)
            else:
                skip_check("sandbox_hidden_tasks", sandbox_probe.detail)
        else:
            skip_check("sandbox_hidden_tasks", "Sandbox execution disabled or syntax gate did not pass.")

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
                syntax_engine=syntax_engine,
                execution_status=execution_status,
                execution_detail=execution_detail,
                sandbox_status=sandbox_status,
                sandbox_detail=sandbox_detail,
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
            syntax_engine=syntax_engine,
            execution_status=execution_status,
            execution_detail=execution_detail,
            sandbox_status=sandbox_status,
            sandbox_detail=sandbox_detail,
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
            "execution_probe": 18
            if validation.execution_status == "passed"
            else (-18 if validation.execution_status == "failed" else -2),
            "sandbox_hidden_tasks": 22
            if validation.sandbox_status == "passed"
            else (-20 if validation.sandbox_status == "failed" else -3),
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
        syntax_result = run_syntax_gate(code)
        if syntax_result.status == "passed":
            return LuacCheckOutcome(
                status="passed",
                detail=syntax_result.detail,
                engine=syntax_result.engine,
            )
        return LuacCheckOutcome(
            status="failed",
            detail=syntax_result.detail,
            engine=syntax_result.engine,
            issue=ValidationIssue(rule="luac_parse", message=syntax_result.detail),
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
        return extract_json_context(task)

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

    def _requires_return(self, lower_task: str, expects_json_result: bool) -> bool:
        if expects_json_result:
            return False
        if any(marker in lower_task for marker in LAST_MARKERS):
            return True
        if any(marker in lower_task for marker in INCREMENT_MARKERS):
            return True
        if any(marker in lower_task for marker in REST_CLEANUP_MARKERS):
            return True
        if any(marker in lower_task for marker in ARRAY_FILTER_MARKERS):
            return True
        if any(marker in lower_task for marker in UNIX_TIME_MARKERS):
            return True
        if "return" in lower_task or "верни" in lower_task:
            return True
        return False

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

    def _looks_like_explanatory_string_return(self, lower_task: str, normalized_code: str) -> bool:
        match = DIRECT_STRING_RETURN_RE.fullmatch(normalized_code.strip())
        if match is None:
            return False

        literal = match.group(2).strip()
        if not literal:
            return False

        literal_lower = literal.casefold()
        literal_tokens = re.findall(r"[A-Za-zА-Яа-я0-9_]+", literal_lower)
        if len(literal_tokens) < 5:
            return False

        explicit_string_request_markers = (
            "верни строк",
            "return string",
            "literal string",
            "текст ",
            "text ",
            "сообщени",
            "message",
            "статус",
            "status",
        )
        if any(marker in lower_task for marker in explicit_string_request_markers):
            return False

        explanatory_markers = (
            "я могу",
            "я умею",
            "i can",
            "i'm able",
            "workflow",
            "localscript",
            "lua",
            "переменн",
            "операц",
            "операци",
            "значени",
            "значения",
            "простые операции",
            "can return",
            "help",
            "помочь",
        )
        looks_like_explanation = any(marker in literal_lower for marker in explanatory_markers)
        looks_like_sentence = (" " in literal and literal[-1] in ".!?") or len(literal_tokens) >= 8
        return looks_like_explanation and looks_like_sentence

    def _looks_like_lazy_generic_passthrough(self, lower_task: str, normalized_code: str) -> bool:
        match = DIRECT_WF_RETURN_RE.fullmatch(normalized_code.strip())
        if match is None:
            return False

        path = match.group(1)
        leaf = path.split(".")[-1].casefold()
        if leaf not in GENERIC_WORKFLOW_LEAVES:
            return False
        if path.casefold() in lower_task:
            return False
        if any(marker in lower_task for marker in DIRECT_PASSTHROUGH_MARKERS):
            return False
        return any(marker in lower_task for marker in TRANSFORMATION_HINTS)

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
