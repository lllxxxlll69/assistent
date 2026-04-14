from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from assistant.app import build_backend
from assistant.config.settings import SettingsManager
from assistant.localscript.eval_cases import EvalCase, get_eval_cases
from assistant.localscript.knowledge import find_exact_prompt_overlaps, find_semantic_prompt_overlaps
from assistant.localscript.semantic_checks import (
    code_handles_timezone_offset,
    looks_like_iso8601_builder,
    parse_json_payload,
    task_requires_timezone_aware_unix,
    verify_json_payload_shape,
)
from assistant.models import Message


@dataclass(slots=True)
class PropertyCheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class EvalCaseResult:
    case_id: str
    category: str
    difficulty: str
    ok: bool
    selected_strategy: str
    response: str
    metrics: dict[str, Any]
    property_results: list[PropertyCheckResult] = field(default_factory=list)
    required_substring_results: dict[str, bool] = field(default_factory=dict)
    forbidden_pattern_results: dict[str, bool] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    notes: str = ""


def _property_check(name: str, case: EvalCase, response_text: str, metrics: dict[str, Any]) -> PropertyCheckResult:
    lowered = response_text.lower()

    if name == "validation_passed":
        passed = metrics.get("validation_errors", 1) == 0
        return PropertyCheckResult(name=name, passed=passed, detail=f"validation_errors={metrics.get('validation_errors')}")
    if name == "no_markdown":
        passed = "```" not in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="markdown fences absent" if passed else "markdown fences found")
    if name == "no_jsonpath":
        passed = "$." not in response_text and "$[" not in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="JsonPath absent" if passed else "JsonPath found")
    if name == "contains_return":
        passed = "return " in lowered
        return PropertyCheckResult(name=name, passed=passed, detail="contains return" if passed else "return not found")
    if name == "uses_wf_vars":
        passed = "wf.vars" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="wf.vars used" if passed else "wf.vars missing")
    if name == "uses_init_variables":
        passed = "wf.initVariables" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="wf.initVariables used" if passed else "wf.initVariables missing")
    if name == "uses_array_helper_new":
        passed = "_utils.array.new()" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="_utils.array.new() used" if passed else "array helper missing")
    if name == "uses_mark_as_array":
        passed = "_utils.array.markAsArray" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="markAsArray used" if passed else "markAsArray missing")
    if name == "json_payload_wrapped":
        passed = response_text.strip().startswith("{") and "lua{" in response_text and "}lua" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="JSON payload wrapped" if passed else "JSON wrappers missing")
    if name == "rest_cleanup_pattern":
        passed = "filtered_entry[key] = nil" in response_text or "key ~= \"ID\"" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="REST cleanup pattern detected" if passed else "REST cleanup pattern missing")
    if name == "unix_time_pattern":
        passed = "os.time" in response_text and "recallTime" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="unix time pattern detected" if passed else "unix time pattern missing")
    if name == "iso_8601_pattern":
        passed = "string.format" in response_text and "DATUM" in response_text and "TIME" in response_text
        return PropertyCheckResult(name=name, passed=passed, detail="ISO 8601 pattern detected" if passed else "ISO 8601 pattern missing")
    if name == "semantic_case_match":
        return _semantic_case_match(case, response_text, metrics)

    return PropertyCheckResult(name=name, passed=False, detail="Unknown property check.")


def _semantic_case_match(case: EvalCase, response_text: str, metrics: dict[str, Any]) -> PropertyCheckResult:
    del metrics
    stripped = response_text.strip()
    if case.category == "selection_last":
        expected_path = next((item for item in case.required_substrings if item.startswith("wf.")), "")
        passed = bool(expected_path) and expected_path in response_text and "#" in response_text
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="last-element semantics matched" if passed else "last-element semantics missing",
        )
    if case.category == "increment":
        expected_path = next((item for item in case.required_substrings if item.startswith("wf.")), "")
        passed = bool(expected_path) and expected_path in response_text and "+ 1" in response_text
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="increment semantics matched" if passed else "increment semantics missing",
        )
    if case.category == "direct_return":
        expected_return = next(iter(case.required_substrings), "")
        passed = expected_return in response_text
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="direct return matched" if passed else "direct return missing",
        )
    if case.category == "json_payload":
        payload_check = verify_json_payload_shape(case.prompt, stripped)
        payload = parse_json_payload(stripped) or {}
        wrappers_ok = payload_check.ok and all(
            isinstance(value, str) and value.startswith("lua{") and value.endswith("}lua")
            for value in payload.values()
        )
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=wrappers_ok,
            detail=payload_check.detail if not wrappers_ok else "JSON payload fields and wrappers matched",
        )
    if case.category == "datetime_iso":
        passed = looks_like_iso8601_builder(response_text)
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="ISO 8601 shape matched" if passed else "ISO 8601 shape missing or malformed",
        )
    if case.category == "datetime_unix":
        passed = "os.time" in response_text and (
            not task_requires_timezone_aware_unix(case.prompt) or code_handles_timezone_offset(response_text)
        )
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="unix timestamp semantics matched" if passed else "timezone-aware unix conversion missing",
        )
    if case.category == "rest_cleanup":
        passed = "wf.vars.RESTbody.result" in response_text and (
            "filtered_entry[key] = nil" in response_text or "key ~= \"ID\"" in response_text
        )
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="REST cleanup semantics matched" if passed else "REST cleanup semantics missing",
        )
    if case.category == "array_filter":
        passed = all(marker in response_text for marker in ("_utils.array.new()", "wf.vars.parsedCsv", "table.insert"))
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="array filter semantics matched" if passed else "array filter semantics missing",
        )
    if case.category == "array_helpers":
        passed = "_utils.array.markAsArray" in response_text and any(
            marker in response_text for marker in ("obj.items", "wf.vars.items", "wf.initVariables.packages", "ZCDF_PACKAGES")
        )
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail="array helper semantics matched" if passed else "array helper semantics missing",
        )
    if case.category == "assumptions":
        payload_check = verify_json_payload_shape(case.prompt, stripped)
        payload = parse_json_payload(stripped) or {}
        wrappers_ok = payload_check.ok and all(
            isinstance(value, str) and value.startswith("lua{") and value.endswith("}lua")
            for value in payload.values()
        )
        passed = wrappers_ok and '"squared"' in stripped
        return PropertyCheckResult(
            name="semantic_case_match",
            passed=passed,
            detail=payload_check.detail if not passed else "judged assumption payload matched",
        )
    return PropertyCheckResult(name="semantic_case_match", passed=True, detail="No extra semantic check for this category.")


async def run_eval_suite(
    *,
    smoke_only: bool = False,
    json_out: str | Path | None = None,
) -> dict[str, Any]:
    cases = get_eval_cases(smoke_only=smoke_only)
    overlap_report = find_exact_prompt_overlaps([(case.id, case.prompt) for case in cases])
    semantic_overlap_report = find_semantic_prompt_overlaps([(case.id, case.prompt) for case in cases])
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_root = Path(tmp_dir)
        settings_manager = SettingsManager(temp_root / "settings.json")
        if smoke_only:
            settings_manager.update_settings(
                localscript_candidate_count=1,
            )
        backend = build_backend(settings_manager=settings_manager, history_path=temp_root / "history.json")
        settings = settings_manager.get_settings()
        results: list[EvalCaseResult] = []
        failure_reasons = Counter[str]()
        by_category_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
        by_strategy = Counter[str]()
        by_luac_status = Counter[str]()
        by_syntax_engine = Counter[str]()
        by_execution_status = Counter[str]()
        by_sandbox_status = Counter[str]()
        runtime_guard_results = Counter[str]()
        repair_attempts: list[int] = []
        assumption_counts: list[int] = []

        for case in cases:
            backend.memory_manager.create_session(f"eval:{case.id}", assistant_mode="localscript")
            for role, content in case.context_messages:
                backend.memory_manager.add_message(role, content)

            response = await backend.orchestrator.generate_localscript_response(
                case.prompt,
                allow_clarification=False,
                persist_memory=False,
                use_memory_context=bool(case.context_messages),
            )

            required_results = {item: (item in response.text) for item in case.required_substrings}
            forbidden_results = {item: (item not in response.text) for item in case.forbidden_patterns}
            property_checks = list(case.property_checks)
            if "semantic_case_match" not in property_checks:
                property_checks.append("semantic_case_match")
            property_results = [_property_check(name, case, response.text, response.metrics) for name in property_checks]

            case_failures: list[str] = []
            for name, passed in required_results.items():
                if not passed:
                    case_failures.append(f"missing_required:{name}")
            for name, passed in forbidden_results.items():
                if not passed:
                    case_failures.append(f"forbidden_found:{name}")
            for item in property_results:
                if not item.passed:
                    case_failures.append(f"property_failed:{item.name}")
            if len(response.metrics.get("assumptions", [])) < case.expected_assumptions_min:
                case_failures.append("assumptions:too_few")

            ok = not case_failures
            result = EvalCaseResult(
                case_id=case.id,
                category=case.category,
                difficulty=case.difficulty,
                ok=ok,
                selected_strategy=str(response.metrics.get("selected_strategy", "")),
                response=response.text,
                metrics=response.metrics,
                property_results=property_results,
                required_substring_results=required_results,
                forbidden_pattern_results=forbidden_results,
                failure_reasons=case_failures,
                notes=case.notes,
            )
            results.append(result)

            by_category_totals[case.category]["total"] += 1
            if ok:
                by_category_totals[case.category]["passed"] += 1
            by_strategy.update([result.selected_strategy or "unknown"])
            by_luac_status.update([str(response.metrics.get("luac_status", "unknown"))])
            by_syntax_engine.update([str(response.metrics.get("syntax_engine", "unknown"))])
            by_execution_status.update([str(response.metrics.get("execution_status", "unknown"))])
            by_sandbox_status.update([str(response.metrics.get("sandbox_status", "unknown"))])
            runtime_info = response.metrics.get("runtime_info")
            if isinstance(runtime_info, dict) and runtime_info:
                runtime_guard_results.update(["present"])
            else:
                runtime_guard_results.update(["absent"])
            repair_attempts.append(int(response.metrics.get("repair_attempts_used", 0)))
            assumption_counts.append(len(response.metrics.get("assumptions", [])))
            for reason in case_failures:
                failure_reasons.update([reason])

        total = len(results)
        passed = sum(1 for item in results if item.ok)
        report = {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "report_schema_version": 2,
            "suite": "smoke" if smoke_only else "full",
            "model": settings.localscript_model,
            "chat_model": settings.model,
            "localscript_model": settings.localscript_model,
            "localscript_runtime_guard": settings.localscript_runtime_guard,
            "localscript_require_full_gpu": settings.localscript_require_full_gpu,
            "localscript_context_size": settings.localscript_context_size,
            "localscript_num_predict": settings.localscript_num_predict,
            "cases_total": total,
            "cases_passed": passed,
            "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
            "luac_status_distribution": dict(by_luac_status),
            "syntax_engine_distribution": dict(by_syntax_engine),
            "execution_status_distribution": dict(by_execution_status),
            "sandbox_status_distribution": dict(by_sandbox_status),
            "selected_strategy_distribution": dict(by_strategy),
            "runtime_guard_distribution": dict(runtime_guard_results),
            "knowledge_eval_overlap": {
                "exact_overlap_count": len(overlap_report),
                "exact_overlaps": overlap_report,
                "semantic_overlap_count": len(semantic_overlap_report),
                "semantic_overlaps": semantic_overlap_report,
            },
            "failure_reasons": dict(failure_reasons),
            "avg_repair_attempts_used": round(mean(repair_attempts), 3) if repair_attempts else 0.0,
            "avg_assumptions_used": round(mean(assumption_counts), 3) if assumption_counts else 0.0,
            "categories": {
                category: {
                    "passed": values["passed"],
                    "total": values["total"],
                    "pass_rate": round((values["passed"] / values["total"]) * 100, 2) if values["total"] else 0.0,
                }
                for category, values in sorted(by_category_totals.items())
            },
            "results": [
                {
                    **asdict(item),
                    "property_results": [asdict(check) for check in item.property_results],
                }
                for item in results
            ],
        }

        if json_out is not None:
            output_path = Path(json_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LocalScript benchmark / eval runner.")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--json-out", default="")
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    report = asyncio.run(
        run_eval_suite(
            smoke_only=args.suite == "smoke",
            json_out=args.json_out or None,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
