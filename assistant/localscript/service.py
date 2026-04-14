from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from assistant.config.settings import SettingsManager
from assistant.llm.client import LLMClient, LLMClientError
from assistant.model_setup import list_installed_ollama_models
from assistant.localscript.semantic_checks import (
    extract_json_context,
    extract_numeric_literals,
    extract_requested_json_fields,
    task_requires_timezone_aware_unix,
)
from assistant.localscript.runtime import build_runtime_constraints, probe_ollama_runtime
from assistant.localscript.knowledge import LocalScriptKnowledgeBase
from assistant.localscript.validator import LocalScriptValidator
from assistant.models import (
    ActionLogEntry,
    CandidateArtifact,
    GenerationTraceEntry,
    LocalScriptGeneration,
    Message,
    ValidationCheckResult,
    ValidationIssue,
    ValidationResult,
)


REFINE_MARKERS = ("доработ", "исправ", "дополни", "улучши", "fix", "improve", "update")
REFINEMENT_FEEDBACK_MARKERS = (
    "не так",
    "неверн",
    "ошиб",
    "не подходит",
    "передел",
    "поправ",
)
JSON_RESULT_PHRASES = (
    "json payload",
    "json-payload",
    "верни json",
    "return json",
    "json объект",
    "json-объект",
)
LUA_MODE_MARKERS = ("lua", "localscript", "wf.vars", "wf.initvariables", "lua{")
NON_LUA_LANGUAGE_PATTERNS: dict[str, tuple[str, ...]] = {
    "C++": (r"\bc\+\+\b", r"\bcpp\b", r"\bси\+\+\b"),
    "Python": (r"\bpython\b", r"\bпитон\b"),
    "JavaScript": (r"\bjavascript\b", r"\bjs\b"),
    "TypeScript": (r"\btypescript\b", r"\bts\b"),
    "Java": (r"\bjava\b",),
    "C#": (r"\bc#\b", r"\bcsharp\b", r"\bc sharp\b"),
    "Go": (r"\bgolang\b", r"\bgo lang\b", r"\bна go\b"),
    "Rust": (r"\brust\b",),
    "PHP": (r"\bphp\b",),
    "Ruby": (r"\bruby\b",),
    "Kotlin": (r"\bkotlin\b",),
    "Swift": (r"\bswift\b",),
}
LAST_MARKERS = ("последн", "last")
INCREMENT_MARKERS = ("увелич", "increment", "счетчик", "counter")
REST_CLEANUP_MARKERS = ("rest", "restbody", "entity_id")
ARRAY_FILTER_MARKERS = ("discount", "markdown", "parsedcsv")
UNIX_TIME_MARKERS = ("recalltime", "unix", "timestamp")
ISO_TIME_MARKERS = ("iso 8601", "yyyymmdd", "hhmmss", "datum")
MAX_ASSUMPTIONS = 2
EXPLICIT_WORKFLOW_PATH_RE = re.compile(r"wf\.(?:vars|initVariables)(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
DIRECT_PASSTHROUGH_MARKERS = ("как есть", "as is", "без изменений", "напрямую", "directly")
GENERIC_WORKFLOW_LEAVES = {"result", "value", "data", "item", "items", "payload", "response", "body", "list", "array"}


@dataclass(slots=True)
class _Candidate:
    label: str
    source: str
    raw_response: str
    validation: ValidationResult
    score: int
    score_breakdown: dict[str, int]
    repair_round: int = 0


class LocalScriptService:
    def __init__(
        self,
        settings_manager: SettingsManager,
        llm_client: LLMClient,
        knowledge_base: LocalScriptKnowledgeBase | None = None,
        validator: LocalScriptValidator | None = None,
    ) -> None:
        self.settings_manager = settings_manager
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base or LocalScriptKnowledgeBase()
        self.validator = validator or LocalScriptValidator()
        self._runtime_guard_cache_key: tuple[Any, ...] | None = None
        self._runtime_guard_cache_value: dict[str, Any] | None = None

    def generate(
        self,
        task: str,
        *,
        context_messages: Sequence[Message] | None = None,
        allow_clarification: bool = True,
        interaction_mode: Literal["interactive", "judged"] = "interactive",
    ) -> LocalScriptGeneration:
        context_messages = context_messages or []
        logs: list[ActionLogEntry] = []
        trace: list[GenerationTraceEntry] = []
        assumptions: list[str] = []
        candidate_reports: list[CandidateArtifact] = []
        repair_attempts_used = 0
        runtime_info: dict[str, Any] = {}

        unsupported_language = self._detect_unsupported_language(task)
        if unsupported_language is not None:
            response_text = self._build_mode_guard_message(unsupported_language)
            validation = ValidationResult(
                is_valid=True,
                normalized_code=response_text,
                checks=["mode_guard"],
                check_results=[ValidationCheckResult(name="mode_guard", status="passed", detail=unsupported_language)],
            )
            logs.append(
                ActionLogEntry(
                    message=f"Остановлена LocalScript-генерация: запрос относится к языку {unsupported_language}.",
                    success=False,
                )
            )
            trace.append(
                GenerationTraceEntry(
                    stage="mode_guard",
                    status="passed",
                    detail=f"LocalScript mode does not support {unsupported_language} requests.",
                )
            )
            return LocalScriptGeneration(
                code=response_text,
                validation=validation,
                logs=logs,
                clarification_question=response_text,
                raw_response=response_text,
                selected_strategy="mode_guard",
                candidate_count=0,
                assumptions=assumptions,
                trace=trace,
                candidate_reports=candidate_reports,
                repair_attempts_used=repair_attempts_used,
                runtime_info=runtime_info,
            )

        clarification_question = self._build_clarification_question(task, context_messages)
        if clarification_question and allow_clarification and interaction_mode == "interactive":
            validation = ValidationResult(
                is_valid=True,
                normalized_code=clarification_question,
                checks=["clarification_requested"],
                check_results=[
                    ValidationCheckResult(
                        name="clarification_requested",
                        status="passed",
                        detail="Need more context before generating judged LocalScript.",
                    )
                ],
                issues=[ValidationIssue(rule="clarification", message="Need additional user context.", severity="info")],
            )
            logs.append(ActionLogEntry(message="Перед генерацией запрошено уточнение."))
            trace.append(GenerationTraceEntry(stage="clarified", status="requested", detail=clarification_question))
            return LocalScriptGeneration(
                code=clarification_question,
                validation=validation,
                logs=logs,
                clarification_question=clarification_question,
                raw_response=clarification_question,
                selected_strategy="clarification",
                candidate_count=0,
                assumptions=assumptions,
                trace=trace,
                candidate_reports=candidate_reports,
                repair_attempts_used=repair_attempts_used,
                runtime_info=runtime_info,
            )

        if clarification_question:
            assumptions = self._derive_assumptions(task, context_messages)
            for assumption in assumptions:
                logs.append(ActionLogEntry(message=f"Использовано допущение: {assumption}"))
                trace.append(GenerationTraceEntry(stage="assumed", status="applied", detail=assumption))

        settings = self.settings_manager.get_settings()
        generation_model = self._resolve_generation_model(
            settings,
            interaction_mode=interaction_mode,
            logs=logs,
            trace=trace,
        )
        is_refinement_request = self._is_refinement_request(task)
        conversation_guidance = self._conversation_guidance(task, context_messages)
        if conversation_guidance:
            trace.append(
                GenerationTraceEntry(
                    stage="feedback_synthesized",
                    status="applied",
                    detail=f"constraints={len(conversation_guidance)}",
                )
            )
            logs.append(ActionLogEntry(message=f"Собраны активные ограничения диалога: {len(conversation_guidance)}"))
        if interaction_mode == "judged" and settings.localscript_runtime_guard:
            runtime_info = self._ensure_judged_runtime(settings)
            logs.append(
                ActionLogEntry(
                    message=(
                        "Judged runtime guard passed: "
                        f"model={settings.localscript_model}, "
                        f"warm_up={runtime_info.get('warm_up_seconds', 0.0)}s"
                    )
                )
            )
            trace.append(
                GenerationTraceEntry(
                    stage="runtime_guard",
                    status="passed",
                    detail=(
                        f"model={settings.localscript_model}, "
                        f"loaded_models={runtime_info.get('loaded_models_count', 0)}"
                    ),
                )
            )
        candidates: list[_Candidate] = []
        symbolic_candidate = self._build_symbolic_candidate(task)
        if symbolic_candidate is not None:
            candidate = self._build_candidate(
                task=task,
                label="symbolic",
                source="symbolic",
                raw_response=symbolic_candidate,
                settings=settings,
                run_sandbox=False,
            )
            candidates.append(candidate)
            candidate_reports.append(self._to_candidate_artifact(candidate))
            logs.append(
                ActionLogEntry(
                    message=(
                        "Собран детерминированный LocalScript-кандидат "
                        f"(score={candidate.score}, valid={candidate.validation.is_valid}, luac={candidate.validation.luac_status})"
                    ),
                    success=candidate.validation.is_valid,
                )
            )
            trace.append(
                GenerationTraceEntry(
                    stage="symbolic_generated",
                    status="candidate_ready",
                    detail=f"symbolic: score={candidate.score}, valid={candidate.validation.is_valid}",
                )
            )
            if interaction_mode == "judged" and candidate.validation.is_valid:
                candidate = self._revalidate_candidate(task=task, candidate=candidate, settings=settings, run_sandbox=True)
                self._sync_candidate_artifact(candidate_reports, candidate)
                trace.append(
                    GenerationTraceEntry(
                        stage="final_selected",
                        status="passed",
                        detail=(
                            "source=symbolic, repairs=0, "
                            f"sandbox={candidate.validation.sandbox_status}"
                        ),
                    )
                )
                return LocalScriptGeneration(
                    code=candidate.validation.normalized_code,
                    validation=candidate.validation,
                    logs=logs + self._validation_issue_logs(candidate.validation),
                    clarification_question=None,
                    raw_response=candidate.raw_response,
                    selected_strategy="symbolic",
                    candidate_count=1,
                    assumptions=assumptions,
                    trace=trace,
                    candidate_reports=candidate_reports,
                    repair_attempts_used=repair_attempts_used,
                    runtime_info=runtime_info,
                )

        feedback_hints = self._feedback_hints()
        context_repair_candidate = self._build_context_repair_candidate(
            task=task,
            context_messages=context_messages,
            assumptions=assumptions,
            settings=settings,
            model_name=generation_model,
            conversation_guidance=conversation_guidance,
        )
        if context_repair_candidate is not None:
            candidates.append(context_repair_candidate)
            candidate_reports.append(self._to_candidate_artifact(context_repair_candidate))
            logs.append(
                ActionLogEntry(
                    message=(
                        "Собран refinement-кандидат из предыдущего ответа "
                        f"(score={context_repair_candidate.score}, valid={context_repair_candidate.validation.is_valid}, "
                        f"luac={context_repair_candidate.validation.luac_status})"
                    ),
                    success=context_repair_candidate.validation.is_valid,
                )
            )
            trace.append(
                GenerationTraceEntry(
                    stage="context_repair_generated",
                    status="candidate_ready",
                    detail=(
                        f"context_repair: score={context_repair_candidate.score}, "
                        f"valid={context_repair_candidate.validation.is_valid}"
                    ),
                )
            )
            if settings.localscript_fast_path and self._is_high_confidence_candidate(context_repair_candidate):
                context_repair_candidate = self._revalidate_candidate(
                    task=task,
                    candidate=context_repair_candidate,
                    settings=settings,
                    run_sandbox=True,
                )
                self._sync_candidate_artifact(candidate_reports, context_repair_candidate)
                trace.append(
                    GenerationTraceEntry(
                        stage="fast_path_selected",
                        status="passed",
                        detail="context_repair candidate satisfied the fast-path threshold.",
                    )
                )
                return LocalScriptGeneration(
                    code=context_repair_candidate.validation.normalized_code,
                    validation=context_repair_candidate.validation,
                    logs=logs + self._validation_issue_logs(context_repair_candidate.validation),
                    clarification_question=None,
                    raw_response=context_repair_candidate.raw_response,
                    selected_strategy=context_repair_candidate.label,
                    candidate_count=len(candidates),
                    assumptions=assumptions,
                    trace=trace,
                    candidate_reports=candidate_reports,
                    repair_attempts_used=repair_attempts_used,
                    runtime_info=runtime_info,
                )

        labels = self._candidate_labels(
            task,
            settings.localscript_candidate_count,
            has_feedback=is_refinement_request,
        )
        if context_repair_candidate is not None and settings.localscript_candidate_count <= 1:
            labels = []
        trace.append(GenerationTraceEntry(stage="llm_cycle_started", status="running", detail=f"candidate_count={len(labels)}"))

        for label in labels:
            raw_response = self._generate_once(
                task=task,
                context_messages=context_messages,
                strategy=label,
                assumptions=assumptions,
                feedback_hints=feedback_hints,
                conversation_guidance=conversation_guidance,
                settings=settings,
                model_name=generation_model,
            )
            candidate = self._build_candidate(
                task=task,
                label=label,
                source="llm",
                raw_response=raw_response,
                settings=settings,
                run_sandbox=False,
            )
            candidates.append(candidate)
            candidate_reports.append(self._to_candidate_artifact(candidate))
            logs.append(
                ActionLogEntry(
                    message=(
                        f"Сгенерирован кандидат LocalScript '{label}' "
                        f"(score={candidate.score}, valid={candidate.validation.is_valid}, luac={candidate.validation.luac_status})"
                    ),
                    success=candidate.validation.is_valid,
                )
            )
            trace.append(
                GenerationTraceEntry(
                    stage="llm_generated",
                    status="candidate_ready",
                    detail=f"{label}: score={candidate.score}, valid={candidate.validation.is_valid}",
                )
            )
            if settings.localscript_fast_path and self._is_high_confidence_candidate(candidate):
                trace.append(
                    GenerationTraceEntry(
                        stage="fast_path_triggered",
                        status="passed",
                        detail=f"{label}: stopped after first high-confidence candidate.",
                    )
                )
                break

        best_candidate = self._select_best_candidate(candidates)
        logs.append(ActionLogEntry(message=f"Предварительно выбрана стратегия LocalScript '{best_candidate.label}'."))

        if settings.localscript_auto_validate and not best_candidate.validation.is_valid:
            for candidate in self._repair_pool(candidates):
                current = candidate
                for attempt in range(1, settings.localscript_repair_attempts + 1):
                    repair_attempts_used += 1
                    repaired_raw = self._repair(
                        task=task,
                        candidate_code=current.validation.normalized_code,
                        validation=current.validation,
                        context_messages=context_messages,
                        assumptions=assumptions,
                        conversation_guidance=conversation_guidance,
                        settings=settings,
                        model_name=generation_model,
                    )
                    repaired = self._build_candidate(
                        task=task,
                        label=f"{candidate.label}_repair_{attempt}",
                        source="repair",
                        raw_response=repaired_raw,
                        repair_round=attempt,
                        settings=settings,
                        run_sandbox=False,
                    )
                    candidates.append(repaired)
                    candidate_reports.append(self._to_candidate_artifact(repaired))
                    logs.append(
                        ActionLogEntry(
                            message=(
                                f"Repair candidate '{repaired.label}' "
                                f"(score={repaired.score}, valid={repaired.validation.is_valid}, luac={repaired.validation.luac_status})"
                            ),
                            success=repaired.validation.is_valid,
                        )
                    )
                    trace.append(
                        GenerationTraceEntry(
                            stage="repaired",
                            status="candidate_ready",
                            detail=f"{repaired.label}: score={repaired.score}, valid={repaired.validation.is_valid}",
                        )
                    )
                    if repaired.validation.is_valid:
                        break
                    if repaired.score <= current.score:
                        break
                    current = repaired

        candidates = self._strictly_validate_shortlist(
            task=task,
            candidates=candidates,
            candidate_reports=candidate_reports,
            settings=settings,
            trace=trace,
        )
        best_candidate = self._select_best_candidate(candidates)
        if settings.localscript_auto_validate and not best_candidate.validation.is_valid and best_candidate.validation.sandbox_status == "failed":
            repair_attempts_used += 1
            repaired_raw = self._repair(
                task=task,
                candidate_code=best_candidate.validation.normalized_code,
                validation=best_candidate.validation,
                context_messages=context_messages,
                assumptions=assumptions,
                conversation_guidance=conversation_guidance,
                settings=settings,
                model_name=generation_model,
            )
            sandbox_repair = self._build_candidate(
                task=task,
                label=f"{best_candidate.label}_sandbox_repair",
                source="repair",
                raw_response=repaired_raw,
                repair_round=1,
                settings=settings,
                run_sandbox=True,
            )
            candidates.append(sandbox_repair)
            candidate_reports.append(self._to_candidate_artifact(sandbox_repair))
            trace.append(
                GenerationTraceEntry(
                    stage="sandbox_repaired",
                    status="candidate_ready",
                    detail=(
                        f"{sandbox_repair.label}: score={sandbox_repair.score}, "
                        f"valid={sandbox_repair.validation.is_valid}"
                    ),
                )
            )
            best_candidate = self._select_best_candidate(candidates)
        validation = best_candidate.validation
        logs.append(ActionLogEntry(message=f"Выбрана финальная стратегия LocalScript '{best_candidate.label}'.")) 
        logs.extend(self._validation_issue_logs(validation))
        if not validation.is_valid:
            trace.append(
                GenerationTraceEntry(
                    stage="validation_failed",
                    status="failed",
                    detail=f"final_issues={len(validation.issues)}",
                )
            )
        trace.append(
            GenerationTraceEntry(
                stage="final_selected",
                status="passed" if validation.is_valid else "failed",
                detail=(
                    f"source={best_candidate.source}, score={best_candidate.score}, "
                    f"luac={validation.luac_status}, repairs={repair_attempts_used}"
                ),
            )
        )

        return LocalScriptGeneration(
            code=validation.normalized_code,
            validation=validation,
            logs=logs,
            clarification_question=None,
            raw_response=best_candidate.raw_response,
            selected_strategy=best_candidate.label,
            candidate_count=len(candidates),
            assumptions=assumptions,
            trace=trace,
            candidate_reports=candidate_reports,
            repair_attempts_used=repair_attempts_used,
            runtime_info=runtime_info,
        )

    def _build_candidate(
        self,
        *,
        task: str,
        label: str,
        source: str,
        raw_response: str,
        repair_round: int = 0,
        settings=None,
        run_sandbox: bool = True,
    ) -> _Candidate:
        validation = self.validator.validate(
            task,
            raw_response,
            run_sandbox=run_sandbox,
            sandbox_timeout_ms=getattr(settings, "localscript_sandbox_timeout_ms", 900),
            sandbox_case_count=getattr(settings, "localscript_hidden_case_count", 2),
        )
        score, breakdown = self.validator.score_with_breakdown(
            validation,
            validation.normalized_code,
            source=source,
            repair_round=repair_round,
        )
        validation.score_breakdown = {**breakdown, "total": score}
        return _Candidate(
            label=label,
            source=source,
            raw_response=raw_response,
            validation=validation,
            score=score,
            score_breakdown=validation.score_breakdown,
            repair_round=repair_round,
        )

    def _to_candidate_artifact(self, candidate: _Candidate) -> CandidateArtifact:
        return CandidateArtifact(
            label=candidate.label,
            source=candidate.source,
            code=candidate.validation.normalized_code,
            score=candidate.score,
            is_valid=candidate.validation.is_valid,
            issues=[issue.message for issue in candidate.validation.issues],
            checks=list(candidate.validation.checks),
            luac_status=candidate.validation.luac_status,
            syntax_engine=candidate.validation.syntax_engine,
            execution_status=candidate.validation.execution_status,
            sandbox_status=candidate.validation.sandbox_status,
            repair_round=candidate.repair_round,
            score_breakdown=dict(candidate.score_breakdown),
        )

    def _validation_issue_logs(self, validation: ValidationResult) -> list[ActionLogEntry]:
        if validation.is_valid:
            return [ActionLogEntry(message="Проверка кандидата пройдена.")]
        return [
            ActionLogEntry(message=f"Проблема валидации: {issue.message}", success=False)
            for issue in validation.issues
            if issue.severity != "info"
        ]

    def _candidate_labels(self, task: str, candidate_count: int, *, has_feedback: bool = False) -> list[str]:
        family = self._task_family(task)
        labels: list[str] = ["baseline"]
        if has_feedback:
            labels.append("feedback_strict")
        if family in {"selection_last", "increment", "datetime_unix", "datetime_iso", "direct_return"}:
            labels.append("return_first")
        if family in {"rest_cleanup", "array_filter", "array_helpers"}:
            labels.append("domain_strict")
        if family == "json_payload":
            labels.extend(["json_payload", "json_shape_strict"])
        if family == "datetime_unix":
            labels.append("timezone_strict")
        if family == "datetime_iso":
            labels.append("iso_shape_strict")
        if family == "array_helpers":
            labels.append("array_helper_strict")
        labels.extend(["strict", "repair_ready"])
        deduplicated: list[str] = []
        for label in labels:
            if label not in deduplicated:
                deduplicated.append(label)
        return deduplicated[: max(1, candidate_count)]

    def _build_symbolic_candidate(self, task: str) -> str | None:
        family = self._task_family(task)
        if family == "selection_last":
            path = self._single_workflow_path(task)
            if path:
                return f"return {path}[#{path}]"
        if family == "increment":
            path = self._single_workflow_path(task)
            if path:
                return f"return {path} + 1"
        if family == "direct_return":
            path = self._requested_direct_return_path(task)
            if path:
                return f"return {path}"
        if family == "datetime_unix":
            return self._build_symbolic_unix_time_candidate(task)
        if family == "json_payload":
            return self._build_symbolic_json_payload_candidate(task)
        return None

    def _single_workflow_path(self, task: str, *, allow_implicit_single: bool = True) -> str | None:
        context = self._extract_json_context(task)
        if not context:
            return None
        candidates = self._workflow_paths_from_context(context)
        if len(candidates) == 1 and allow_implicit_single:
            return candidates[0]
        lowered = task.lower()
        exact_path_matches = [path for path in candidates if path.lower() in lowered]
        if len(exact_path_matches) == 1:
            return exact_path_matches[0]
        leaf_matches = [path for path in candidates if path.split(".")[-1].lower() in lowered]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        if exact_path_matches:
            exact_leafs = {path.split(".")[-1] for path in exact_path_matches}
            if len(exact_leafs) == 1:
                return sorted(exact_path_matches, key=len)[-1]
        return None

    def _requested_direct_return_path(self, task: str) -> str | None:
        lowered = task.casefold()
        explicit_paths = EXPLICIT_WORKFLOW_PATH_RE.findall(task)
        if explicit_paths:
            return sorted(set(explicit_paths), key=len)[-1]

        allow_implicit_single = any(marker in lowered for marker in DIRECT_PASSTHROUGH_MARKERS)
        path = self._single_workflow_path(task, allow_implicit_single=allow_implicit_single)
        if path is None:
            return None

        leaf = path.split(".")[-1].casefold()
        if leaf in GENERIC_WORKFLOW_LEAVES and not allow_implicit_single:
            return None
        if leaf in lowered:
            return path
        if allow_implicit_single:
            return path
        return None

    def _build_symbolic_unix_time_candidate(self, task: str) -> str | None:
        recall_time_required = "wf.initvariables.recalltime" in task.lower() or "recalltime" in task.lower()
        if not recall_time_required:
            return None
        return (
            "local y, m, d, h, mi, s, sign, tz_h, tz_m = "
            "wf.initVariables.recallTime:match(\"^(%d%d%d%d)%-(%d%d)%-(%d%d)T(%d%d):(%d%d):(%d%d)([%+%-])(%d%d):(%d%d)$\")\n"
            "if not y then\n"
            "    y, m, d, h, mi, s = wf.initVariables.recallTime:match(\"^(%d%d%d%d)%-(%d%d)%-(%d%d)T(%d%d):(%d%d):(%d%d)Z$\")\n"
            "    sign, tz_h, tz_m = \"+\", \"00\", \"00\"\n"
            "end\n"
            "local offset = (tonumber(tz_h) * 3600) + (tonumber(tz_m) * 60)\n"
            "local timestamp = os.time({year = tonumber(y), month = tonumber(m), day = tonumber(d), hour = tonumber(h), min = tonumber(mi), sec = tonumber(s)})\n"
            "return sign == \"+\" and (timestamp - offset) or (timestamp + offset)"
        )

    def _build_symbolic_json_payload_candidate(self, task: str) -> str | None:
        requested_fields = extract_requested_json_fields(task)
        if not requested_fields:
            return None

        numeric_literals = extract_numeric_literals(task)
        if {field.casefold() for field in requested_fields} == {"num", "squared"} and numeric_literals:
            literal = numeric_literals[0]
            payload = {
                "num": f"lua{{return tonumber('{literal}')}}lua",
                "squared": f"lua{{local n = tonumber('{literal}'); return n * n}}lua",
            }
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        context = self._extract_json_context(task)
        if not context:
            return None
        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return None

        payload: dict[str, str] = {}
        vars_payload = wf_payload.get("vars")
        init_payload = wf_payload.get("initVariables")
        for field in requested_fields:
            if isinstance(vars_payload, dict) and field in vars_payload:
                payload[field] = f"lua{{return wf.vars.{field}}}lua"
                continue
            if isinstance(init_payload, dict) and field in init_payload:
                payload[field] = f"lua{{return wf.initVariables.{field}}}lua"
                continue
            return None
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if payload else None

    def _generate_once(
        self,
        *,
        task: str,
        context_messages: Sequence[Message],
        strategy: str,
        assumptions: Sequence[str],
        feedback_hints: Sequence[str],
        conversation_guidance: Sequence[str],
        settings,
        model_name: str,
    ) -> str:
        messages = self._build_generation_messages(
            task,
            context_messages=context_messages,
            strategy=strategy,
            assumptions=assumptions,
            feedback_hints=feedback_hints,
            conversation_guidance=conversation_guidance,
        )
        return self.llm_client.chat(
            messages,
            model=model_name,
            stream=False,
            max_tokens_override=settings.localscript_num_predict,
            context_size_override=settings.localscript_context_size,
            temperature_override=settings.localscript_temperature,
        )

    def _repair(
        self,
        *,
        task: str,
        candidate_code: str,
        validation: ValidationResult,
        context_messages: Sequence[Message],
        assumptions: Sequence[str],
        conversation_guidance: Sequence[str],
        settings,
        model_name: str,
    ) -> str:
        top_errors = validation.issues[:5]
        issues_text = "\n".join(f"- [{issue.rule}] {issue.message}" for issue in top_errors) or "- No issues."
        assumptions_block = ""
        if assumptions:
            assumptions_block = "Assumptions:\n" + "\n".join(f"- {item}" for item in assumptions) + "\n\n"
        guidance_block = ""
        if conversation_guidance:
            guidance_block = "Active conversation constraints:\n" + "\n".join(f"- {item}" for item in conversation_guidance) + "\n\n"
        reference_examples = self.knowledge_base.render_examples(task, limit=1)
        messages = [
            {
                "role": "system",
                "content": (
                    self.knowledge_base.render_rules()
                    + "\n\n"
                    + self._self_check_block()
                    + "\n\n"
                    + self._task_contract(task)
                    + "\n\nRepair only the failing parts. Keep the result minimal and executable. "
                    + "Never use print() for final judged output. Return the final value instead."
                ),
            },
            *self._context_messages(context_messages),
            {
                "role": "user",
                "content": (
                    f"{assumptions_block}"
                    f"{guidance_block}"
                    "Original task:\n"
                    f"{task}\n\n"
                    "Reference shape for a similar task:\n"
                    f"{reference_examples}\n\n"
                    "Current code:\n"
                    f"{candidate_code}\n\n"
                    "Quality-gate failures:\n"
                    f"{issues_text}\n\n"
                    "Fix the exact failures and preserve task semantics. "
                    "Return only final LocalScript code. No markdown, no comments, no explanations, no print()."
                ),
            },
        ]
        return self.llm_client.chat(
            messages,
            model=model_name,
            stream=False,
            max_tokens_override=settings.localscript_num_predict,
            context_size_override=settings.localscript_context_size,
            temperature_override=settings.localscript_temperature,
        )

    def _build_generation_messages(
        self,
        task: str,
        *,
        context_messages: Sequence[Message],
        strategy: str,
        assumptions: Sequence[str],
        feedback_hints: Sequence[str],
        conversation_guidance: Sequence[str] = (),
    ) -> list[dict[str, str]]:
        guidance = self.knowledge_base.render_generation_guidance(task, limit=3)
        examples = self.knowledge_base.render_examples(task, limit=2)
        optional_blocks: list[str] = []
        if assumptions:
            optional_blocks.append("Assumptions:\n" + "\n".join(f"- {item}" for item in assumptions))
        if feedback_hints:
            optional_blocks.append("Recent negative user feedback patterns:\n" + "\n".join(f"- {item}" for item in feedback_hints))
        if conversation_guidance:
            optional_blocks.append("Active conversation constraints:\n" + "\n".join(f"- {item}" for item in conversation_guidance))

        system_prompt = "\n\n".join(
            [
                self.knowledge_base.render_rules(),
                self._self_check_block(),
                self._task_contract(task),
                f"Family-specific hints:\n{self._family_specific_hints(task)}",
                self._strategy_prompt(strategy),
                f"Context summary:\n{self._context_summary(task)}",
                f"Task-family signals:\n{guidance}",
                (
                    "Reference guidance for similar task shapes. "
                    "Adapt the structure, but never copy identifiers or sample literals that are absent from the current task:\n"
                    f"{examples}"
                ),
                *optional_blocks,
                "Return only final LocalScript code. No markdown, no comments, no explanations, no print().",
            ]
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._context_messages(context_messages, include_code_context=self._is_refinement_request(task)))
        messages.append(
            {
                "role": "user",
                "content": (
                    "Task:\n"
                    f"{task}\n\n"
                    "Generate the final LocalScript now. "
                    "Use wf.vars / wf.initVariables when context is provided. "
                    "Return the final value instead of printing it."
                ),
            }
        )
        return messages

    def _context_messages(
        self,
        context_messages: Sequence[Message],
        *,
        include_code_context: bool = True,
    ) -> list[dict[str, str]]:
        relevant = list(context_messages[-6:])
        if relevant and relevant[-1].role == "user":
            relevant = relevant[:-1]
        if not include_code_context:
            relevant = [
                item
                for item in relevant
                if not (item.role == "assistant" and self._looks_like_code(item.content))
            ]
        return [{"role": item.role, "content": item.content} for item in relevant]

    def _strategy_prompt(self, strategy: str) -> str:
        if strategy == "return_first":
            return (
                "Strategy: optimize for judged output shape. "
                "Use return as the final operation. Do not use print(), logs, or mutation-only answers."
            )
        if strategy == "domain_strict":
            return (
                "Strategy: follow the domain-specific structure exactly. "
                "Prefer the canonical workflow path and helper functions that best fit the task family."
            )
        if strategy == "json_payload":
            return (
                "Strategy: return a JSON object only. "
                "Wrap executable Lua values as lua{...}lua strings. Do not add extra fields."
            )
        if strategy == "json_shape_strict":
            return (
                "Strategy: infer the exact JSON field set from the task wording. "
                "Return only the requested fields, and wrap every executable Lua value as lua{...}lua."
            )
        if strategy == "timezone_strict":
            return (
                "Strategy: preserve timestamp semantics exactly. "
                "If the source ISO string contains a timezone offset, parse and apply that offset instead of ignoring it."
            )
        if strategy == "iso_shape_strict":
            return (
                "Strategy: build a canonical ISO 8601 string. "
                "Insert date and time separators explicitly and avoid returning compact YYYYMMDDTHHMMSS strings."
            )
        if strategy == "array_helper_strict":
            return (
                "Strategy: use array helpers explicitly. "
                "When normalizing existing tables, prefer _utils.array.markAsArray; use _utils.array.new() only for newly created arrays."
            )
        if strategy == "strict":
            return (
                "Strategy: choose the shortest semantically correct LocalScript. "
                "Do not invent variables or workflow keys that are absent from the prompt."
            )
        if strategy == "repair_ready":
            return (
                "Strategy: prepare an answer that passes quality-gate checks. "
                "Avoid markdown, placeholders, debug prints, and sample literals from prompt examples."
            )
        if strategy == "feedback_strict":
            return (
                "Strategy: treat the latest user correction as binding feedback. "
                "Preserve the previous intent, patch only the mismatched behavior, and avoid reintroducing prior mistakes."
            )
        return (
            "Strategy: build the solution from the current task and context only. "
            "Do not rely on canned templates or placeholder scaffolding."
        )

    def _self_check_block(self) -> str:
        return (
            "Internal self-check before final answer:\n"
            "1. Verify syntax and luac-level executability.\n"
            "2. Verify workflow paths, undefined names, and side effects.\n"
            "3. Verify edge cases such as nil, empty arrays, and missing fields.\n"
            "4. If a defect is found, rebuild the code and return only the corrected final version."
        )

    def _ensure_judged_runtime(self, settings) -> dict[str, Any]:
        cache_key = (
            settings.localscript_model,
            settings.localscript_context_size,
            settings.localscript_num_predict,
            settings.batch_size,
            settings.localscript_require_full_gpu,
            settings.localscript_full_gpu_ratio,
            settings.localscript_max_vram_bytes,
            settings.localscript_expected_digest,
        )
        if self._runtime_guard_cache_key == cache_key and self._runtime_guard_cache_value is not None:
            return dict(self._runtime_guard_cache_value)

        warm_up_seconds = self.llm_client.warm_up(
            settings.localscript_model,
            max_tokens_override=8,
            context_size_override=settings.localscript_context_size,
            temperature_override=settings.localscript_temperature,
        )
        runtime = probe_ollama_runtime(settings)
        constraints = build_runtime_constraints(settings, runtime)
        failures = [name for name, passed, _ in constraints if not passed]
        runtime_info = {
            "model": settings.localscript_model,
            "warm_up_seconds": round(warm_up_seconds, 3),
            "loaded_models_count": len(runtime.loaded_models),
            "constraints": [
                {"name": name, "passed": passed, "detail": detail}
                for name, passed, detail in constraints
            ],
            "probe": runtime.to_dict(),
        }
        if failures:
            details = ", ".join(
                f"{name}={detail}"
                for name, passed, detail in constraints
                if not passed
            )
            raise LLMClientError(f"Judged runtime guard failed: {details}")
        self._runtime_guard_cache_key = cache_key
        self._runtime_guard_cache_value = dict(runtime_info)
        return runtime_info

    def _select_best_candidate(self, candidates: Sequence[_Candidate]) -> _Candidate:
        return max(
            candidates,
            key=lambda item: (
                item.validation.is_valid,
                item.validation.sandbox_status == "passed",
                item.validation.luac_status == "passed",
                item.score,
                len(item.validation.checks),
                -len(item.validation.issues),
                -len(item.validation.normalized_code),
            ),
        )

    def _repair_pool(self, candidates: Sequence[_Candidate]) -> list[_Candidate]:
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.validation.is_valid,
                item.score,
                -len(item.validation.issues),
            ),
            reverse=True,
        )
        invalid = [item for item in ranked if not item.validation.is_valid]
        return invalid[:2]

    def _revalidate_candidate(self, *, task: str, candidate: _Candidate, settings, run_sandbox: bool) -> _Candidate:
        refreshed = self._build_candidate(
            task=task,
            label=candidate.label,
            source=candidate.source,
            raw_response=candidate.raw_response,
            repair_round=candidate.repair_round,
            settings=settings,
            run_sandbox=run_sandbox,
        )
        candidate.validation = refreshed.validation
        candidate.score = refreshed.score
        candidate.score_breakdown = refreshed.score_breakdown
        return candidate

    def _strictly_validate_shortlist(
        self,
        *,
        task: str,
        candidates: list[_Candidate],
        candidate_reports: list[CandidateArtifact],
        settings,
        trace: list[GenerationTraceEntry],
    ) -> list[_Candidate]:
        shortlist_size = max(1, settings.localscript_strict_shortlist_size)
        shortlist = [
            item
            for item in sorted(
                candidates,
                key=lambda candidate: (
                    candidate.validation.is_valid,
                    candidate.validation.execution_status == "passed",
                    candidate.validation.luac_status == "passed",
                    candidate.score,
                ),
                reverse=True,
            )
            if item.validation.luac_status == "passed"
        ][:shortlist_size]
        for candidate in shortlist:
            self._revalidate_candidate(task=task, candidate=candidate, settings=settings, run_sandbox=settings.localscript_sandbox_enabled)
            self._sync_candidate_artifact(candidate_reports, candidate)
            trace.append(
                GenerationTraceEntry(
                    stage="sandbox_validated",
                    status=candidate.validation.sandbox_status,
                    detail=f"{candidate.label}: sandbox={candidate.validation.sandbox_status}",
                )
            )
        return candidates

    def _sync_candidate_artifact(self, candidate_reports: list[CandidateArtifact], candidate: _Candidate) -> None:
        for index, artifact in enumerate(candidate_reports):
            if artifact.label == candidate.label:
                candidate_reports[index] = self._to_candidate_artifact(candidate)
                return

    def _is_high_confidence_candidate(self, candidate: _Candidate) -> bool:
        return (
            candidate.validation.is_valid
            and candidate.validation.luac_status == "passed"
            and candidate.validation.execution_status != "failed"
            and not candidate.validation.issues
        )

    def _recent_assistant_code(self, context_messages: Sequence[Message]) -> str | None:
        for message in reversed(context_messages):
            if message.role == "assistant" and self._looks_like_code(message.content):
                return message.content
        return None

    def _is_refinement_request(self, task: str) -> bool:
        lowered = task.casefold()
        if any(marker in lowered for marker in REFINE_MARKERS):
            return True
        return any(marker in lowered for marker in REFINEMENT_FEEDBACK_MARKERS)

    def _should_use_context_repair(self, task: str, context_messages: Sequence[Message]) -> bool:
        if not self._recent_assistant_code(context_messages):
            return False
        return self._is_refinement_request(task)

    def _resolve_generation_model(
        self,
        settings,
        *,
        interaction_mode: Literal["interactive", "judged"],
        logs: list[ActionLogEntry],
        trace: list[GenerationTraceEntry],
    ) -> str:
        preferred_model = settings.localscript_model.strip()
        fallback_model = settings.model.strip()

        if interaction_mode != "interactive":
            return preferred_model
        if not preferred_model:
            return fallback_model
        if not fallback_model or preferred_model == fallback_model:
            return preferred_model

        try:
            installed_models = list_installed_ollama_models(
                settings.api_url,
                request_timeout=settings.request_timeout,
            )
        except RuntimeError:
            return preferred_model

        installed_names = {item.name.strip().lower() for item in installed_models}
        if preferred_model.lower() in installed_names:
            return preferred_model
        if fallback_model.lower() not in installed_names:
            return preferred_model

        logs.append(
            ActionLogEntry(
                message=(
                    f"LocalScript model {preferred_model} не найдена в Ollama. "
                    f"Для интерактивного режима использую {fallback_model}."
                ),
                success=False,
            )
        )
        trace.append(
            GenerationTraceEntry(
                stage="interactive_model_fallback",
                status="applied",
                detail=f"{preferred_model} -> {fallback_model}",
            )
        )
        return fallback_model

    def _conversation_guidance(self, task: str, context_messages: Sequence[Message]) -> list[str]:
        guidance: list[str] = []
        lowered = task.casefold()
        if self._is_refinement_request(task) and self._recent_assistant_code(context_messages):
            guidance.append("There is previous assistant code in the dialog; preserve working parts and patch only the mismatched behavior.")
        if "print(" in lowered or "без print" in lowered or "no print" in lowered:
            guidance.append("Do not use print() in the final judged answer.")
        if "initvariables" in lowered:
            guidance.append("Prefer wf.initVariables for the requested data path.")
        if any(phrase in lowered for phrase in JSON_RESULT_PHRASES):
            guidance.append("Return only the requested JSON payload fields and keep executable Lua values wrapped.")
        deduplicated: list[str] = []
        for item in guidance:
            if item not in deduplicated:
                deduplicated.append(item)
        return deduplicated[:4]

    def _build_context_repair_candidate(
        self,
        *,
        task: str,
        context_messages: Sequence[Message],
        assumptions: Sequence[str],
        settings,
        model_name: str,
        conversation_guidance: Sequence[str],
    ) -> _Candidate | None:
        if not self._should_use_context_repair(task, context_messages):
            return None
        existing_code = self._recent_assistant_code(context_messages)
        if existing_code is None:
            return None
        existing_validation = self.validator.validate(task, existing_code, run_sandbox=False)
        refined_raw = self._repair(
            task=task,
            candidate_code=existing_validation.normalized_code or existing_code,
            validation=existing_validation,
            context_messages=context_messages,
            assumptions=assumptions,
            conversation_guidance=conversation_guidance,
            settings=settings,
            model_name=model_name,
        )
        return self._build_candidate(
            task=task,
            label="context_repair",
            source="repair",
            raw_response=refined_raw,
            settings=settings,
            run_sandbox=False,
        )

    def _build_clarification_question(self, task: str, context_messages: Sequence[Message]) -> str | None:
        lowered = task.lower()
        has_explicit_context = any(marker in lowered for marker in ("wf.vars", "wf.initvariables", "{", "lua", "localscript"))
        has_code_context = any(self._looks_like_code(message.content) for message in context_messages[-4:])
        asks_json = any(marker in lowered for marker in JSON_RESULT_PHRASES)
        is_short = len(task.strip()) < 18 or len(task.split()) < 3
        is_refine_without_code = any(marker in lowered for marker in REFINE_MARKERS) and not has_code_context
        lacks_domain_signal = not has_explicit_context and "wf." not in lowered

        if has_explicit_context or has_code_context:
            return None
        if is_refine_without_code:
            return (
                "Пришлите текущий LocalScript-код или короткий JSON-контекст "
                "с `wf.vars` / `wf.initVariables`, который нужно исправить или доработать."
            )
        if is_short and lacks_domain_signal:
            return (
                "Что именно должен делать LocalScript и какие данные приходят во `wf.vars` "
                "или `wf.initVariables`? Пришлите короткий пример входного контекста."
            )
        if asks_json and lacks_domain_signal:
            return (
                "Какие поля нужно вернуть в JSON и из каких `wf.vars` / `wf.initVariables` "
                "их брать? Пришлите короткий workflow-контекст."
            )
        return None

    def _derive_assumptions(self, task: str, context_messages: Sequence[Message]) -> list[str]:
        lowered = task.lower()
        assumptions: list[str] = []
        if any(marker in lowered for marker in REFINE_MARKERS) and not any(
            self._looks_like_code(message.content) for message in context_messages[-4:]
        ):
            assumptions.append("Исходный код не предоставлен, поэтому нужен новый LocalScript-кандидат с нуля.")
        if not self._extract_json_context(task):
            assumptions.append("Workflow-контекст не задан; нельзя придумывать sample values и лишние поля.")
        if any(marker in lowered for marker in JSON_RESULT_PHRASES):
            assumptions.append("Если схема JSON не указана, нужно вернуть только минимальные явно запрошенные поля.")
        return assumptions[:MAX_ASSUMPTIONS]

    def _task_family(self, task: str) -> str:
        lowered = task.lower()
        if any(marker in lowered for marker in JSON_RESULT_PHRASES):
            return "json_payload"
        if any(marker in lowered for marker in REST_CLEANUP_MARKERS):
            return "rest_cleanup"
        if any(marker in lowered for marker in ARRAY_FILTER_MARKERS):
            return "array_filter"
        if "markasarray" in lowered or ("помет" in lowered and "массив" in lowered):
            return "array_helpers"
        if any(marker in lowered for marker in UNIX_TIME_MARKERS):
            return "datetime_unix"
        if any(marker in lowered for marker in ISO_TIME_MARKERS):
            return "datetime_iso"
        if any(marker in lowered for marker in LAST_MARKERS):
            return "selection_last"
        if any(marker in lowered for marker in INCREMENT_MARKERS):
            return "increment"
        if ("return" in lowered or "верни" in lowered) and self._requested_direct_return_path(task):
            return "direct_return"
        return "generic"

    def _task_contract(self, task: str) -> str:
        family = self._task_family(task)
        common = (
            "Output contract:\n"
            "- Return executable LocalScript only.\n"
            "- Never use print() or debug output in judged mode.\n"
            "- Prefer return over side-effect-only code.\n"
            "- Do not invent workflow keys that are absent from the prompt context.\n"
            "- Never return a natural-language explanation wrapped in quotes."
        )
        family_contracts = {
            "selection_last": (
                "- For last-element tasks, return the last item directly from the workflow array.\n"
                "- Prefer a direct return expression over temporary variables."
            ),
            "increment": (
                "- For increment tasks, compute the incremented value and return it.\n"
                "- Do not stop at assignment-only code."
            ),
            "rest_cleanup": (
                "- For REST cleanup tasks, work from wf.vars.RESTbody.result.\n"
                "- Remove keys other than ID, ENTITY_ID, and CALL, then return the filtered result."
            ),
            "array_filter": (
                "- For Discount/Markdown filtering, use wf.vars.parsedCsv when the source array is not named explicitly.\n"
                "- Build the result with _utils.array.new(), add matched rows with table.insert, and return the new array."
            ),
            "array_helpers": (
                "- Use _utils.array.markAsArray for existing tables that must be marked as arrays.\n"
                "- Use _utils.array.new() only when creating a new array."
            ),
            "datetime_unix": (
                "- Read recallTime from wf.initVariables when present.\n"
                "- Convert to unix timestamp with os.time and return the timestamp.\n"
                "- If recallTime contains a timezone offset, parse and apply that offset instead of ignoring it.\n"
                "- Prefer a compact parser with short local variables to stay within the judged token budget."
            ),
            "datetime_iso": (
                "- Build an ISO 8601 string from DATUM and TIME fields.\n"
                "- Insert separators so the final shape is YYYY-MM-DDTHH:MM:SS(.fraction)Z.\n"
                "- Return the final formatted string."
            ),
            "json_payload": (
                "- Return a JSON object only.\n"
                "- Wrap executable Lua values inside lua{...}lua strings.\n"
                "- If the task names JSON fields explicitly, return exactly those fields and no extras."
            ),
            "direct_return": (
                "- Return the requested workflow path directly.\n"
                "- Do not print the value and do not wrap it in extra variables unless necessary."
            ),
        }
        specific = family_contracts.get(family)
        return common if specific is None else f"{common}\n{specific}"

    def _looks_like_code(self, text: str) -> bool:
        lowered = text.lower()
        return "wf." in lowered or "return " in lowered or "lua{" in lowered or "function " in lowered

    def _context_summary(self, task: str) -> str:
        context = self._extract_json_context(task)
        if not context:
            return "Structured JSON workflow context not found."

        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return "JSON context found, but wf payload is missing."

        parts: list[str] = []
        vars_payload = wf_payload.get("vars")
        if isinstance(vars_payload, dict) and vars_payload:
            vars_keys = ", ".join(sorted(vars_payload.keys())[:12])
            parts.append(f"wf.vars keys: {vars_keys}")

        init_payload = wf_payload.get("initVariables")
        if isinstance(init_payload, dict) and init_payload:
            init_keys = ", ".join(sorted(init_payload.keys())[:12])
            parts.append(f"wf.initVariables keys: {init_keys}")

        if not parts:
            return "JSON context found, but wf.vars and wf.initVariables are empty."
        return "\n".join(parts)

    def _workflow_paths_from_context(self, context: dict[str, Any]) -> list[str]:
        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return []

        candidates: list[str] = []
        vars_payload = wf_payload.get("vars")
        if isinstance(vars_payload, dict):
            candidates.extend(self._collect_workflow_paths(vars_payload, "wf.vars"))
        init_payload = wf_payload.get("initVariables")
        if isinstance(init_payload, dict):
            candidates.extend(self._collect_workflow_paths(init_payload, "wf.initVariables"))
        return candidates

    def _collect_workflow_paths(self, value: Any, prefix: str) -> list[str]:
        if isinstance(value, dict):
            collected: list[str] = []
            for key, nested in value.items():
                child_prefix = f"{prefix}.{key}"
                collected.extend(self._collect_workflow_paths(nested, child_prefix))
            return collected or [prefix]
        return [prefix]

    def _extract_json_context(self, task: str) -> dict[str, Any] | None:
        return extract_json_context(task)

    def _family_specific_hints(self, task: str) -> str:
        family = self._task_family(task)
        hints: list[str] = []
        if family == "json_payload":
            requested_fields = extract_requested_json_fields(task)
            if requested_fields:
                hints.append("Requested JSON fields: " + ", ".join(requested_fields))
            numeric_literals = extract_numeric_literals(task)
            if numeric_literals:
                hints.append("Numeric literals from the task: " + ", ".join(numeric_literals[:4]))
        if family == "datetime_unix" and task_requires_timezone_aware_unix(task):
            hints.append("The source recallTime contains a timezone offset, so the conversion must preserve absolute time.")
        if family == "datetime_iso":
            hints.append("The target format must include date separators, time separators, and a trailing UTC marker.")
        if family == "array_helpers":
            hints.append("Normalize nested items with explicit array helpers instead of ad hoc plain tables.")
        return "\n".join(f"- {item}" for item in hints) if hints else "- No extra family-specific hints."

    def _detect_unsupported_language(self, task: str) -> str | None:
        lowered = task.lower()
        if any(marker in lowered for marker in LUA_MODE_MARKERS):
            return None
        for language, patterns in NON_LUA_LANGUAGE_PATTERNS.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                return language
        return None

    def _feedback_hints(self) -> list[str]:
        feedback_path = self.settings_manager.settings_path.parent / "feedback.log"
        if not feedback_path.exists():
            return []
        try:
            rows = feedback_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        negative_messages: list[str] = []
        for row in rows[-50:]:
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                continue
            if payload.get("positive") is False and isinstance(payload.get("message"), str):
                negative_messages.append(payload["message"])

        hints = Counter[str]()
        for message in negative_messages:
            stripped = message.strip()
            lowered = stripped.lower()
            if stripped in {"{}", "[]"}:
                hints["Do not return an empty container instead of executable LocalScript code."] += 1
            if "```" in stripped:
                hints["Do not include markdown fences in the final answer."] += 1
            if "$." in stripped or "$[" in stripped:
                hints["Do not use JsonPath. Use wf.vars / wf.initVariables only."] += 1
            if stripped.startswith("{") and "lua{" not in stripped:
                hints["For JSON payloads, wrap executable Lua values as lua{...}lua strings."] += 1
            if "print(" in lowered:
                hints["Do not use print() for final judged output. Return the value instead."] += 1
            if "wf." not in stripped and any(token in lowered for token in ("return", "local", "function")):
                hints["If workflow context exists, use wf.vars or wf.initVariables explicitly."] += 1
        return [hint for hint, _ in hints.most_common(4)]

    def _build_mode_guard_message(self, language: str) -> str:
        return (
            f"Этот чат сейчас работает в режиме генерации LocalScript/Lua. "
            f"Запрос на {language} здесь не поддерживается. "
            "Переключите чат в режим «Чат-бот», если нужен обычный ответ или код на другом языке."
        )
