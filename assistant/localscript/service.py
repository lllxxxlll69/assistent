from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from assistant.config.settings import SettingsManager
from assistant.llm.client import LLMClient
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
ISO_TIME_MARKERS = ("iso 8601", "yyyymmdd", "hhmmss", "datum", "time")
MAX_ASSUMPTIONS = 2


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
            )

        if clarification_question:
            assumptions = self._derive_assumptions(task, context_messages)
            for assumption in assumptions:
                logs.append(ActionLogEntry(message=f"Использовано допущение: {assumption}"))
                trace.append(GenerationTraceEntry(stage="assumed", status="applied", detail=assumption))

        settings = self.settings_manager.get_settings()
        feedback_hints = self._feedback_hints()
        candidates: list[_Candidate] = []
        labels = self._candidate_labels(task, settings.localscript_candidate_count)
        trace.append(GenerationTraceEntry(stage="llm_cycle_started", status="running", detail=f"candidate_count={len(labels)}"))

        for label in labels:
            raw_response = self._generate_once(
                task=task,
                context_messages=context_messages,
                strategy=label,
                assumptions=assumptions,
                feedback_hints=feedback_hints,
            )
            candidate = self._build_candidate(task=task, label=label, source="llm", raw_response=raw_response)
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
                    )
                    repaired = self._build_candidate(
                        task=task,
                        label=f"{candidate.label}_repair_{attempt}",
                        source="repair",
                        raw_response=repaired_raw,
                        repair_round=attempt,
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
        )

    def _build_candidate(
        self,
        *,
        task: str,
        label: str,
        source: str,
        raw_response: str,
        repair_round: int = 0,
    ) -> _Candidate:
        validation = self.validator.validate(task, raw_response)
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

    def _candidate_labels(self, task: str, candidate_count: int) -> list[str]:
        family = self._task_family(task)
        labels: list[str] = ["baseline"]
        if family in {"selection_last", "increment", "datetime_unix", "datetime_iso", "direct_return"}:
            labels.append("return_first")
        if family in {"rest_cleanup", "array_filter", "array_helpers"}:
            labels.append("domain_strict")
        if family == "json_payload":
            labels.append("json_payload")
        labels.extend(["strict", "repair_ready"])
        deduplicated: list[str] = []
        for label in labels:
            if label not in deduplicated:
                deduplicated.append(label)
        return deduplicated[: max(1, candidate_count)]

    def _generate_once(
        self,
        *,
        task: str,
        context_messages: Sequence[Message],
        strategy: str,
        assumptions: Sequence[str],
        feedback_hints: Sequence[str],
    ) -> str:
        settings = self.settings_manager.get_settings()
        messages = self._build_generation_messages(
            task,
            context_messages=context_messages,
            strategy=strategy,
            assumptions=assumptions,
            feedback_hints=feedback_hints,
        )
        return self.llm_client.chat(
            messages,
            model=settings.model,
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
    ) -> str:
        settings = self.settings_manager.get_settings()
        top_errors = validation.issues[:5]
        issues_text = "\n".join(f"- [{issue.rule}] {issue.message}" for issue in top_errors) or "- No issues."
        assumptions_block = ""
        if assumptions:
            assumptions_block = "Assumptions:\n" + "\n".join(f"- {item}" for item in assumptions) + "\n\n"
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
            model=settings.model,
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
    ) -> list[dict[str, str]]:
        guidance = self.knowledge_base.render_generation_guidance(task, limit=3)
        examples = self.knowledge_base.render_examples(task, limit=2)
        optional_blocks: list[str] = []
        if assumptions:
            optional_blocks.append("Assumptions:\n" + "\n".join(f"- {item}" for item in assumptions))
        if feedback_hints:
            optional_blocks.append("Recent negative user feedback patterns:\n" + "\n".join(f"- {item}" for item in feedback_hints))

        system_prompt = "\n\n".join(
            [
                self.knowledge_base.render_rules(),
                self._self_check_block(),
                self._task_contract(task),
                self._strategy_prompt(strategy),
                f"Context summary:\n{self._context_summary(task)}",
                f"Task-family signals:\n{guidance}",
                (
                    "Reference implementations for similar task shapes. "
                    "Adapt the structure, but never copy identifiers or sample literals that are absent from the current task:\n"
                    f"{examples}"
                ),
                *optional_blocks,
                "Return only final LocalScript code. No markdown, no comments, no explanations, no print().",
            ]
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._context_messages(context_messages))
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

    def _context_messages(self, context_messages: Sequence[Message]) -> list[dict[str, str]]:
        relevant = list(context_messages[-6:])
        if relevant and relevant[-1].role == "user":
            relevant = relevant[:-1]
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

    def _select_best_candidate(self, candidates: Sequence[_Candidate]) -> _Candidate:
        return max(
            candidates,
            key=lambda item: (
                item.validation.is_valid,
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
        if ("return" in lowered or "верни" in lowered) and ("wf.vars" in lowered or "wf.initvariables" in lowered or "{" in lowered):
            return "direct_return"
        return "generic"

    def _task_contract(self, task: str) -> str:
        family = self._task_family(task)
        common = (
            "Output contract:\n"
            "- Return executable LocalScript only.\n"
            "- Never use print() or debug output in judged mode.\n"
            "- Prefer return over side-effect-only code.\n"
            "- Do not invent workflow keys that are absent from the prompt context."
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
                "- Prefer a compact parser with short local variables to stay within the judged token budget."
            ),
            "datetime_iso": (
                "- Build an ISO 8601 string from DATUM and TIME fields.\n"
                "- Return the final formatted string."
            ),
            "json_payload": (
                "- Return a JSON object only.\n"
                "- Wrap executable Lua values inside lua{...}lua strings."
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

    def _extract_json_context(self, task: str) -> dict[str, Any] | None:
        start = task.find("{")
        end = task.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(task[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

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
