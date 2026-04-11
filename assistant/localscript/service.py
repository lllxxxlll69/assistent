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
                    detail=f"LocalScript-режим не поддерживает запросы на языке {unsupported_language}.",
                )
            )
            validation = ValidationResult(
                is_valid=True,
                normalized_code=response_text,
                checks=["mode_guard"],
                check_results=[ValidationCheckResult(name="mode_guard", status="passed", detail=unsupported_language)],
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
                        detail="Запрошено уточнение из-за неоднозначности входа.",
                    )
                ],
                issues=[
                    ValidationIssue(
                        rule="clarification",
                        message="Нужен дополнительный контекст пользователя.",
                        severity="info",
                    )
                ],
            )
            logs.append(ActionLogEntry(message="Перед генерацией запрошено уточнение."))
            trace.append(
                GenerationTraceEntry(
                    stage="clarified",
                    status="requested",
                    detail=clarification_question,
                )
            )
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
                trace.append(GenerationTraceEntry(stage="assumed", status="applied", detail=assumption))
                logs.append(ActionLogEntry(message=f"Использовано допущение: {assumption}"))

        settings = self.settings_manager.get_settings()
        feedback_hints = self._feedback_hints()
        candidates: list[_Candidate] = []
        candidate_labels = self._candidate_labels(task, settings.localscript_candidate_count)
        trace.append(
            GenerationTraceEntry(
                stage="llm_cycle_started",
                status="running",
                detail=f"candidate_count={len(candidate_labels)}",
            )
        )
        for label in candidate_labels:
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
            trace.append(
                GenerationTraceEntry(
                    stage="llm_generated",
                    status="candidate_ready",
                    detail=f"{label}: score={candidate.score}, valid={candidate.validation.is_valid}",
                )
            )
            logs.append(
                ActionLogEntry(
                    message=(
                        f"Сгенерирован кандидат LocalScript '{label}' "
                        f"(score={candidate.score}, luac={candidate.validation.luac_status})"
                    ),
                    success=candidate.validation.is_valid,
                )
            )

        best_candidate = self._select_best_candidate(candidates)
        logs.append(ActionLogEntry(message=f"Предварительно выбрана стратегия LocalScript '{best_candidate.label}'.")) 

        if settings.localscript_auto_validate and not best_candidate.validation.is_valid:
            repair_pool = self._repair_pool(candidates)
            for candidate in repair_pool:
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
                    current_best_before = self._select_best_candidate(candidates)
                    trace.append(
                        GenerationTraceEntry(
                            stage="repaired",
                            status="candidate_ready",
                            detail=(
                                f"{repaired.label}: score={repaired.score}, "
                                f"valid={repaired.validation.is_valid}, luac={repaired.validation.luac_status}"
                            ),
                        )
                    )
                    if repaired.validation.is_valid:
                        logs.append(ActionLogEntry(message=f"Попытка исправления {attempt} дала валидный результат."))
                    else:
                        logs.append(
                            ActionLogEntry(
                                message=f"Попытка исправления {attempt} не прошла quality gate.",
                                success=False,
                            )
                        )
                    if repaired.validation.is_valid:
                        break
                    if repaired.score <= current.score:
                        break
                    current = repaired
                    best_candidate = current_best_before

        best_candidate = self._select_best_candidate(candidates)
        validation = best_candidate.validation
        raw_response = best_candidate.raw_response
        selected_strategy = best_candidate.label
        logs.append(ActionLogEntry(message=f"Выбрана финальная стратегия LocalScript '{selected_strategy}'.")) 
        logs.extend(self._validation_issue_logs(validation))
        if not validation.is_valid:
            trace.append(
                GenerationTraceEntry(
                    stage="validation_failed",
                    status="failed",
                    detail=f"Финальный кандидат содержит {len(validation.issues)} ошибок валидации.",
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
            raw_response=raw_response,
            selected_strategy=selected_strategy,
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
        labels: list[str] = ["baseline"]
        if any(marker in task.lower() for marker in JSON_RESULT_PHRASES):
            labels.append("json_payload")
        labels.append("strict")
        labels.append("repair_ready")

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
        top_errors = validation.issues[:4]
        issues_text = "\n".join(f"- [{issue.rule}] {issue.message}" for issue in top_errors) or "- Нет ошибок."
        assumptions_block = ""
        if assumptions:
            assumptions_block = "Допущения:\n" + "\n".join(f"- {item}" for item in assumptions) + "\n\n"
        messages = [
            {
                "role": "system",
                "content": (
                    self.knowledge_base.render_rules()
                    + "\n\n"
                    + self._self_check_block()
                    + "\n\nИсправь только то, что ломает валидацию. "
                    + "Пройди repair loop: синтаксис, логика, крайние случаи. "
                    + "Верни только итоговый LocalScript-код без пояснений."
                ),
            },
            *self._context_messages(context_messages),
            {
                "role": "user",
                "content": (
                    f"{assumptions_block}"
                    "Исходная задача:\n"
                    f"{task}\n\n"
                    "Текущий код:\n"
                    f"{candidate_code}\n\n"
                    "Проблемы quality gate:\n"
                    f"{issues_text}\n\n"
                    "Исправь только перечисленные проблемы. "
                    "Проверь выполнимость, семантику и крайние случаи перед ответом. "
                    "Не добавляй markdown, комментарии и пояснения."
                ),
            },
        ]
        return self.llm_client.chat(
            messages,
            model=settings.model,
            stream=False,
            max_tokens_override=settings.localscript_num_predict,
            context_size_override=settings.localscript_context_size,
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
        optional_blocks: list[str] = []
        if assumptions:
            optional_blocks.append(
                "Без диалога используй такие безопасные допущения:\n" + "\n".join(f"- {item}" for item in assumptions)
            )
        if feedback_hints:
            optional_blocks.append(
                "Недавний негативный пользовательский фидбек по плохим ответам:\n"
                + "\n".join(f"- {item}" for item in feedback_hints)
            )
        system_prompt = "\n\n".join(
            [
                self.knowledge_base.render_rules(),
                self._self_check_block(),
                self._strategy_prompt(strategy),
                f"Сводка контекста:\n{self._context_summary(task)}",
                f"Релевантные сигналы по похожим задачам:\n{guidance}",
                *optional_blocks,
                "Отвечай только итоговым кодом LocalScript без пояснений.",
            ]
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._context_messages(context_messages))
        messages.append(
            {
                "role": "user",
                "content": (
                    "Задача:\n"
                    f"{task}\n\n"
                    "Верни только финальный результат LocalScript. "
                    "Не объясняй решение. "
                    "Если в задаче есть workflow-контекст, используй wf.vars или wf.initVariables."
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
        if strategy == "json_payload":
            return (
                "Стратегия: верни только JSON-объект. "
                "Каждое исполняемое Lua-значение должно быть обёрнуто в строку lua{...}lua. "
                "Не добавляй лишние поля и не используй шаблонный каркас."
            )
        if strategy == "strict":
            return (
                "Стратегия: выбери самое короткое корректное решение LocalScript. "
                "Не придумывай переменные и поля, которых нет в контексте задачи. "
                "Сначала проверь семантику выполнения, потом сокращай код."
            )
        if strategy == "repair_ready":
            return (
                "Стратегия: подготовь ответ, который легко проходит quality gate. "
                "Избегай placeholder-ов, markdown, комментариев и примерных литералов из prompt-а. "
                "Если внутренний self-check находит дефект, перепиши ответ до возврата."
            )
        return (
            "Стратегия: сгенерируй самый прямой и практичный ответ LocalScript для этой задачи. "
            "Строй решение с нуля по данным задачи, а не по заготовке."
        )

    def _self_check_block(self) -> str:
        return (
            "Внутренний self-check перед ответом:\n"
            "1. Мысленно проверь синтаксис и luac-level выполнимость.\n"
            "2. Проверь доступы к wf.vars / wf.initVariables и отсутствие выдуманных переменных.\n"
            "3. Проверь семантику: входные данные, изменения состояния и точный итоговый output.\n"
            "4. Проверь крайние случаи: nil, пустой массив, отсутствующее поле, пустая строка.\n"
            "5. Если найден дефект, пересобери код и верни только исправленную финальную версию."
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
            return "Пришлите текущий LocalScript-код или JSON-контекст, который нужно исправить или доработать."
        if is_short and lacks_domain_signal:
            return "Уточните задачу и пришлите пример входного контекста, чтобы я сгенерировал корректный LocalScript."
        if asks_json and lacks_domain_signal:
            return "Уточните, какие поля должны попасть в JSON, и пришлите workflow-контекст для LocalScript."
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

    def _looks_like_code(self, text: str) -> bool:
        lowered = text.lower()
        return "wf." in lowered or "return " in lowered or "lua{" in lowered or "function " in lowered

    def _context_summary(self, task: str) -> str:
        context = self._extract_json_context(task)
        if not context:
            return "Структурированный JSON workflow-контекст не найден."

        wf_payload = context.get("wf")
        if not isinstance(wf_payload, dict):
            return "JSON-контекст найден, но payload wf отсутствует."

        parts: list[str] = []
        vars_payload = wf_payload.get("vars")
        if isinstance(vars_payload, dict) and vars_payload:
            vars_keys = ", ".join(sorted(vars_payload.keys())[:12])
            parts.append(f"Ключи wf.vars: {vars_keys}")

        init_payload = wf_payload.get("initVariables")
        if isinstance(init_payload, dict) and init_payload:
            init_keys = ", ".join(sorted(init_payload.keys())[:12])
            parts.append(f"Ключи wf.initVariables: {init_keys}")

        if not parts:
            return "JSON-контекст найден, но wf.vars и wf.initVariables пусты."
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
            if stripped in {"{}", "[]"}:
                hints["Не возвращай пустой контейнер вместо исполняемого LocalScript-кода."] += 1
            if "```" in message:
                hints["Не используй markdown fences в итоговом ответе."] += 1
            if "$." in message or "$[" in message:
                hints["Не используй JsonPath, только wf.vars и wf.initVariables."] += 1
            if stripped.startswith("{") and "lua{" not in stripped:
                hints["Для JSON payload оборачивай исполняемые Lua-значения в lua{...}lua."] += 1
            if "wf." not in message and any(token in message for token in ("return", "local", "function")):
                hints["Если в задаче есть workflow-контекст, используй wf.vars или wf.initVariables."] += 1
        return [hint for hint, _ in hints.most_common(3)]

    def _build_mode_guard_message(self, language: str) -> str:
        return (
            f"Этот чат сейчас работает в режиме генерации LocalScript/Lua. "
            f"Запрос на {language} здесь не поддерживается. "
            "Переключите чат в режим «Чат-бот», если нужен обычный ответ или код на другом языке."
        )
