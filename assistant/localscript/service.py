from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from assistant.config.settings import SettingsManager
from assistant.llm.client import LLMClient
from assistant.localscript.knowledge import LocalScriptKnowledgeBase
from assistant.localscript.templates import LocalScriptTemplateEngine
from assistant.localscript.validator import LocalScriptValidator
from assistant.models import (
    ActionLogEntry,
    LocalScriptGeneration,
    Message,
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


@dataclass(slots=True)
class _Candidate:
    label: str
    raw_response: str
    validation: ValidationResult


class LocalScriptService:
    def __init__(
        self,
        settings_manager: SettingsManager,
        llm_client: LLMClient,
        knowledge_base: LocalScriptKnowledgeBase | None = None,
        validator: LocalScriptValidator | None = None,
        template_engine: LocalScriptTemplateEngine | None = None,
    ) -> None:
        self.settings_manager = settings_manager
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base or LocalScriptKnowledgeBase()
        self.validator = validator or LocalScriptValidator()
        self.template_engine = template_engine or LocalScriptTemplateEngine()

    def generate(
        self,
        task: str,
        *,
        context_messages: Sequence[Message] | None = None,
        allow_clarification: bool = True,
    ) -> LocalScriptGeneration:
        context_messages = context_messages or []
        logs: list[ActionLogEntry] = []

        unsupported_language = self._detect_unsupported_language(task)
        if unsupported_language is not None:
            response_text = self._build_mode_guard_message(unsupported_language)
            logs.append(
                ActionLogEntry(
                    message=f"Остановлена LocalScript-генерация: запрос относится к языку {unsupported_language}.",
                    success=False,
                )
            )
            validation = ValidationResult(
                is_valid=True,
                normalized_code=response_text,
                checks=["mode_guard"],
            )
            return LocalScriptGeneration(
                code=response_text,
                validation=validation,
                logs=logs,
                clarification_question=response_text,
                raw_response=response_text,
                selected_strategy="mode_guard",
                candidate_count=0,
            )

        clarification_question = (
            self._build_clarification_question(task, context_messages) if allow_clarification else None
        )
        if clarification_question:
            validation = ValidationResult(
                is_valid=True,
                normalized_code=clarification_question,
                checks=["clarification_requested"],
                issues=[ValidationIssue(rule="clarification", message="Нужен дополнительный контекст пользователя.", severity="info")],
            )
            logs.append(ActionLogEntry(message="Перед генерацией запрошено уточнение."))
            return LocalScriptGeneration(
                code=clarification_question,
                validation=validation,
                logs=logs,
                clarification_question=clarification_question,
                raw_response=clarification_question,
                selected_strategy="clarification",
                candidate_count=0,
            )

        template_code = self.template_engine.render(task)
        if template_code is not None:
            validation = self.validator.validate(task, template_code)
            logs.append(ActionLogEntry(message="Задача обработана шаблонным движком LocalScript."))
            logs.extend(self._validation_issue_logs(validation))
            if validation.is_valid:
                return LocalScriptGeneration(
                    code=validation.normalized_code,
                    validation=validation,
                    logs=logs,
                    clarification_question=None,
                    raw_response=template_code,
                    selected_strategy="template",
                    candidate_count=1,
                )
            logs.append(
                ActionLogEntry(
                    message="Шаблон не прошёл валидацию, переключаюсь на генерацию кандидатов через LLM.",
                    success=False,
                )
            )

        settings = self.settings_manager.get_settings()
        candidates: list[_Candidate] = []
        candidate_labels = self._candidate_labels(task, settings.localscript_candidate_count)
        for label in candidate_labels:
            raw_response = self._generate_once(task=task, context_messages=context_messages, strategy=label)
            validation = self.validator.validate(task, raw_response)
            candidates.append(_Candidate(label=label, raw_response=raw_response, validation=validation))
            logs.append(
                ActionLogEntry(
                    message=(
                        f"Сгенерирован кандидат LocalScript '{label}' "
                        f"(score={self.validator.score(validation, validation.normalized_code)})"
                    ),
                    success=validation.is_valid,
                )
            )

        if template_code is not None:
            template_validation = self.validator.validate(task, template_code)
            candidates.append(
                _Candidate(label="template_fallback", raw_response=template_code, validation=template_validation)
            )

        best_candidate = self._select_best_candidate(candidates)
        validation = best_candidate.validation
        raw_response = best_candidate.raw_response
        selected_strategy = best_candidate.label
        logs.append(ActionLogEntry(message=f"Выбрана стратегия LocalScript '{selected_strategy}'."))
        logs.extend(self._validation_issue_logs(validation))

        if settings.localscript_auto_validate and not validation.is_valid:
            for attempt in range(1, settings.localscript_repair_attempts + 1):
                repaired_raw = self._repair(
                    task=task,
                    candidate_code=validation.normalized_code,
                    validation=validation,
                    context_messages=context_messages,
                )
                repaired_validation = self.validator.validate(task, repaired_raw)
                repaired_score = self.validator.score(repaired_validation, repaired_validation.normalized_code)
                current_score = self.validator.score(validation, validation.normalized_code)
                if repaired_score >= current_score:
                    raw_response = repaired_raw
                    validation = repaired_validation
                    selected_strategy = f"{selected_strategy}_repair_{attempt}"
                if validation.is_valid:
                    logs.append(ActionLogEntry(message=f"Попытка исправления {attempt} дала валидный результат."))
                    break
                logs.append(
                    ActionLogEntry(
                        message=f"Попытка исправления {attempt} всё ещё не прошла валидацию.",
                        success=False,
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
        if candidate_count >= 3:
            labels.append("repair_ready")

        deduplicated: list[str] = []
        for label in labels:
            if label not in deduplicated:
                deduplicated.append(label)
        return deduplicated[: max(1, candidate_count)]

    def _generate_once(self, task: str, context_messages: Sequence[Message], strategy: str) -> str:
        settings = self.settings_manager.get_settings()
        messages = self._build_generation_messages(task, context_messages=context_messages, strategy=strategy)
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
    ) -> str:
        settings = self.settings_manager.get_settings()
        issues_text = "\n".join(f"- {issue.message}" for issue in validation.issues)
        messages = [
            {
                "role": "system",
                "content": (
                    self.knowledge_base.render_rules()
                    + "\n\nТы исправляешь ответ LocalScript. Верни только исправленный код без пояснений."
                ),
            },
            *self._context_messages(context_messages),
            {
                "role": "user",
                "content": (
                    "Исходная задача:\n"
                    f"{task}\n\n"
                    "Текущий код:\n"
                    f"{candidate_code}\n\n"
                    "Ошибки валидации:\n"
                    f"{issues_text}\n\n"
                    "Верни только исправленный LocalScript-совместимый код."
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
    ) -> list[dict[str, str]]:
        examples = self.knowledge_base.render_examples(task, limit=3)
        system_prompt = "\n\n".join(
            [
                self.knowledge_base.render_rules(),
                self._strategy_prompt(strategy),
                f"Сводка контекста:\n{self._context_summary(task)}",
                f"Релевантные примеры:\n{examples}",
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
                "Каждое исполняемое Lua-значение должно быть обёрнуто в строку lua{...}lua."
            )
        if strategy == "strict":
            return (
                "Стратегия: выбери самое короткое корректное решение LocalScript. "
                "Не придумывай переменные и поля, которых нет в контексте задачи."
            )
        if strategy == "repair_ready":
            return (
                "Стратегия: подготовь ответ, который легко проходит валидацию. "
                "Избегай placeholder-ов, markdown, комментариев и примерных литералов из prompt-а."
            )
        return "Стратегия: сгенерируй самый прямой и практичный ответ LocalScript для этой задачи."

    def _select_best_candidate(self, candidates: Sequence[_Candidate]) -> _Candidate:
        return max(
            candidates,
            key=lambda item: (
                item.validation.is_valid,
                self.validator.score(item.validation, item.validation.normalized_code),
                len(item.validation.checks),
                -len(item.validation.issues),
                -len(item.validation.normalized_code),
            ),
        )

    def _build_clarification_question(self, task: str, context_messages: Sequence[Message]) -> str | None:
        lowered = task.lower()
        has_explicit_context = any(marker in lowered for marker in ("wf.vars", "wf.initvariables", "{", "lua", "localscript"))
        has_code_context = any(self._looks_like_code(message.content) for message in context_messages[-4:])

        if has_explicit_context or has_code_context:
            return None
        if any(marker in lowered for marker in REFINE_MARKERS):
            return "Пришлите текущий LocalScript-код или JSON-контекст, который нужно исправить или доработать."
        if len(task.strip()) < 18 or len(task.split()) < 3:
            return "Уточните задачу и пришлите пример входного контекста, чтобы я сгенерировал корректный LocalScript."
        return None

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

    def _build_mode_guard_message(self, language: str) -> str:
        return (
            f"Этот чат сейчас работает в режиме генерации LocalScript/Lua. "
            f"Запрос на {language} здесь не поддерживается. "
            "Переключите чат в режим «Чат-бот», если нужен обычный ответ или код на другом языке."
        )
