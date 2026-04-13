from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assistant.config.settings import SettingsManager
from assistant.llm.client import LLMClient
from assistant.llm.prompts import build_chat_messages, build_system_prompt
from assistant.models import ActionLogEntry, Message, ProjectAgentResult, RetrievalChunk
from assistant.tools.file_tools import FileTools
from assistant.tools.search_tools import SearchTools

if TYPE_CHECKING:
    from assistant.core.agent import Agent


CODE_FENCE_RE = re.compile(r"```(?:[\w.+-]+)?\s*(.*?)```", re.DOTALL)
MAX_TREE_CHARS = 3000
MAX_RETRIEVAL_CHARS = 5000
MAX_FILE_CONTEXT_CHARS = 12000


@dataclass(slots=True)
class _PlannedAction:
    type: str
    path: str = ""
    instructions: str = ""
    reason: str = ""


@dataclass(slots=True)
class _AgentPlan:
    thought: str
    reply: str
    actions: list[_PlannedAction]


@dataclass(slots=True)
class _SelfCheckOutcome:
    ok: bool
    summary: str
    issues: list[str] = field(default_factory=list)
    fixed_content: str = ""


@dataclass(slots=True)
class _VerifiedFileContent:
    content: str
    logs: list[ActionLogEntry] = field(default_factory=list)
    review_attempts_used: int = 0
    unresolved_issues: list[str] = field(default_factory=list)


class ProjectAgentService:
    def __init__(
        self,
        settings_manager: SettingsManager,
        llm_client: LLMClient,
        agent: Any | None = None,
    ) -> None:
        self.settings_manager = settings_manager
        self.llm_client = llm_client
        self.agent = agent

    def run(
        self,
        task: str,
        *,
        workspace_root: str,
        context_messages: list[Message] | None = None,
        on_progress_update: Callable[[str], None] | None = None,
    ) -> ProjectAgentResult:
        root = Path(workspace_root).expanduser().resolve(strict=False)
        logs: list[ActionLogEntry] = []
        review_attempts_used = 0
        unresolved_review_issues: list[str] = []
        if not root.exists():
            return ProjectAgentResult(
                text="Выбранная папка проекта не существует. Укажите другую рабочую папку для режима агента.",
                logs=[ActionLogEntry(message=f"Рабочая папка не найдена: {root}", success=False)],
                workspace_root=str(root),
            )
        if not root.is_dir():
            return ProjectAgentResult(
                text="Для режима агента нужно выбрать именно папку проекта, а не файл.",
                logs=[ActionLogEntry(message=f"Ожидалась папка проекта, получен путь: {root}", success=False)],
                workspace_root=str(root),
            )

        tools = self._build_workspace_tools(str(root))
        self._push_progress(on_progress_update, f"Подключаю рабочую папку проекта: {root}")
        logs.append(ActionLogEntry(message=f"Подключена рабочая папка агента: {root}"))
        self._push_progress(on_progress_update, "Сканирую структуру проекта и собираю контекст.")
        logs.append(ActionLogEntry(message="Осматриваю структуру проекта и ищу релевантные файлы."))

        direct_action = self.agent.decide(task) if self.agent is not None else None
        if direct_action is not None and direct_action.action_type.value in {"create_file", "edit_file", "read_file", "create_folder", "search"}:
            self._push_progress(on_progress_update, "Понял прямую команду и выполняю её без промежуточного плана.")
            return self._run_direct_action(
                task,
                direct_action,
                tools,
                str(root),
                logs,
                context_messages=context_messages,
                on_progress_update=on_progress_update,
            )

        retrieval = tools["search"].retrieve_chunks(task, root)
        tree_summary = self._render_tree(root)
        context_messages = list(context_messages or [])
        relevant_context = self._select_chat_context(task, context_messages)
        if relevant_context:
            context_preview = self._context_brief(relevant_context)
            self._push_progress(on_progress_update, f"Учитываю контекст этого чата: {context_preview}")
            logs.append(ActionLogEntry(message=f"Учитываю контекст этого чата: {context_preview}"))
        if retrieval:
            paths = ", ".join(self._summarize_chunk_paths(retrieval, root))
            self._push_progress(on_progress_update, f"Нашёл релевантные файлы: {paths}.")
            self._announce_retrieval_reads(retrieval, root, on_progress_update)
        else:
            self._push_progress(on_progress_update, "Явных совпадений не нашёл, опираюсь на структуру проекта.")
        self._push_progress(on_progress_update, "Составляю план изменений по проекту.")
        plan = self._build_plan(
            task,
            workspace_root=root,
            context_messages=context_messages,
            tree_summary=tree_summary,
            retrieval=retrieval,
        )
        self._push_progress(on_progress_update, f"План готов: {plan.thought}")
        logs.append(ActionLogEntry(message=f"План агента: {plan.thought}"))

        changed_files: list[str] = []
        for item in plan.actions[:3]:
            if item.type == "answer":
                self._push_progress(on_progress_update, item.reason or "Готовлю ответ без изменения файлов.")
                logs.append(ActionLogEntry(message=item.reason or "Подготавливаю ответ без правок файлов."))
                continue

            if item.type == "create":
                if not item.path:
                    logs.append(ActionLogEntry(message="План агента не содержит путь для нового файла.", success=False))
                    continue
                self._push_progress(on_progress_update, f"Создаю новый файл {item.path}.")
                new_content = self._generate_file_content(
                    task=task,
                    action=item,
                    current_content="",
                    workspace_root=str(root),
                    tree_summary=tree_summary,
                    retrieval=retrieval,
                    context_messages=relevant_context,
                )
                verified = self._verify_generated_content(
                    task=task,
                    action=item,
                    candidate_content=new_content,
                    current_content="",
                    workspace_root=str(root),
                    tree_summary=tree_summary,
                    retrieval=retrieval,
                    context_messages=relevant_context,
                    on_progress_update=on_progress_update,
                )
                logs.extend(verified.logs)
                review_attempts_used += verified.review_attempts_used
                unresolved_review_issues.extend(verified.unresolved_issues)
                result = tools["file"].create_file(item.path, verified.content)
                logs.extend(result.logs)
                if result.logs and result.logs[-1].success:
                    self._push_progress(on_progress_update, f"Файл {item.path} создан.")
                    changed_files.append(item.path)
                continue

            if item.type == "edit":
                if not item.path:
                    logs.append(ActionLogEntry(message="План агента не содержит путь для изменения файла.", success=False))
                    continue
                self._push_progress(on_progress_update, f"Открываю файл {item.path} и подготавливаю правки.")
                read_result = tools["file"].read_file(item.path)
                logs.extend(read_result.logs)
                if read_result.logs and not read_result.logs[-1].success:
                    self._push_progress(on_progress_update, f"Файл {item.path} не найден, переключаюсь на создание.")
                    logs.append(
                        ActionLogEntry(
                            message=f"Файл {item.path} не найден, вместо редактирования создам новый файл.",
                            success=False,
                        )
                    )
                    new_content = self._generate_file_content(
                        task=task,
                        action=_PlannedAction(
                            type="create",
                            path=item.path,
                            instructions=item.instructions,
                            reason=item.reason,
                        ),
                        current_content="",
                        workspace_root=str(root),
                        tree_summary=tree_summary,
                        retrieval=retrieval,
                        context_messages=relevant_context,
                    )
                    verified = self._verify_generated_content(
                        task=task,
                        action=_PlannedAction(
                            type="create",
                            path=item.path,
                            instructions=item.instructions,
                            reason=item.reason,
                        ),
                        candidate_content=new_content,
                        current_content="",
                        workspace_root=str(root),
                        tree_summary=tree_summary,
                        retrieval=retrieval,
                        context_messages=relevant_context,
                        on_progress_update=on_progress_update,
                    )
                    logs.extend(verified.logs)
                    review_attempts_used += verified.review_attempts_used
                    unresolved_review_issues.extend(verified.unresolved_issues)
                    result = tools["file"].create_file(item.path, verified.content)
                else:
                    self._push_progress(on_progress_update, f"Прочитал файл {item.path}, готовлю точечные правки.")
                    new_content = self._generate_file_content(
                        task=task,
                        action=item,
                        current_content=read_result.content,
                        workspace_root=str(root),
                        tree_summary=tree_summary,
                        retrieval=retrieval,
                        context_messages=relevant_context,
                    )
                    verified = self._verify_generated_content(
                        task=task,
                        action=item,
                        candidate_content=new_content,
                        current_content=read_result.content,
                        workspace_root=str(root),
                        tree_summary=tree_summary,
                        retrieval=retrieval,
                        context_messages=relevant_context,
                        on_progress_update=on_progress_update,
                    )
                    logs.extend(verified.logs)
                    review_attempts_used += verified.review_attempts_used
                    unresolved_review_issues.extend(verified.unresolved_issues)
                    result = tools["file"].edit_file(item.path, verified.content)
                logs.extend(result.logs)
                if result.logs and result.logs[-1].success:
                    self._push_progress(on_progress_update, f"Изменения в {item.path} сохранены.")
                    changed_files.append(item.path)

        reply = plan.reply.strip() or "Готово. Агент обработал задачу в выбранной папке."
        if changed_files:
            changed_list = "\n".join(f"- {path}" for path in changed_files)
            reply = f"{reply}\n\nИзменённые файлы:\n{changed_list}"
        self._push_progress(on_progress_update, "Завершаю задачу и собираю итоговый ответ.")
        return ProjectAgentResult(
            text=reply,
            logs=logs,
            changed_files=changed_files,
            workspace_root=str(root),
            review_attempts_used=review_attempts_used,
            unresolved_review_issues=self._dedupe_strings(unresolved_review_issues),
        )

    def _run_direct_action(
        self,
        task: str,
        action,
        tools: dict[str, Any],
        workspace_root: str,
        logs: list[ActionLogEntry],
        *,
        context_messages: list[Message] | None = None,
        on_progress_update: Callable[[str], None] | None = None,
    ) -> ProjectAgentResult:
        changed_files: list[str] = []
        review_attempts_used = 0
        unresolved_review_issues: list[str] = []
        context_messages = list(context_messages or [])
        relevant_context = self._select_chat_context(task, context_messages)
        if relevant_context:
            context_preview = self._context_brief(relevant_context)
            self._push_progress(on_progress_update, f"Учитываю контекст этого чата: {context_preview}")
            logs.append(ActionLogEntry(message=f"Учитываю контекст этого чата: {context_preview}"))
        if action.action_type.value == "create_folder" and action.target_path:
            self._push_progress(on_progress_update, f"Создаю папку {action.target_path}.")
            result = tools["file"].create_folder(action.target_path)
            logs.append(ActionLogEntry(message="Агент создал папку по прямой команде."))
            logs.extend(result.logs)
            return ProjectAgentResult(text=result.content, logs=logs, workspace_root=workspace_root)

        if action.action_type.value == "create_file" and action.target_path:
            self._push_progress(on_progress_update, f"Создаю файл {action.target_path}.")
            inline_content = action.content or self._extract_inline_content(task)
            generated_action = _PlannedAction(type="create", path=action.target_path, instructions=task, reason="Создаю файл")
            retrieval = tools["search"].retrieve_chunks(task, Path(workspace_root))
            tree_summary = self._render_tree(Path(workspace_root))
            self._announce_retrieval_reads(retrieval, Path(workspace_root), on_progress_update)
            content = inline_content or self._generate_file_content(
                task=task,
                action=generated_action,
                current_content="",
                workspace_root=workspace_root,
                tree_summary=tree_summary,
                retrieval=retrieval,
                context_messages=relevant_context,
            )
            verified = self._verify_generated_content(
                task=task,
                action=generated_action,
                candidate_content=content,
                current_content="",
                workspace_root=workspace_root,
                tree_summary=tree_summary,
                retrieval=retrieval,
                context_messages=relevant_context,
                allow_llm_review=inline_content is None,
                on_progress_update=on_progress_update,
            )
            logs.extend(verified.logs)
            review_attempts_used += verified.review_attempts_used
            unresolved_review_issues.extend(verified.unresolved_issues)
            result = tools["file"].create_file(action.target_path, verified.content)
            logs.append(ActionLogEntry(message=f"Агент создаёт файл {action.target_path}."))
            logs.extend(result.logs)
            if result.logs and result.logs[-1].success:
                self._push_progress(on_progress_update, f"Файл {action.target_path} успешно создан.")
                changed_files.append(action.target_path)
            return ProjectAgentResult(
                text=result.content,
                logs=logs,
                changed_files=changed_files,
                workspace_root=workspace_root,
                review_attempts_used=review_attempts_used,
                unresolved_review_issues=self._dedupe_strings(unresolved_review_issues),
            )

        if action.action_type.value == "edit_file" and action.target_path:
            self._push_progress(on_progress_update, f"Открываю файл {action.target_path} для изменения.")
            read_result = tools["file"].read_file(action.target_path)
            logs.append(ActionLogEntry(message=f"Агент открывает файл {action.target_path} для обновления."))
            logs.extend(read_result.logs)
            inline_content = action.content or self._extract_inline_content(task)
            retrieval = tools["search"].retrieve_chunks(task, Path(workspace_root))
            tree_summary = self._render_tree(Path(workspace_root))
            self._announce_retrieval_reads(retrieval, Path(workspace_root), on_progress_update)
            content = inline_content
            if not content and (not read_result.logs or read_result.logs[-1].success):
                self._push_progress(on_progress_update, f"Прочитал файл {action.target_path}, формирую новую версию.")
                content = self._generate_file_content(
                    task=task,
                    action=_PlannedAction(type="edit", path=action.target_path, instructions=task, reason="Правлю файл"),
                    current_content=read_result.content,
                    workspace_root=workspace_root,
                    tree_summary=tree_summary,
                    retrieval=retrieval,
                    context_messages=relevant_context,
                )
            if not content:
                return ProjectAgentResult(
                    text="Не удалось подготовить новое содержимое файла. Уточните, что именно нужно поменять.",
                    logs=logs,
                    workspace_root=workspace_root,
                )
            verified = self._verify_generated_content(
                task=task,
                action=_PlannedAction(type="edit", path=action.target_path, instructions=task, reason="Правлю файл"),
                candidate_content=content,
                current_content=read_result.content if not read_result.logs or read_result.logs[-1].success else "",
                workspace_root=workspace_root,
                tree_summary=tree_summary,
                retrieval=retrieval,
                context_messages=relevant_context,
                allow_llm_review=inline_content is None,
                on_progress_update=on_progress_update,
            )
            logs.extend(verified.logs)
            review_attempts_used += verified.review_attempts_used
            unresolved_review_issues.extend(verified.unresolved_issues)
            result = tools["file"].edit_file(action.target_path, verified.content)
            logs.extend(result.logs)
            if result.logs and result.logs[-1].success:
                self._push_progress(on_progress_update, f"Файл {action.target_path} обновлён.")
                changed_files.append(action.target_path)
            return ProjectAgentResult(
                text=result.content,
                logs=logs,
                changed_files=changed_files,
                workspace_root=workspace_root,
                review_attempts_used=review_attempts_used,
                unresolved_review_issues=self._dedupe_strings(unresolved_review_issues),
            )

        if action.action_type.value == "read_file" and action.target_path:
            self._push_progress(on_progress_update, f"Читаю файл {action.target_path}.")
            result = tools["file"].read_file(action.target_path)
            logs.append(ActionLogEntry(message=f"Агент читает файл {action.target_path}."))
            logs.extend(result.logs)
            return ProjectAgentResult(text=result.content, logs=logs, workspace_root=workspace_root)

        if action.action_type.value == "search" and action.search_query:
            self._push_progress(on_progress_update, "Выполняю локальный поиск по проекту.")
            result = tools["search"].search_local_files(action.search_query, root=workspace_root)
            logs.append(ActionLogEntry(message="Агент выполнил локальный поиск по проекту."))
            logs.extend(result.logs)
            return ProjectAgentResult(text=result.content, logs=logs, workspace_root=workspace_root)

        return ProjectAgentResult(
            text="Агент не смог интерпретировать прямую команду. Сформулируйте задачу подробнее.",
            logs=logs,
            workspace_root=workspace_root,
        )

    def _build_plan(
        self,
        task: str,
        *,
        workspace_root: Path,
        context_messages: list[Message],
        tree_summary: str,
        retrieval: list[RetrievalChunk],
    ) -> _AgentPlan:
        rendered_tree = self._truncate_text(tree_summary, MAX_TREE_CHARS, "Структура проекта сокращена")
        rendered_retrieval = self._truncate_text(
            self._render_retrieval(retrieval, workspace_root),
            MAX_RETRIEVAL_CHARS,
            "Релевантный код сокращён",
        )
        rendered_chat_context = self._render_chat_context(self._select_chat_context(task, context_messages))
        messages = build_chat_messages(
            system_prompt=(
                build_system_prompt("ru")
                + "\n\n"
                + "Ты работаешь как локальный coding agent внутри выбранной папки проекта. "
                "Сначала продумай самые полезные действия, потом верни только JSON.\n"
                "Обязательно учитывай контекст этого же чата: предыдущие ограничения пользователя, названия файлов, договорённости по формату и уже упомянутые данные.\n"
                "Формат ответа:\n"
                "{"
                "\"thought\":\"кратко, что собираешься сделать\","
                "\"reply\":\"короткий итог для пользователя\","
                "\"actions\":["
                "{\"type\":\"edit|create|answer\",\"path\":\"relative/path.ext\",\"instructions\":\"что именно сделать\",\"reason\":\"зачем\"}"
                "]"
                "}\n"
                "Правила:\n"
                "- Только относительные пути внутри проекта.\n"
                "- Не удаляй файлы и не переименовывай папки.\n"
                "- Если запрос просит изменение кода, выбери edit.\n"
                "- Если файла ещё нет, выбери create.\n"
                "- Если правки не нужны, верни action type=answer.\n"
                "- Если запроса недостаточно для безопасной правки, не придумывай изменения: "
                "верни action type=answer и задай один короткий уточняющий вопрос в поле reply.\n"
                "- Не более 3 действий.\n"
                "- Никакого markdown."
            ),
            context_messages=context_messages[-6:],
            user_input=(
                f"Задача пользователя:\n{task}\n\n"
                f"Контекст этого чата:\n{rendered_chat_context}\n\n"
                f"Структура проекта:\n{rendered_tree}\n\n"
                f"Релевантный код:\n{rendered_retrieval}"
            ),
            retrieval_chunks=[],
        )
        raw = self.llm_client.chat(messages, stream=False)
        try:
            payload = self._parse_json(raw)
            actions_payload = payload.get("actions") if isinstance(payload, dict) else None
            actions: list[_PlannedAction] = []
            if isinstance(actions_payload, list):
                for item in actions_payload[:3]:
                    if not isinstance(item, dict):
                        continue
                    actions.append(
                        _PlannedAction(
                            type=str(item.get("type", "answer")).strip().lower(),
                            path=self._normalize_relative_path(str(item.get("path", "")).strip(), workspace_root),
                            instructions=str(item.get("instructions", "")).strip(),
                            reason=str(item.get("reason", "")).strip(),
                        )
                    )
            if not actions:
                actions = [self._fallback_action(task, retrieval, workspace_root)]
            return _AgentPlan(
                thought=str(payload.get("thought", "Разбираю проект и вношу нужные правки.")).strip(),
                reply=str(payload.get("reply", "Задача обработана в режиме агента.")).strip(),
                actions=actions,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return _AgentPlan(
                thought="Не получил чистый JSON-план от модели, перехожу к безопасному fallback-сценарию.",
                reply="Обработал задачу в безопасном режиме агента.",
                actions=[self._fallback_action(task, retrieval, workspace_root)],
            )

    def _generate_file_content(
        self,
        *,
        task: str,
        action: _PlannedAction,
        current_content: str,
        workspace_root: str,
        tree_summary: str,
        retrieval: list[RetrievalChunk],
        context_messages: list[Message],
    ) -> str:
        operation = "обнови" if action.type == "edit" else "создай"
        rendered_tree = self._truncate_text(tree_summary, MAX_TREE_CHARS, "Структура проекта сокращена")
        rendered_retrieval = self._truncate_text(
            self._render_retrieval(retrieval, Path(workspace_root)),
            MAX_RETRIEVAL_CHARS,
            "Релевантный код сокращён",
        )
        rendered_current_content = self._truncate_text(
            current_content or "<новый файл>",
            MAX_FILE_CONTEXT_CHARS,
            "Текущее содержимое файла сокращено",
        )
        rendered_chat_context = self._render_chat_context(context_messages)
        prompt = (
            f"Рабочая папка проекта: {workspace_root}\n"
            f"Нужно {operation} файл {action.path}.\n"
            f"Запрос пользователя: {task}\n"
            f"Что именно сделать: {action.instructions or task}\n\n"
            f"Контекст этого чата:\n{rendered_chat_context}\n\n"
            f"Структура проекта:\n{rendered_tree}\n\n"
            f"Релевантный код:\n{rendered_retrieval}\n\n"
            "Текущее содержимое файла:\n"
            f"{rendered_current_content}\n\n"
            "Верни только полное итоговое содержимое файла без markdown и без пояснений. "
            "Сохраняй стиль проекта и меняй только то, что нужно для задачи. "
            "Если в контексте этого чата уже были ограничения или конкретные данные, обязательно опирайся на них."
        )
        raw = self.llm_client.chat(
            [{"role": "system", "content": build_system_prompt("ru")}, {"role": "user", "content": prompt}],
            stream=False,
        )
        return self._strip_code_fences(raw).rstrip() + "\n"

    def _verify_generated_content(
        self,
        *,
        task: str,
        action: _PlannedAction,
        candidate_content: str,
        current_content: str,
        workspace_root: str,
        tree_summary: str,
        retrieval: list[RetrievalChunk],
        context_messages: list[Message],
        allow_llm_review: bool = True,
        on_progress_update: Callable[[str], None] | None = None,
    ) -> _VerifiedFileContent:
        logs: list[ActionLogEntry] = []
        current = candidate_content
        if not current:
            return _VerifiedFileContent(
                content=current,
                logs=[ActionLogEntry(message=f"Самопроверка {action.path}: модель вернула пустой кандидат.", success=False)],
                unresolved_issues=["Пустой кандидат для записи в файл."],
            )

        local_issues = self._run_local_file_checks(action.path, current)
        if not allow_llm_review:
            if local_issues:
                self._push_progress(on_progress_update, f"Нашёл локальные замечания в {action.path}.")
                logs.extend(
                    ActionLogEntry(message=f"Самопроверка {action.path}: {issue}", success=False)
                    for issue in local_issues
                )
            else:
                logs.append(ActionLogEntry(message=f"Самопроверка {action.path}: локальные проверки пройдены."))
            return _VerifiedFileContent(
                content=current,
                logs=logs,
                review_attempts_used=0,
                unresolved_issues=local_issues,
            )

        settings = self.settings_manager.get_settings()
        max_rounds = max(1, settings.agent_self_check_attempts)
        unresolved_issues: list[str] = []
        review_attempts_used = 0

        for round_index in range(1, max_rounds + 1):
            self._push_progress(
                on_progress_update,
                f"Запускаю самопроверку {action.path}: раунд {round_index}/{max_rounds}.",
            )
            local_issues = self._run_local_file_checks(action.path, current)
            review = self._run_llm_self_check(
                task=task,
                action=action,
                candidate_content=current,
                current_content=current_content,
                workspace_root=workspace_root,
                tree_summary=tree_summary,
                retrieval=retrieval,
                context_messages=context_messages,
                local_issues=local_issues,
            )
            review_attempts_used += 1

            summary = review.summary or "Самопроверка завершена."
            logs.append(ActionLogEntry(message=f"Самопроверка {action.path}: {summary}"))

            unresolved_issues = self._dedupe_strings([*local_issues, *review.issues])
            for issue in unresolved_issues[:4]:
                logs.append(ActionLogEntry(message=f"Самопроверка {action.path}: {issue}", success=False))

            next_content = review.fixed_content or current
            if review.ok and not unresolved_issues:
                self._push_progress(on_progress_update, f"Самопроверка {action.path} пройдена без замечаний.")
                return _VerifiedFileContent(
                    content=next_content,
                    logs=logs,
                    review_attempts_used=review_attempts_used,
                    unresolved_issues=[],
                )

            if next_content == current:
                self._push_progress(
                    on_progress_update,
                    f"Самопроверка {action.path} нашла проблемы, но автоматический фикс не улучшил результат.",
                )
                break

            logs.append(
                ActionLogEntry(message=f"Исправляю {action.path} по результатам самопроверки, раунд {round_index}.")
            )
            self._push_progress(
                on_progress_update,
                f"Нашёл проблемы в {action.path}, вношу исправления и повторяю проверку.",
            )
            current = next_content

        final_local_issues = self._run_local_file_checks(action.path, current)
        unresolved_issues = self._dedupe_strings([*unresolved_issues, *final_local_issues])
        if unresolved_issues:
            logs.append(
                ActionLogEntry(
                    message=(
                        f"Самопроверка {action.path}: после {review_attempts_used} раундов "
                        "остались замечания, сохраняю лучший найденный вариант."
                    ),
                    success=False,
                )
            )
            self._push_progress(
                on_progress_update,
                f"Самопроверка {action.path} завершена: лучший вариант найден, но замечания ещё остались.",
            )
        else:
            self._push_progress(on_progress_update, f"Самопроверка {action.path} завершена успешно.")

        return _VerifiedFileContent(
            content=current,
            logs=logs,
            review_attempts_used=review_attempts_used,
            unresolved_issues=unresolved_issues,
        )

    def _run_llm_self_check(
        self,
        *,
        task: str,
        action: _PlannedAction,
        candidate_content: str,
        current_content: str,
        workspace_root: str,
        tree_summary: str,
        retrieval: list[RetrievalChunk],
        context_messages: list[Message],
        local_issues: list[str],
    ) -> _SelfCheckOutcome:
        rendered_tree = self._truncate_text(tree_summary, MAX_TREE_CHARS, "Структура проекта сокращена")
        rendered_retrieval = self._truncate_text(
            self._render_retrieval(retrieval, Path(workspace_root)),
            MAX_RETRIEVAL_CHARS,
            "Релевантный код сокращён",
        )
        rendered_current_content = self._truncate_text(
            current_content or "<новый файл>",
            MAX_FILE_CONTEXT_CHARS,
            "Исходное содержимое файла сокращено",
        )
        rendered_chat_context = self._render_chat_context(context_messages)
        issues_block = "\n".join(f"- {item}" for item in local_issues) if local_issues else "- Локальные проверки не нашли ошибок."
        prompt = (
            f"Рабочая папка проекта: {workspace_root}\n"
            f"Путь файла: {action.path}\n"
            f"Тип действия: {action.type}\n"
            f"Запрос пользователя: {task}\n"
            f"Что именно нужно сделать: {action.instructions or task}\n\n"
            f"Контекст этого чата:\n{rendered_chat_context}\n\n"
            f"Структура проекта:\n{rendered_tree}\n\n"
            f"Релевантный код:\n{rendered_retrieval}\n\n"
            "Исходное содержимое файла до правки:\n"
            f"{rendered_current_content}\n\n"
            "Текущий кандидат для записи в файл:\n"
            f"{candidate_content}\n\n"
            "Результат локальных проверок:\n"
            f"{issues_block}\n\n"
            "Проведи внутренний self-check как для реально исполняемого кода. "
            "Проверь синтаксис, явные runtime-проблемы, соответствие задаче, лишние изменения и незакрытые placeholder-ы. "
            "Отдельно проверь, что кандидат не противоречит данным и договорённостям из этого чата. "
            "Если нашёл проблемы, сразу исправь их и верни полный текст файла.\n\n"
            "Верни только JSON объекта вида:\n"
            "{"
            "\"ok\":true,"
            "\"summary\":\"краткий итог проверки\","
            "\"issues\":[\"короткое замечание 1\"],"
            "\"fixed_content\":\"полный текст файла после исправлений\""
            "}\n"
            "Если кандидат уже хороший, ok=true и fixed_content должен содержать финальный полный текст файла без markdown."
        )
        raw = self.llm_client.chat(
            [{"role": "system", "content": build_system_prompt("ru")}, {"role": "user", "content": prompt}],
            stream=False,
        )
        try:
            payload = self._parse_json(raw)
        except (ValueError, TypeError, json.JSONDecodeError):
            return _SelfCheckOutcome(
                ok=not local_issues,
                summary="Не удалось разобрать ответ self-check, оставляю текущий кандидат.",
                issues=list(local_issues),
                fixed_content=candidate_content,
            )

        issues_payload = payload.get("issues")
        issues = [
            str(item).strip()
            for item in issues_payload
            if isinstance(item, str) and str(item).strip()
        ] if isinstance(issues_payload, list) else []
        fixed_content = payload.get("fixed_content")
        normalized_content = candidate_content
        if isinstance(fixed_content, str) and fixed_content.strip():
            normalized_content = self._strip_code_fences(fixed_content)
            if candidate_content.endswith("\n") and normalized_content and not normalized_content.endswith("\n"):
                normalized_content += "\n"
        summary = str(payload.get("summary", "")).strip()
        ok = bool(payload.get("ok")) and not issues
        return _SelfCheckOutcome(
            ok=ok,
            summary=summary or ("Самопроверка пройдена." if ok else "Найдены проблемы в кандидате."),
            issues=issues,
            fixed_content=normalized_content,
        )

    def _run_local_file_checks(self, path: str, content: str) -> list[str]:
        issues: list[str] = []
        if not content.strip():
            issues.append("Кандидат для записи в файл пустой.")
            return issues

        lowered = content.lower()
        placeholder_markers = (
            "todo",
            "placeholder",
            "insert code here",
            "write code here",
            "your code here",
            "your logic here",
        )
        found_placeholders = [marker for marker in placeholder_markers if marker in lowered]
        if found_placeholders:
            issues.append("В кандидате остались placeholder-маркеры: " + ", ".join(found_placeholders))

        suffix = Path(path).suffix.lower()
        if suffix == ".py":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                location = f"строка {exc.lineno}" if exc.lineno else "неизвестная строка"
                issues.append(f"Python syntax error: {location}: {exc.msg}")
        elif suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                issues.append(f"JSON syntax error: строка {exc.lineno}: {exc.msg}")

        return self._dedupe_strings(issues)

    def _build_workspace_tools(self, workspace_root: str) -> dict[str, Any]:
        settings = self.settings_manager.get_settings()
        scoped_settings = replace(settings, workspace_root=workspace_root, search_root=workspace_root)
        return {
            "file": FileTools(scoped_settings),
            "search": SearchTools(scoped_settings),
        }

    def _render_tree(self, root: Path, limit: int = 120) -> str:
        paths: list[str] = []
        for path in root.rglob("*"):
            if len(paths) >= limit:
                break
            if self._should_skip(path, root):
                continue
            relative = path.relative_to(root)
            suffix = "/" if path.is_dir() else ""
            paths.append(str(relative) + suffix)
        if not paths:
            return "(папка пуста)"
        return "\n".join(paths)

    def _render_retrieval(self, retrieval: list[RetrievalChunk], workspace_root: Path) -> str:
        if not retrieval:
            return "Совпадений пока нет."
        blocks: list[str] = []
        for chunk in retrieval[:4]:
            path = self._normalize_relative_path(chunk.path, workspace_root)
            blocks.append(f"File: {path}\nScore: {chunk.score:.3f}\n{chunk.snippet}")
        return "\n\n".join(blocks)

    def _announce_retrieval_reads(
        self,
        retrieval: list[RetrievalChunk],
        workspace_root: Path,
        on_progress_update: Callable[[str], None] | None,
    ) -> None:
        for path in self._summarize_chunk_paths(retrieval, workspace_root):
            self._push_progress(on_progress_update, f"Читаю для контекста файл {path}.")

    def _truncate_text(self, text: str, max_chars: int, label: str) -> str:
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"{text[:max_chars].rstrip()}\n\n... [{label}: скрыто ещё {omitted} символов]"

    def _fallback_action(self, task: str, retrieval: list[RetrievalChunk], workspace_root: Path) -> _PlannedAction:
        lowered = task.lower()
        if retrieval and any(marker in lowered for marker in ("измени", "поменяй", "исправ", "обнови", "refactor", "fix", "update")):
            return _PlannedAction(
                type="edit",
                path=self._relative_from_chunk(retrieval[0], workspace_root),
                instructions=task,
                reason="Нашёл самый релевантный файл и обновляю его по задаче.",
            )
        if any(marker in lowered for marker in ("создай", "добавь", "new file", "create file")):
            guessed = self._guess_file_path(task, workspace_root)
            return _PlannedAction(
                type="create",
                path=guessed,
                instructions=task,
                reason="В задаче явно просят создать новый файл.",
            )
        return _PlannedAction(type="answer", reason="Для задачи достаточно анализа проекта без записи файлов.")

    def _relative_from_chunk(self, chunk: RetrievalChunk, workspace_root: Path) -> str:
        return self._normalize_relative_path(chunk.path, workspace_root)

    def _guess_file_path(self, task: str, workspace_root: Path) -> str:
        match = re.search(r"\b([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)\b", task)
        if match:
            return self._normalize_relative_path(match.group(1), workspace_root)
        return "agent_generated.py"

    def _normalize_relative_path(self, value: str, workspace_root: Path) -> str:
        raw = value.replace("\\", "/").strip()
        if not raw:
            return ""
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            try:
                return candidate.resolve(strict=False).relative_to(workspace_root).as_posix()
            except ValueError:
                return ""
        parts = [part for part in raw.split("/") if part not in {"", ".", ".."}]
        return "/".join(parts)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON object not found.")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise TypeError("Expected JSON object.")
        return payload

    def _strip_code_fences(self, text: str) -> str:
        match = CODE_FENCE_RE.fullmatch(text.strip())
        if match:
            return match.group(1).strip()
        return text.strip()

    def _extract_inline_content(self, user_input: str) -> str | None:
        marker = "content:"
        lowered = user_input.lower()
        if marker not in lowered:
            return None
        start = lowered.index(marker) + len(marker)
        return user_input[start:].strip()

    def _should_skip(self, path: Path, root: Path) -> bool:
        skipped_parts = {".git", ".venv", "__pycache__", "node_modules", ".idea", ".assistant_data"}
        try:
            relative = path.relative_to(root)
        except ValueError:
            return True
        return any(part in skipped_parts for part in relative.parts)

    def _summarize_chunk_paths(self, retrieval: list[RetrievalChunk], workspace_root: Path) -> list[str]:
        seen: list[str] = []
        for chunk in retrieval[:4]:
            normalized = self._normalize_relative_path(chunk.path, workspace_root)
            if normalized and normalized not in seen:
                seen.append(normalized)
        return seen

    def _select_chat_context(self, task: str, context_messages: list[Message], limit: int = 6) -> list[Message]:
        if not context_messages:
            return []
        filtered = [message for message in context_messages if message.role != "system"]
        if filtered and filtered[-1].role == "user" and filtered[-1].content.strip() == task.strip():
            filtered = filtered[:-1]
        return filtered[-limit:]

    def _render_chat_context(self, context_messages: list[Message], max_chars: int = 2500) -> str:
        if not context_messages:
            return "Дополнительный контекст в этом чате пока не накоплен."

        lines: list[str] = []
        for message in context_messages:
            role = "Пользователь" if message.role == "user" else "Ассистент"
            lines.append(f"{role}: {message.content.strip()}")
        return self._truncate_text("\n".join(lines), max_chars, "Контекст чата сокращён")

    def _context_brief(self, context_messages: list[Message]) -> str:
        if not context_messages:
            return "дополнительных сообщений нет"
        last = context_messages[-1]
        snippet = " ".join(last.content.split())
        if len(snippet) > 90:
            snippet = snippet[:87].rstrip() + "..."
        role = "пользователь" if last.role == "user" else "ассистент"
        return f"{len(context_messages)} сообщ. Последнее: {role} — {snippet}"

    def _dedupe_strings(self, items: list[str]) -> list[str]:
        seen: list[str] = []
        for item in items:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen

    def _push_progress(
        self,
        callback: Callable[[str], None] | None,
        message: str,
    ) -> None:
        if callback is not None and message.strip():
            callback(message.strip())
