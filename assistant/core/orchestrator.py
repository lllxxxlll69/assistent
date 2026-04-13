from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from assistant.config.settings import SettingsManager
from assistant.core.agent import Agent
from assistant.core.clarification import ClarificationHelper
from assistant.llm.client import LLMClient
from assistant.llm.prompts import build_auto_mode_prompt, build_chat_messages, build_system_prompt
from assistant.localscript.service import LocalScriptService
from assistant.memory.memory_manager import MemoryManager
from assistant.models import (
    ActionLogEntry,
    ActionType,
    AssistantResponse,
    RetrievalChunk,
    VisionRequest,
)
from assistant.project_agent.service import ProjectAgentService
from assistant.tools.file_tools import FileTools
from assistant.tools.search_tools import SearchTools
from assistant.tools.vision_tools import VisionTools


class Orchestrator:
    def __init__(
        self,
        agent: Agent,
        settings_manager: SettingsManager,
        memory_manager: MemoryManager,
        llm_client: LLMClient,
        file_tools: FileTools,
        vision_tools: VisionTools,
        search_tools: SearchTools,
        localscript_service: LocalScriptService,
        project_agent_service: ProjectAgentService,
    ) -> None:
        self.agent = agent
        self.settings_manager = settings_manager
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.file_tools = file_tools
        self.vision_tools = vision_tools
        self.search_tools = search_tools
        self.localscript_service = localscript_service
        self.project_agent_service = project_agent_service
        self.clarification_helper = ClarificationHelper()
        self.request_count = 0

    async def handle(self, user_input: str, *, assistant_profile: str | None = None) -> AssistantResponse:
        return await self.handle_with_callbacks(user_input, assistant_profile=assistant_profile)

    async def handle_with_callbacks(
        self,
        user_input: str,
        on_text_chunk: Callable[[str], None] | None = None,
        on_status_update: Callable[[str], None] | None = None,
        assistant_profile: str | None = None,
    ) -> AssistantResponse:
        self.request_count += 1
        previous_context = self.memory_manager.get_context()
        self.memory_manager.add_message("user", user_input)
        if self._is_agent_profile(assistant_profile):
            clarification = self.clarification_helper.for_agent(user_input, previous_context)
            if clarification.should_ask:
                return self._finalize_response(
                    clarification.question,
                    [ActionLogEntry(message=clarification.reason)],
                )
            return await self.generate_project_agent_response(
                user_input,
                persist_memory=True,
                user_already_recorded=True,
                use_memory_context=True,
                count_request=False,
                on_status_update=on_status_update,
            )
        action = self.agent.decide(user_input)
        logs: list[ActionLogEntry] = []

        if action.action_type == ActionType.CREATE_FOLDER and action.target_path:
            result = await asyncio.to_thread(self.file_tools.create_folder, action.target_path)
            return self._finalize_response(result.content, result.logs)

        if action.action_type == ActionType.CREATE_FILE and action.target_path:
            content = action.content or self._extract_inline_content(user_input) or "# New file\n"
            result = await asyncio.to_thread(self.file_tools.create_file, action.target_path, content)
            return self._finalize_response(result.content, result.logs)

        if action.action_type == ActionType.EDIT_FILE and action.target_path:
            content = action.content or self._extract_inline_content(user_input)
            if not content:
                content = await asyncio.to_thread(
                    self._generate_text_response,
                    f"Подготовь обновленное содержимое файла {action.target_path} на основе запроса: {user_input}",
                    [],
                )
            result = await asyncio.to_thread(self.file_tools.edit_file, action.target_path, content)
            return self._finalize_response(result.content, result.logs)

        if action.action_type == ActionType.READ_FILE and action.target_path:
            result = await asyncio.to_thread(self.file_tools.read_file, action.target_path)
            return self._finalize_response(result.content, result.logs)

        if action.action_type == ActionType.ANALYZE_IMAGE and action.image_path:
            vision_result = await asyncio.to_thread(
                self.vision_tools.analyze_image,
                VisionRequest(image_path=action.image_path, prompt=user_input),
            )
            logs.append(ActionLogEntry(message=f"Analyzed image {action.image_path}"))
            logs[-1].message = f"Проанализировано изображение {action.image_path}"
            details = "\n".join(f"- {item}" for item in vision_result.details)
            text = vision_result.summary if not details else f"{vision_result.summary}\n{details}"
            return self._finalize_response(text, logs)

        if action.action_type == ActionType.SEARCH and action.search_query:
            result = await asyncio.to_thread(self.search_tools.search_local_files, action.search_query)
            retrieval_chunks = [RetrievalChunk(**item) for item in result.structured_data.get("chunks", [])]
            logs.extend(result.logs)
            if self._should_use_localscript_pipeline(user_input, assistant_profile=assistant_profile):
                generation = await asyncio.to_thread(
                    self.localscript_service.generate,
                    user_input,
                    context_messages=self.memory_manager.get_context(),
                    interaction_mode="interactive",
                )
                logs.extend(generation.logs)
                logs.extend(self._validation_logs(generation.validation.is_valid, generation.validation.issues))
                return self._finalize_response(
                    generation.code,
                    logs,
                    extra_metrics=self._localscript_metrics(generation),
                )

            answer = await self._generate_user_visible_response(user_input, retrieval_chunks, on_text_chunk)
            logs.append(ActionLogEntry(message="Сформирован ответ с учетом локального поиска"))
            return self._finalize_response(answer, logs)

        if action.action_type == ActionType.AUTO:
            return await self._run_auto_mode(user_input, assistant_profile=assistant_profile)

        if self._should_use_localscript_pipeline(user_input, assistant_profile=assistant_profile):
            return await self.generate_localscript_response(
                user_input,
                allow_clarification=True,
                persist_memory=True,
                user_already_recorded=True,
                use_memory_context=True,
                count_request=False,
            )

        if action.action_type == ActionType.RESPOND:
            clarification = self.clarification_helper.for_chat(user_input, previous_context)
            if clarification.should_ask:
                return self._finalize_response(
                    clarification.question,
                    [ActionLogEntry(message=clarification.reason)],
                )

        answer = await self._generate_user_visible_response(user_input, [], on_text_chunk)
        return self._finalize_response(answer, logs)

    async def generate_localscript_response(
        self,
        user_input: str,
        *,
        allow_clarification: bool = False,
        persist_memory: bool = False,
        user_already_recorded: bool = False,
        use_memory_context: bool = False,
        count_request: bool = True,
    ) -> AssistantResponse:
        if count_request:
            self.request_count += 1

        if persist_memory and not user_already_recorded:
            self.memory_manager.add_message("user", user_input)

        context_messages = self.memory_manager.get_context() if use_memory_context else []
        generation = await asyncio.to_thread(
            self.localscript_service.generate,
            user_input,
            context_messages=context_messages,
            allow_clarification=allow_clarification,
            interaction_mode="interactive" if allow_clarification else "judged",
        )
        logs = list(generation.logs)
        logs.extend(self._validation_logs(generation.validation.is_valid, generation.validation.issues))
        text = generation.clarification_question or generation.code
        return self._finalize_response(
            text,
            logs,
            persist_memory=persist_memory,
            extra_metrics=self._localscript_metrics(generation),
        )

    async def generate_project_agent_response(
        self,
        user_input: str,
        *,
        persist_memory: bool = False,
        user_already_recorded: bool = False,
        use_memory_context: bool = False,
        count_request: bool = True,
        on_status_update: Callable[[str], None] | None = None,
    ) -> AssistantResponse:
        if count_request:
            self.request_count += 1

        if persist_memory and not user_already_recorded:
            self.memory_manager.add_message("user", user_input)

        workspace_root = self.memory_manager.get_active_workspace_root()
        if not workspace_root:
            return self._finalize_response(
                "Для режима агента сначала выберите рабочую папку проекта.",
                [ActionLogEntry(message="Не выбрана рабочая папка агента.", success=False)],
                persist_memory=persist_memory,
                extra_metrics={"agent_changed_files": 0},
            )

        context_messages = self.memory_manager.get_context() if use_memory_context else []
        result = await asyncio.to_thread(
            self.project_agent_service.run,
            user_input,
            workspace_root=workspace_root,
            context_messages=context_messages,
            on_progress_update=on_status_update,
        )
        return self._finalize_response(
            result.text,
            result.logs,
            persist_memory=persist_memory,
            extra_metrics={
                "agent_changed_files": len(result.changed_files),
                "agent_workspace": result.workspace_root,
                "agent_review_attempts_used": result.review_attempts_used,
                "agent_unresolved_review_issues": list(result.unresolved_review_issues),
            },
        )

    def _generate_text_response(self, user_input: str, retrieval_chunks: list[RetrievalChunk]) -> str:
        settings = self.settings_manager.get_settings()
        messages = build_chat_messages(
            system_prompt=build_system_prompt(settings.system_prompt_language),
            context_messages=self.memory_manager.get_context(),
            user_input=user_input,
            retrieval_chunks=retrieval_chunks,
        )
        return self.llm_client.chat(
            messages,
            model=settings.model,
            stream=False,
            max_tokens_override=self._select_generation_limit(user_input, retrieval_chunks),
        )

    def _generate_streamed_text_response(
        self,
        user_input: str,
        retrieval_chunks: list[RetrievalChunk],
        on_text_chunk: Callable[[str], None],
    ) -> str:
        settings = self.settings_manager.get_settings()
        messages = build_chat_messages(
            system_prompt=build_system_prompt(settings.system_prompt_language),
            context_messages=self.memory_manager.get_context(),
            user_input=user_input,
            retrieval_chunks=retrieval_chunks,
        )
        chunks: list[str] = []
        for chunk in self.llm_client.chat_stream(
            messages,
            model=settings.model,
            max_tokens_override=self._select_generation_limit(user_input, retrieval_chunks),
        ):
            chunks.append(chunk)
            on_text_chunk(chunk)
        answer = "".join(chunks)
        if not answer.strip():
            raise RuntimeError("Модель вернула пустой потоковый ответ.")
        return answer

    async def _generate_user_visible_response(
        self,
        user_input: str,
        retrieval_chunks: list[RetrievalChunk],
        on_text_chunk: Callable[[str], None] | None,
    ) -> str:
        settings = self.settings_manager.get_settings()
        if settings.stream and on_text_chunk is not None:
            return await asyncio.to_thread(
                self._generate_streamed_text_response,
                user_input,
                retrieval_chunks,
                on_text_chunk,
            )
        return await asyncio.to_thread(self._generate_text_response, user_input, retrieval_chunks)

    async def _run_auto_mode(self, task: str, *, assistant_profile: str | None = None) -> AssistantResponse:
        if self._should_use_localscript_pipeline(task, assistant_profile=assistant_profile):
            generation = await asyncio.to_thread(
                self.localscript_service.generate,
                task,
                context_messages=self.memory_manager.get_context(),
                allow_clarification=False,
                interaction_mode="judged",
            )
            logs = [ActionLogEntry(message="Выполнен автоматический LocalScript-пайплайн")]
            logs.extend(generation.logs)
            logs.extend(self._validation_logs(generation.validation.is_valid, generation.validation.issues))
            return self._finalize_response(
                generation.code,
                logs,
                extra_metrics=self._localscript_metrics(generation),
            )

        plan = await asyncio.to_thread(
            self.llm_client.chat,
            [{"role": "user", "content": build_auto_mode_prompt(task)}],
        )
        logs = [ActionLogEntry(message="Сформирован план для автоматического режима")]
        lower_task = task.lower()

        if "rest api" in lower_task or "api" in lower_task:
            target_folder = Path("generated_api")
            mkdir_result = await asyncio.to_thread(self.file_tools.create_folder, str(target_folder))
            main_result = await asyncio.to_thread(
                self.file_tools.create_file,
                str(target_folder / "main.py"),
                self._default_rest_api_content(),
            )
            logs.extend(mkdir_result.logs)
            logs.extend(main_result.logs)
            answer = f"{plan}\n\nВыполнение завершено.\nСоздан файл {target_folder / 'main.py'}."
            return self._finalize_response(answer, logs)

        answer = f"{plan}\n\nАвто-режим подготовлен, но для этой задачи пока нет встроенного исполнителя."
        return self._finalize_response(answer, logs)

    def warm_up_models(self) -> dict[str, float]:
        settings = self.settings_manager.get_settings()
        timings: dict[str, float] = {}
        for model_name in dict.fromkeys([settings.model, settings.vision_model]):
            timings[model_name] = self.llm_client.warm_up(model_name)
        return timings

    def _finalize_response(
        self,
        text: str,
        logs: list[ActionLogEntry],
        *,
        persist_memory: bool = True,
        extra_metrics: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        if persist_memory:
            self.memory_manager.add_message("assistant", text)
        metrics = {
            "request_count": self.request_count,
            "estimated_tokens": max(1, len(text) // 4),
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        return AssistantResponse(text=text, logs=logs, metrics=metrics)

    def _extract_inline_content(self, user_input: str) -> str | None:
        marker = "content:"
        lowered = user_input.lower()
        if marker not in lowered:
            return None
        start = lowered.index(marker) + len(marker)
        return user_input[start:].strip()

    def _default_rest_api_content(self) -> str:
        return (
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self) -> None:\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Type', 'application/json')\n"
            "        self.end_headers()\n"
            "        self.wfile.write(b'{\"status\": \"ok\"}')\n\n"
            "if __name__ == '__main__':\n"
            "    server = HTTPServer(('127.0.0.1', 8000), Handler)\n"
            "    print('Serving on http://127.0.0.1:8000')\n"
            "    server.serve_forever()\n"
        )

    def _select_generation_limit(self, user_input: str, retrieval_chunks: list[RetrievalChunk]) -> int:
        settings = self.settings_manager.get_settings()
        lowered = user_input.lower()
        is_vision_model = "vl" in settings.model.lower()

        if any(marker in lowered for marker in ("подроб", "деталь", "пошаг", "код", "json", "api", "файл")):
            return min(settings.max_tokens, 900)
        if retrieval_chunks:
            return min(settings.max_tokens, 448 if is_vision_model else 384)
        if len(user_input) < 80:
            return min(settings.max_tokens, 256 if is_vision_model else 192)
        if len(user_input) < 240:
            return min(settings.max_tokens, 384 if is_vision_model else 320)
        return min(settings.max_tokens, 640 if is_vision_model else 512)

    def _should_use_localscript_pipeline(self, user_input: str, *, assistant_profile: str | None = None) -> bool:
        settings = self.settings_manager.get_settings()
        profile = (assistant_profile or settings.assistant_profile).strip().lower()
        if profile != "localscript":
            return False

        lowered = user_input.lower()
        generic_meta_markers = (
            "openapi",
            "docker",
            "model",
            "модель",
            "настройк",
            "лог",
            "benchmark",
            "self-check",
            "self check",
            "оценк",
            "балл",
            "что умеешь",
            "как запустить",
        )
        if any(marker in lowered for marker in generic_meta_markers):
            return False
        return True

    def _is_agent_profile(self, assistant_profile: str | None = None) -> bool:
        settings = self.settings_manager.get_settings()
        profile = (assistant_profile or settings.assistant_profile).strip().lower()
        return profile == "agent"

    def _validation_logs(self, is_valid: bool, issues: list[object]) -> list[ActionLogEntry]:
        if is_valid:
            return [ActionLogEntry(message="Валидация LocalScript пройдена")]

        logs = [ActionLogEntry(message="Валидация LocalScript не пройдена", success=False)]
        for issue in issues:
            message = getattr(issue, "message", str(issue))
            logs.append(ActionLogEntry(message=message, success=False))
        return logs

    def _localscript_metrics(self, generation) -> dict[str, Any]:
        return {
            "validation_checks": len(generation.validation.checks),
            "validation_errors": len(generation.validation.issues),
            "candidate_count": generation.candidate_count,
            "selected_strategy": generation.selected_strategy,
            "luac_status": generation.validation.luac_status,
            "repair_attempts_used": generation.repair_attempts_used,
            "assumptions": list(generation.assumptions),
            "trace": [f"{item.stage}:{item.status}" for item in generation.trace],
            "strategy_distribution": {
                item.source: sum(1 for candidate in generation.candidate_reports if candidate.source == item.source)
                for item in generation.candidate_reports
            },
            "runtime_info": dict(generation.runtime_info),
        }
