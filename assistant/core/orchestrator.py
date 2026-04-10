from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from assistant.config.settings import SettingsManager
from assistant.core.agent import Agent
from assistant.llm.client import LLMClient
from assistant.llm.prompts import build_auto_mode_prompt, build_chat_messages, build_system_prompt
from assistant.memory.memory_manager import MemoryManager
from assistant.models import (
    ActionLogEntry,
    ActionType,
    AssistantResponse,
    RetrievalChunk,
    VisionRequest,
)
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
    ) -> None:
        self.agent = agent
        self.settings_manager = settings_manager
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.file_tools = file_tools
        self.vision_tools = vision_tools
        self.search_tools = search_tools
        self.request_count = 0

    async def handle(self, user_input: str) -> AssistantResponse:
        return await self.handle_with_callbacks(user_input)

    async def handle_with_callbacks(
        self,
        user_input: str,
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> AssistantResponse:
        self.request_count += 1
        self.memory_manager.add_message("user", user_input)
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
                    f"Write updated file content for {action.target_path} based on: {user_input}",
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
            details = "\n".join(f"- {item}" for item in vision_result.details)
            text = vision_result.summary if not details else f"{vision_result.summary}\n{details}"
            return self._finalize_response(text, logs)

        if action.action_type == ActionType.SEARCH and action.search_query:
            result = await asyncio.to_thread(self.search_tools.search_local_files, action.search_query)
            retrieval_chunks = [RetrievalChunk(**item) for item in result.structured_data.get("chunks", [])]
            logs.extend(result.logs)
            answer = await self._generate_user_visible_response(user_input, retrieval_chunks, on_text_chunk)
            logs.append(ActionLogEntry(message="Generated response using local search context"))
            return self._finalize_response(answer, logs)

        if action.action_type == ActionType.AUTO:
            return await self._run_auto_mode(user_input)

        answer = await self._generate_user_visible_response(user_input, [], on_text_chunk)
        return self._finalize_response(answer, logs)

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
        answer = "".join(chunks).strip()
        if not answer:
            raise RuntimeError("Model returned an empty streamed response.")
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

    async def _run_auto_mode(self, task: str) -> AssistantResponse:
        plan = await asyncio.to_thread(
            self.llm_client.chat,
            [{"role": "user", "content": build_auto_mode_prompt(task)}],
        )
        logs = [ActionLogEntry(message="Created execution plan for auto mode")]
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
            answer = f"{plan}\n\nExecution finished.\nCreated {target_folder / 'main.py'}."
            return self._finalize_response(answer, logs)

        answer = f"{plan}\n\nAuto mode is ready, but no built-in executor matched this task."
        return self._finalize_response(answer, logs)

    def warm_up_models(self) -> dict[str, float]:
        settings = self.settings_manager.get_settings()
        timings: dict[str, float] = {}
        for model_name in dict.fromkeys([settings.model, settings.vision_model]):
            timings[model_name] = self.llm_client.warm_up(model_name)
        return timings

    def _finalize_response(self, text: str, logs: list[ActionLogEntry]) -> AssistantResponse:
        self.memory_manager.add_message("assistant", text)
        metrics = {
            "request_count": self.request_count,
            "estimated_tokens": max(1, len(text) // 4),
        }
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
