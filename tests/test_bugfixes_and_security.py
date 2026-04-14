from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from assistant.config.settings import (
    FIXED_BATCH_SIZE,
    FIXED_CONTEXT_SIZE,
    FIXED_NUM_PREDICT,
    Settings,
    SettingsManager,
)
from assistant.core.agent import Agent
from assistant.llm.client import LLMClient
from assistant.localscript.service import LocalScriptService
from assistant.localscript.validator import LocalScriptValidator
from assistant.memory.memory_manager import MemoryManager
from assistant.models import ActionType, Message
from assistant.tools.file_tools import FileTools


class _GuardLLMClient:
    def chat(self, *_: object, **__: object) -> str:
        raise AssertionError("LLM must not be called for unsupported non-Lua requests.")


class BugfixAndSecurityTests(unittest.TestCase):
    def test_env_overrides_apply_on_first_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict("os.environ", {"ASSISTANT_MODEL": "env-model"}):
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            settings = manager.get_settings()
        self.assertEqual(settings.model, "env-model")

    def test_runtime_env_overrides_apply_on_first_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            "os.environ",
            {
                "ASSISTANT_GPU_LAYERS": "42",
                "ASSISTANT_KEEP_ALIVE": "45m",
                "ASSISTANT_LOW_VRAM": "1",
            },
        ):
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            settings = manager.get_settings()

        self.assertEqual(settings.gpu_layers, 42)
        self.assertEqual(settings.keep_alive, "45m")
        self.assertTrue(settings.low_vram)

    def test_settings_manager_enforces_fixed_runtime_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            settings = manager.update_settings(
                max_tokens=1200,
                context_size=8192,
                batch_size=7,
                localscript_context_size=2048,
                localscript_num_predict=1024,
            )

        self.assertEqual(settings.max_tokens, FIXED_NUM_PREDICT)
        self.assertEqual(settings.context_size, FIXED_CONTEXT_SIZE)
        self.assertEqual(settings.batch_size, FIXED_BATCH_SIZE)
        self.assertEqual(settings.localscript_context_size, FIXED_CONTEXT_SIZE)
        self.assertEqual(settings.localscript_num_predict, FIXED_NUM_PREDICT)

    def test_memory_summary_does_not_recurse_on_previous_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(memory_length=2, memory_max_tokens=10_000)
            memory = MemoryManager(manager, history_path=Path(tmp_dir) / "history.json")

            for index in range(6):
                memory.add_message("user", f"Запрос {index}")
                memory.add_message("assistant", f"Ответ {index}")

            summary = memory.summarize_context()

        self.assertIn("Сводка диалога", summary)
        self.assertNotIn("Ассистент: Сводка диалога", summary)

    def test_file_tools_block_access_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            outside_file = Path(tmp_dir) / "outside.txt"
            tools = FileTools(Settings(workspace_root=str(workspace)))

            result = tools.create_file(str(outside_file), "secret")

        self.assertFalse(result.logs[0].success)
        self.assertIn("Access denied outside workspace root", result.content)

    def test_localscript_service_blocks_non_lua_language_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            service = LocalScriptService(settings_manager=manager, llm_client=_GuardLLMClient())  # type: ignore[arg-type]

            generation = service.generate(
                "сгенерируй код для часов на cpp",
                context_messages=[Message(role="assistant", content="return wf.vars.timer")],
                allow_clarification=True,
            )

        self.assertEqual(generation.selected_strategy, "mode_guard")
        self.assertIn("C++", generation.code)
        self.assertIn("Чат-бот", generation.code)

    def test_validator_rejects_empty_container_instead_of_code(self) -> None:
        validator = LocalScriptValidator()

        result = validator.validate("Сгенерируй Lua-код для таймера.", "{}")

        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule in {"non_trivial_code", "standalone_table_literal"} for issue in result.issues))

    def test_streaming_preserves_newlines_and_spaces_in_code_chunks(self) -> None:
        client = LLMClient(Settings(api_url="http://127.0.0.1:11434/api/chat"))
        response = Mock()
        response.status_code = 200
        response.iter_lines.return_value = iter(
            [
                json.dumps({"message": {"content": "return"}}),
                json.dumps({"message": {"content": "\n"}}),
                json.dumps({"message": {"content": "  wf.vars.value"}}),
                json.dumps({"done": True}),
            ]
        )
        response.close = Mock()
        client._session.post = Mock(return_value=response)  # type: ignore[method-assign]

        chunks = list(client.chat_stream([{"role": "user", "content": "test"}]))

        self.assertEqual("".join(chunks), "return\n  wf.vars.value")
        response.close.assert_called_once()

    def test_vision_chat_recovers_answer_from_thinking_when_content_empty(self) -> None:
        client = LLMClient(
            Settings(
                api_url="http://127.0.0.1:11434/api/chat",
                model="qwen2.5-coder:3b",
                vision_model="qwen3-vl:4b",
            )
        )
        vision_response = Mock()
        vision_response.status_code = 200
        vision_response.json.return_value = {
            "message": {
                "content": "",
                "thinking": (
                    "Хорошо, мне нужно проанализировать изображение. "
                    "Вижу чёрный квадратный контур и красный круг в центре."
                ),
            }
        }
        summary_response = Mock()
        summary_response.status_code = 200
        summary_response.json.return_value = {
            "message": {"content": "На изображении чёрный квадратный контур с красным кругом в центре."}
        }
        client._session.post = Mock(side_effect=[vision_response, summary_response])  # type: ignore[method-assign]

        result = client.vision_chat("Что на картинке?", image_base64="abc")

        self.assertEqual(result, "На изображении чёрный квадратный контур с красным кругом в центре.")
        self.assertEqual(client._session.post.call_count, 2)  # type: ignore[union-attr]
        second_payload = client._session.post.call_args_list[1].kwargs["json"]  # type: ignore[union-attr]
        self.assertEqual(second_payload["model"], "qwen2.5-coder:3b")

    def test_vision_chat_uses_heuristic_fallback_when_summary_model_matches_vision(self) -> None:
        client = LLMClient(
            Settings(
                api_url="http://127.0.0.1:11434/api/chat",
                model="qwen3-vl:4b",
                vision_model="qwen3-vl:4b",
            )
        )
        vision_response = Mock()
        vision_response.status_code = 200
        vision_response.json.return_value = {
            "message": {
                "content": "",
                "thinking": (
                    "Хорошо, мне нужно разобрать картинку. "
                    "Вижу чёрный квадрат и красный круг в центре. "
                    "Это простой геометрический рисунок."
                ),
            }
        }
        client._session.post = Mock(return_value=vision_response)  # type: ignore[method-assign]

        result = client.vision_chat("Что на картинке?", image_base64="abc")

        self.assertIn("Вижу чёрный квадрат", result)
        self.assertNotIn("мне нужно", result.lower())

    def test_agent_strips_image_path_from_analysis_prompt(self) -> None:
        action = Agent().decide("Что изображено на фото? C:\\tmp\\photo.png")

        self.assertEqual(action.action_type, ActionType.ANALYZE_IMAGE)
        self.assertEqual(action.image_path, "C:\\tmp\\photo.png")
        self.assertEqual(action.response_text, "Что изображено на фото?")

    def test_llm_payload_carries_gpu_and_keepalive_runtime_options(self) -> None:
        client = LLMClient(
            Settings(
                api_url="http://127.0.0.1:11434/api/chat",
                gpu_layers=-1,
                main_gpu=0,
                cpu_threads=8,
                keep_alive="2h",
                low_vram=True,
            )
        )

        payload = client._build_payload(
            messages=[{"role": "user", "content": "test"}],
            model="qwen2.5-coder:7b",
            stream=False,
        )

        self.assertEqual(payload["keep_alive"], "2h")
        self.assertEqual(payload["options"]["num_gpu"], -1)
        self.assertEqual(payload["options"]["main_gpu"], 0)
        self.assertEqual(payload["options"]["num_thread"], 8)
        self.assertTrue(payload["options"]["low_vram"])

    def test_llm_payload_enforces_fixed_contest_limits(self) -> None:
        client = LLMClient(
            Settings(
                api_url="http://127.0.0.1:11434/api/chat",
                max_tokens=4096,
                context_size=16384,
                batch_size=9,
            )
        )

        payload = client._build_payload(
            messages=[{"role": "user", "content": "test"}],
            model="qwen2.5-coder:7b",
            stream=False,
            max_tokens_override=32,
            context_size_override=2048,
        )

        self.assertEqual(payload["options"]["num_predict"], FIXED_NUM_PREDICT)
        self.assertEqual(payload["options"]["num_ctx"], FIXED_CONTEXT_SIZE)
        self.assertEqual(payload["options"]["num_batch"], FIXED_BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
