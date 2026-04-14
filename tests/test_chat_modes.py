from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from assistant.config.settings import SettingsManager
from assistant.core.clarification import ClarificationHelper
from assistant.core.orchestrator import Orchestrator, _unwrap_natural_language_return
from assistant.llm.prompts import build_chat_messages, build_system_prompt
from assistant.memory.memory_manager import MemoryManager


class ChatModeTests(unittest.TestCase):
    def test_unwraps_natural_language_return_into_plain_text(self) -> None:
        wrapped = 'return "Я могу помочь с информацией, отвечать на вопросы, помогать с кодом и выполнять задачи в рамках предоставленного контекста."'

        result = _unwrap_natural_language_return(wrapped)

        self.assertEqual(
            result,
            "Я могу помочь с информацией, отвечать на вопросы, помогать с кодом и выполнять задачи в рамках предоставленного контекста.",
        )

    def test_keeps_short_string_literal_code_unchanged(self) -> None:
        wrapped = 'return "ok"'

        result = _unwrap_natural_language_return(wrapped)

        self.assertEqual(result, wrapped)

    def test_russian_system_prompt_enforces_russian_answers(self) -> None:
        prompt = build_system_prompt("ru")
        self.assertIn("русском языке", prompt.lower())
        self.assertIn("наводящий вопрос", prompt.lower())

        messages = build_chat_messages(
            system_prompt=prompt,
            context_messages=[],
            user_input="Расскажи, что ты умеешь.",
            retrieval_chunks=[],
        )
        self.assertTrue(any("только на русском языке" in item["content"].lower() for item in messages))
        self.assertTrue(any("наводящий вопрос" in item["content"].lower() for item in messages))

    def test_clarification_helper_asks_one_guiding_question_for_vague_chat_request(self) -> None:
        decision = ClarificationHelper().for_chat("Сделай это лучше", [])

        self.assertTrue(decision.should_ask)
        self.assertIn("Что именно", decision.question)

    def test_clarification_helper_allows_project_analysis_in_agent_mode(self) -> None:
        decision = ClarificationHelper().for_agent("Проанализируй весь проект и найди узкие места.", [])

        self.assertFalse(decision.should_ask)

    def test_session_mode_is_saved_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            history_path = Path(tmp_dir) / "history.json"

            manager = SettingsManager(settings_path)
            memory = MemoryManager(manager, history_path=history_path)
            session = memory.get_current_session()
            self.assertEqual(session.assistant_mode, "localscript")

            self.assertTrue(memory.set_session_mode(session.id, "chat"))
            self.assertEqual(memory.get_active_session_mode(), "chat")

            reloaded = MemoryManager(manager, history_path=history_path)
            self.assertEqual(reloaded.get_active_session_mode(), "chat")

    def test_agent_mode_and_workspace_are_saved_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            history_path = Path(tmp_dir) / "history.json"
            workspace_path = str(Path(tmp_dir) / "workspace")

            manager = SettingsManager(settings_path)
            memory = MemoryManager(manager, history_path=history_path)
            session = memory.get_current_session()

            self.assertTrue(memory.set_session_workspace_root(session.id, workspace_path))
            self.assertTrue(memory.set_session_mode(session.id, "agent"))
            self.assertEqual(memory.get_active_session_mode(), "agent")
            self.assertEqual(memory.get_active_workspace_root(), workspace_path)

            reloaded = MemoryManager(manager, history_path=history_path)
            self.assertEqual(reloaded.get_active_session_mode(), "agent")
            self.assertEqual(reloaded.get_active_workspace_root(), workspace_path)

    def test_warm_up_models_uses_chat_model_for_chat_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(model="chat-model:3b", localscript_model="lua-model:7b")
            calls: list[str] = []
            llm_client = SimpleNamespace(warm_up=lambda model_name: calls.append(model_name) or 0.25)
            orchestrator = Orchestrator(
                agent=SimpleNamespace(),
                settings_manager=manager,
                memory_manager=SimpleNamespace(),
                llm_client=llm_client,
                file_tools=SimpleNamespace(),
                vision_tools=SimpleNamespace(),
                search_tools=SimpleNamespace(),
                localscript_service=SimpleNamespace(),
                project_agent_service=SimpleNamespace(),
            )

            timings = orchestrator.warm_up_models("chat")

        self.assertEqual(calls, ["chat-model:3b"])
        self.assertEqual(timings, {"chat-model:3b": 0.25})

    def test_warm_up_models_uses_localscript_model_for_localscript_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(model="chat-model:3b", localscript_model="lua-model:7b")
            calls: list[str] = []
            llm_client = SimpleNamespace(warm_up=lambda model_name: calls.append(model_name) or 0.4)
            orchestrator = Orchestrator(
                agent=SimpleNamespace(),
                settings_manager=manager,
                memory_manager=SimpleNamespace(),
                llm_client=llm_client,
                file_tools=SimpleNamespace(),
                vision_tools=SimpleNamespace(),
                search_tools=SimpleNamespace(),
                localscript_service=SimpleNamespace(),
                project_agent_service=SimpleNamespace(),
            )

            timings = orchestrator.warm_up_models("localscript")

        self.assertEqual(calls, ["lua-model:7b"])
        self.assertEqual(timings, {"lua-model:7b": 0.4})

    def test_warm_up_models_falls_back_to_chat_model_when_localscript_model_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(model="chat-model:3b", localscript_model="lua-model:7b")
            calls: list[str] = []
            llm_client = SimpleNamespace(warm_up=lambda model_name: calls.append(model_name) or 0.6)
            orchestrator = Orchestrator(
                agent=SimpleNamespace(),
                settings_manager=manager,
                memory_manager=SimpleNamespace(),
                llm_client=llm_client,
                file_tools=SimpleNamespace(),
                vision_tools=SimpleNamespace(),
                search_tools=SimpleNamespace(),
                localscript_service=SimpleNamespace(),
                project_agent_service=SimpleNamespace(),
            )

            with patch(
                "assistant.core.orchestrator.list_installed_ollama_models",
                return_value=[SimpleNamespace(name="chat-model:3b")],
            ):
                timings = orchestrator.warm_up_models("localscript")

        self.assertEqual(calls, ["chat-model:3b"])
        self.assertEqual(timings, {"chat-model:3b": 0.6})

    def test_warm_up_specific_models_deduplicates_requested_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            calls: list[str] = []
            llm_client = SimpleNamespace(warm_up=lambda model_name: calls.append(model_name) or 0.3)
            orchestrator = Orchestrator(
                agent=SimpleNamespace(),
                settings_manager=manager,
                memory_manager=SimpleNamespace(),
                llm_client=llm_client,
                file_tools=SimpleNamespace(),
                vision_tools=SimpleNamespace(),
                search_tools=SimpleNamespace(),
                localscript_service=SimpleNamespace(),
                project_agent_service=SimpleNamespace(),
            )

            timings = orchestrator.warm_up_specific_models(["chat-model:3b", "chat-model:3b", "vision-model:4b"])

        self.assertEqual(calls, ["chat-model:3b", "vision-model:4b"])
        self.assertEqual(timings, {"chat-model:3b": 0.3, "vision-model:4b": 0.3})

    def test_resolve_used_warm_up_models_returns_all_configured_installed_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(
                model="chat-model:3b",
                localscript_model="lua-model:7b",
                vision_model="vision-model:4b",
            )
            orchestrator = Orchestrator(
                agent=SimpleNamespace(),
                settings_manager=manager,
                memory_manager=SimpleNamespace(),
                llm_client=SimpleNamespace(),
                file_tools=SimpleNamespace(),
                vision_tools=SimpleNamespace(),
                search_tools=SimpleNamespace(),
                localscript_service=SimpleNamespace(),
                project_agent_service=SimpleNamespace(),
            )

            targets = orchestrator.resolve_used_warm_up_models(
                ["chat-model:3b", "lua-model:7b", "vision-model:4b"]
            )

        self.assertEqual(targets, ["chat-model:3b", "lua-model:7b", "vision-model:4b"])

    def test_resolve_post_install_warm_up_models_returns_only_configured_installed_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(
                model="chat-model:3b",
                localscript_model="lua-model:7b",
                vision_model="vision-model:4b",
            )
            orchestrator = Orchestrator(
                agent=SimpleNamespace(),
                settings_manager=manager,
                memory_manager=SimpleNamespace(),
                llm_client=SimpleNamespace(),
                file_tools=SimpleNamespace(),
                vision_tools=SimpleNamespace(),
                search_tools=SimpleNamespace(),
                localscript_service=SimpleNamespace(),
                project_agent_service=SimpleNamespace(),
            )

            with patch(
                "assistant.core.orchestrator.list_installed_ollama_models",
                return_value=[
                    SimpleNamespace(name="other-model:1b"),
                    SimpleNamespace(name="vision-model:4b"),
                    SimpleNamespace(name="chat-model:3b"),
                ],
            ):
                targets = orchestrator.resolve_post_install_warm_up_models(
                    ["other-model:1b", "vision-model:4b", "chat-model:3b"]
                )

        self.assertEqual(targets, ["chat-model:3b", "vision-model:4b"])

    def test_resolve_post_install_warm_up_models_uses_full_installed_inventory_after_partial_reinstall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(
                model="chat-model:3b",
                localscript_model="lua-model:7b",
                vision_model="vision-model:4b",
            )
            orchestrator = Orchestrator(
                agent=SimpleNamespace(),
                settings_manager=manager,
                memory_manager=SimpleNamespace(),
                llm_client=SimpleNamespace(),
                file_tools=SimpleNamespace(),
                vision_tools=SimpleNamespace(),
                search_tools=SimpleNamespace(),
                localscript_service=SimpleNamespace(),
                project_agent_service=SimpleNamespace(),
            )

            with patch(
                "assistant.core.orchestrator.list_installed_ollama_models",
                return_value=[
                    SimpleNamespace(name="chat-model:3b"),
                    SimpleNamespace(name="vision-model:4b"),
                ],
            ):
                targets = orchestrator.resolve_post_install_warm_up_models(["chat-model:3b"])

        self.assertEqual(targets, ["chat-model:3b", "vision-model:4b"])


if __name__ == "__main__":
    unittest.main()
