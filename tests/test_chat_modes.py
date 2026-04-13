from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.config.settings import SettingsManager
from assistant.core.clarification import ClarificationHelper
from assistant.llm.prompts import build_chat_messages, build_system_prompt
from assistant.memory.memory_manager import MemoryManager


class ChatModeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
