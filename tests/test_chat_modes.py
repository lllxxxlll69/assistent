from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.config.settings import SettingsManager
from assistant.llm.prompts import build_chat_messages, build_system_prompt
from assistant.memory.memory_manager import MemoryManager


class ChatModeTests(unittest.TestCase):
    def test_russian_system_prompt_enforces_russian_answers(self) -> None:
        prompt = build_system_prompt("ru")
        self.assertIn("русском языке", prompt.lower())

        messages = build_chat_messages(
            system_prompt=prompt,
            context_messages=[],
            user_input="Расскажи, что ты умеешь.",
            retrieval_chunks=[],
        )
        self.assertTrue(any("только на русском языке" in item["content"].lower() for item in messages))

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


if __name__ == "__main__":
    unittest.main()
