from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.config.settings import SettingsManager
from assistant.core.agent import Agent
from assistant.project_agent.service import ProjectAgentService


class StubLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def chat(self, messages: list[dict[str, str]], **_: object) -> str:
        if not self.responses:
            raise AssertionError(f"Unexpected chat call without prepared response. Messages: {messages!r}")
        return self.responses.pop(0)


class ProjectAgentServiceTests(unittest.TestCase):
    def test_agent_creates_file_inside_selected_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            service = ProjectAgentService(manager, llm_client=StubLLMClient([]), agent=Agent())

            result = service.run(
                "create file src/app.py content:print('agent ok')",
                workspace_root=str(workspace),
            )

            created_file = workspace / "src" / "app.py"
            self.assertTrue(created_file.exists())
            self.assertEqual(created_file.read_text(encoding="utf-8"), "print('agent ok')")
            self.assertEqual(result.changed_files, ["src/app.py"])
            self.assertIn("File created", result.text)

    def test_agent_edits_existing_file_from_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            target = workspace / "src" / "app.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def old_value():\n    return 'old'\n", encoding="utf-8")
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            service = ProjectAgentService(
                manager,
                llm_client=StubLLMClient(
                    [
                        json.dumps(
                            {
                                "thought": "Нашёл основной файл и обновляю его.",
                                "reply": "Готово, обновил основной модуль.",
                                "actions": [
                                    {
                                        "type": "edit",
                                        "path": "src/app.py",
                                        "instructions": "Добавь функцию main, которая возвращает 'ok'.",
                                        "reason": "Это основной файл проекта.",
                                    }
                                ],
                            }
                        ),
                        "def main():\n    return 'ok'\n",
                    ]
                ),
                agent=None,
            )

            result = service.run(
                "Добавь main в src/app.py и пусть функция возвращает ok.",
                workspace_root=str(workspace),
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "def main():\n    return 'ok'\n")
            self.assertEqual(result.changed_files, ["src/app.py"])
            self.assertIn("src/app.py", result.text)
            self.assertTrue(any("План агента" in log.message for log in result.logs))

    def test_agent_reports_progress_during_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            service = ProjectAgentService(
                manager,
                llm_client=StubLLMClient(
                    [
                        json.dumps(
                            {
                                "thought": "Сначала посмотрю структуру, потом дам ответ без правок.",
                                "reply": "Пока достаточно анализа.",
                                "actions": [{"type": "answer", "reason": "Для этой задачи правки не нужны."}],
                            }
                        )
                    ]
                ),
                agent=None,
            )
            progress: list[str] = []

            service.run(
                "Проанализируй проект и скажи, что ты будешь делать.",
                workspace_root=str(workspace),
                on_progress_update=progress.append,
            )

            self.assertTrue(any("Подключаю рабочую папку" in item for item in progress))
            self.assertTrue(any("Составляю план изменений" in item for item in progress))
            self.assertTrue(any("План готов" in item for item in progress))


if __name__ == "__main__":
    unittest.main()
