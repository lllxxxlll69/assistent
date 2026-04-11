from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.config.settings import SettingsManager
from assistant.core.agent import Agent
from assistant.project_agent.service import ProjectAgentService
from assistant.models import Message


class StubLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **_: object) -> str:
        self.calls.append(messages)
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
                        json.dumps(
                            {
                                "ok": True,
                                "summary": "Самопроверка прошла успешно.",
                                "issues": [],
                                "fixed_content": "def main():\n    return 'ok'\n",
                            }
                        ),
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

    def test_agent_self_checks_generated_code_and_repairs_it(self) -> None:
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
                                "thought": "Обновлю основной файл и прогоню самопроверку.",
                                "reply": "Готово, добавил main после проверки.",
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
                        "def main(\n    return 'ok'\n",
                        json.dumps(
                            {
                                "ok": False,
                                "summary": "Нашёл синтаксическую ошибку и исправил её.",
                                "issues": ["Python syntax error: строка 1: '(' was never closed"],
                                "fixed_content": "def main():\n    return 'ok'\n",
                            }
                        ),
                        json.dumps(
                            {
                                "ok": True,
                                "summary": "Повторная самопроверка прошла успешно.",
                                "issues": [],
                                "fixed_content": "def main():\n    return 'ok'\n",
                            }
                        ),
                    ]
                ),
                agent=None,
            )
            progress: list[str] = []

            result = service.run(
                "Добавь main в src/app.py и пусть функция возвращает ok.",
                workspace_root=str(workspace),
                on_progress_update=progress.append,
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "def main():\n    return 'ok'\n")
            self.assertEqual(result.changed_files, ["src/app.py"])
            self.assertEqual(result.review_attempts_used, 2)
            self.assertEqual(result.unresolved_review_issues, [])
            self.assertTrue(any("самопровер" in log.message.lower() for log in result.logs))
            self.assertTrue(any("Запускаю самопроверку src/app.py" in item for item in progress))
            self.assertTrue(any("повторяю проверку" in item.lower() for item in progress))

    def test_agent_uses_current_chat_context_in_generation_and_self_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            target = workspace / "src" / "summary.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def old():\n    return 'old'\n", encoding="utf-8")
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            llm = StubLLMClient(
                [
                    json.dumps(
                        {
                            "thought": "Обновлю summary.py с учётом контекста чата.",
                            "reply": "Готово, обновил summary.py.",
                            "actions": [
                                {
                                    "type": "edit",
                                    "path": "src/summary.py",
                                    "instructions": "Используй функцию build_summary и имя отчёта sales_report.",
                                    "reason": "Это согласовано в текущем чате.",
                                }
                            ],
                        }
                    ),
                    "def build_summary():\n    return 'sales_report'\n",
                    json.dumps(
                        {
                            "ok": True,
                            "summary": "Самопроверка прошла успешно.",
                            "issues": [],
                            "fixed_content": "def build_summary():\n    return 'sales_report'\n",
                        }
                    ),
                ]
            )
            service = ProjectAgentService(manager, llm_client=llm, agent=None)

            result = service.run(
                "Обнови src/summary.py по тому, что мы уже обсудили.",
                workspace_root=str(workspace),
                context_messages=[
                    Message(role="user", content="Нужна функция build_summary."),
                    Message(role="assistant", content="Ок, использую это имя функции."),
                    Message(role="user", content="Имя отчёта должно быть sales_report."),
                ],
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "def build_summary():\n    return 'sales_report'\n")
            self.assertTrue(any("Учитываю контекст этого чата" in log.message for log in result.logs))
            self.assertGreaterEqual(len(llm.calls), 3)
            plan_prompt = llm.calls[0][-1]["content"]
            generation_prompt = llm.calls[1][1]["content"]
            self.assertIn("Контекст этого чата", plan_prompt)
            self.assertIn("Нужна функция build_summary.", plan_prompt)
            self.assertIn("Имя отчёта должно быть sales_report.", plan_prompt)
            self.assertIn("Контекст этого чата", generation_prompt)
            self.assertIn("Нужна функция build_summary.", generation_prompt)
            self.assertIn("Имя отчёта должно быть sales_report.", generation_prompt)


if __name__ == "__main__":
    unittest.main()
