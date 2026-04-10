from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.config.settings import SettingsManager
from assistant.localscript.knowledge import LocalScriptKnowledgeBase
from assistant.localscript.service import LocalScriptService
from assistant.localscript.templates import LocalScriptTemplateEngine
from assistant.localscript.validator import LocalScriptValidator
from assistant.models import Message


LAST_EMAIL_TASK = (
    'Из полученного списка email получи последний. {"wf":{"vars":{"emails":["a","b"]}}}'
)


class StubLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def chat(self, messages: list[dict[str, str]], **_: object) -> str:
        if not self.responses:
            raise AssertionError("StubLLMClient does not have any responses left.")
        return self.responses.pop(0)


class LocalScriptValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = LocalScriptValidator()

    def test_accepts_direct_wf_access(self) -> None:
        code = "return wf.vars.emails[#wf.vars.emails]"
        result = self.validator.validate(LAST_EMAIL_TASK, code)
        self.assertTrue(result.is_valid)

    def test_rejects_hardcoded_sample_values(self) -> None:
        task = (
            'Из полученного списка email получи последний. '
            '{"wf":{"vars":{"emails":["user1@example.com","user2@example.com"]}}}'
        )
        code = 'local emails = {"user1@example.com", "user2@example.com"}\nreturn emails[#emails]'
        result = self.validator.validate(task, code)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule == "direct_wf_access" for issue in result.issues))

    def test_requires_array_helper_when_building_result_array(self) -> None:
        task = "Отфильтруй элементы из массива parsedCsv."
        code = "local result = {}\nfor _, item in ipairs(wf.vars.parsedCsv) do\n  table.insert(result, item)\nend\nreturn result"
        result = self.validator.validate(task, code)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule == "array_constructor" for issue in result.issues))

    def test_requires_json_wrappers_for_json_result(self) -> None:
        task = "Верни JSON payload с полем squared."
        code = '{"squared": 25}'
        result = self.validator.validate(task, code)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule == "json_lua_wrappers" for issue in result.issues))

    def test_requires_init_variables_when_present_in_context(self) -> None:
        task = 'Конвертируй recallTime. {"wf":{"initVariables":{"recallTime":"2023-10-15T15:30:00+00:00"}}}'
        code = "return wf.vars.recallTime"
        result = self.validator.validate(task, code)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule == "init_variables" for issue in result.issues))


class LocalScriptKnowledgeTests(unittest.TestCase):
    def test_selects_relevant_examples(self) -> None:
        knowledge = LocalScriptKnowledgeBase()
        selected = knowledge.select_examples("Из полученного списка email получи последний.", limit=2)
        self.assertGreaterEqual(len(selected), 1)
        self.assertIn("email", selected[0].prompt.lower())

    def test_template_engine_handles_last_email_case(self) -> None:
        engine = LocalScriptTemplateEngine()
        prompt = (
            "Из полученного списка email получи последний. "
            '{"wf":{"vars":{"emails":["user1@example.com","user2@example.com","user3@example.com"]}}}'
        )
        rendered = engine.render(prompt)
        self.assertEqual(rendered, "return wf.vars.emails[#wf.vars.emails]")

    def test_template_engine_uses_array_helpers_for_items(self) -> None:
        engine = LocalScriptTemplateEngine()
        rendered = engine.render("Сделай так, чтобы все элементы items в ZCDF_PACKAGES всегда были массивами.")
        self.assertIsNotNone(rendered)
        self.assertIn("_utils.array.new()", rendered)
        self.assertIn("_utils.array.markAsArray(value)", rendered)


class LocalScriptServiceTests(unittest.TestCase):
    def test_requests_clarification_for_ambiguous_short_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            service = LocalScriptService(settings_manager=manager, llm_client=StubLLMClient([]))
            generation = service.generate("Скрипт", allow_clarification=True)
        self.assertIsNotNone(generation.clarification_question)
        self.assertEqual(generation.selected_strategy, "clarification")

    def test_uses_previous_code_context_for_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            service = LocalScriptService(
                settings_manager=manager,
                llm_client=StubLLMClient(
                    [
                        "return wf.vars.emails[#wf.vars.emails]",
                        "return wf.vars.emails[#wf.vars.emails]",
                    ]
                ),
            )
            generation = service.generate(
                "Доработай, чтобы вернуть последний email.",
                context_messages=[Message(role="assistant", content="return wf.vars.emails[1]")],
                allow_clarification=True,
            )
        self.assertIsNone(generation.clarification_question)

    def test_selects_best_candidate_from_multiple_llm_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(localscript_candidate_count=2, localscript_repair_attempts=0)
            llm = StubLLMClient(
                responses=[
                    'return "123"',
                    "return wf.vars.orderId",
                ]
            )
            service = LocalScriptService(settings_manager=manager, llm_client=llm)
            generation = service.generate(
                'Верни orderId из workflow контекста. {"wf":{"vars":{"orderId":"123"}}}',
                allow_clarification=False,
            )
        self.assertEqual(generation.code, "return wf.vars.orderId")
        self.assertEqual(generation.candidate_count, 2)


if __name__ == "__main__":
    unittest.main()
