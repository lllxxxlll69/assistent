from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assistant.config.settings import SettingsManager
from assistant.localscript.eval_cases import get_eval_cases
from assistant.localscript.evaluator import run_eval_suite
from assistant.localscript.knowledge import LocalScriptKnowledgeBase, find_exact_prompt_overlaps
from assistant.localscript.service import LocalScriptService
from assistant.localscript.validator import LocalScriptValidator
from assistant.models import Message


LAST_EMAIL_TASK = (
    'Из полученного списка email получи последний. {"wf":{"vars":{"emails":["a","b"]}}}'
)


class StubLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requested_models: list[str] = []

    def chat(self, messages: list[dict[str, str]], **_: object) -> str:
        model = _.get("model")
        if isinstance(model, str):
            self.requested_models.append(model)
        if not self.responses:
            raise AssertionError("StubLLMClient does not have any responses left.")
        return self.responses.pop(0)

    def warm_up(self, *_: object, **__: object) -> float:
        return 0.0


class LocalScriptValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = LocalScriptValidator()

    def test_accepts_direct_wf_access(self) -> None:
        result = self.validator.validate(LAST_EMAIL_TASK, "return wf.vars.emails[#wf.vars.emails]")
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

    def test_reports_structured_check_results(self) -> None:
        result = self.validator.validate(LAST_EMAIL_TASK, "return wf.vars.emails[#wf.vars.emails]")
        self.assertTrue(any(item.name == "luac_parse" for item in result.check_results))
        self.assertIn(result.luac_status, {"passed", "skipped_with_reason"})

    def test_rejects_template_markers(self) -> None:
        task = 'Верни orderId из workflow контекста. {"wf":{"vars":{"orderId":"123"}}}'
        code = "-- insert code here\nreturn wf.vars.orderId"
        result = self.validator.validate(task, code)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule in {"no_placeholders", "no_templates"} for issue in result.issues))

    def test_rejects_randomness_when_not_requested(self) -> None:
        result = self.validator.validate("Верни число.", "return math.random()")
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule == "deterministic_output" for issue in result.issues))

    def test_rejects_print_in_judged_output(self) -> None:
        result = self.validator.validate(LAST_EMAIL_TASK, "local value = wf.vars.emails[#wf.vars.emails]\nprint(value)")
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule in {"must_return", "no_print_debug"} for issue in result.issues))

    def test_requires_rest_cleanup_source_and_pattern(self) -> None:
        task = "Очисти RESTbody result и оставь только ID, ENTITY_ID и CALL."
        code = "wf.vars.ID = nil\nwf.vars.ENTITY_ID = nil\nwf.vars.CALL = nil\nreturn wf.vars"
        result = self.validator.validate(task, code)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.rule in {"rest_result_source", "rest_cleanup_pattern"} for issue in result.issues))


class LocalScriptKnowledgeTests(unittest.TestCase):
    def test_selects_relevant_examples(self) -> None:
        knowledge = LocalScriptKnowledgeBase()
        selected = knowledge.select_examples("Из полученного списка email получи последний.", limit=2)
        self.assertGreaterEqual(len(selected), 1)
        self.assertIn("email", selected[0].prompt.lower())

    def test_generation_guidance_avoids_ready_made_code_templates(self) -> None:
        knowledge = LocalScriptKnowledgeBase()
        guidance = knowledge.render_generation_guidance("Из полученного списка email получи последний.", limit=1)
        self.assertIn("Reference 1:", guidance)
        self.assertNotIn("Expected output:", guidance)
        self.assertNotIn("return wf.vars.emails[#wf.vars.emails]", guidance)


    def test_public_examples_do_not_exactly_match_public_eval_prompts(self) -> None:
        overlaps = find_exact_prompt_overlaps([(case.id, case.prompt) for case in get_eval_cases(smoke_only=False)])
        self.assertEqual(overlaps, [])


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
            manager.update_settings(localscript_candidate_count=1, localscript_repair_attempts=0)
            service = LocalScriptService(
                settings_manager=manager,
                llm_client=StubLLMClient(["return wf.vars.emails[#wf.vars.emails]"]),
            )
            generation = service.generate(
                "Доработай, чтобы вернуть последний email.",
                context_messages=[Message(role="assistant", content="return wf.vars.emails[1]")],
                allow_clarification=True,
            )
        self.assertIsNone(generation.clarification_question)
        self.assertEqual(generation.selected_strategy, "baseline")

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
                'Используй workflow контекст и вытащи orderId. {"wf":{"vars":{"orderId":"123"}}}',
                allow_clarification=False,
            )
        self.assertEqual(generation.code, "return wf.vars.orderId")
        self.assertEqual(generation.candidate_count, 2)
        self.assertEqual(generation.selected_strategy, "strict")

    def test_judged_mode_uses_assumptions_instead_of_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(localscript_candidate_count=1, localscript_repair_attempts=0)
            service = LocalScriptService(
                settings_manager=manager,
                llm_client=StubLLMClient(['{"squared":"lua{return 25}lua"}']),
            )
            generation = service.generate(
                "Доработай и верни json payload с полем squared.",
                allow_clarification=False,
                interaction_mode="judged",
            )
        self.assertIsNone(generation.clarification_question)
        self.assertGreaterEqual(len(generation.assumptions), 1)
        self.assertTrue(any(item.stage == "assumed" for item in generation.trace))

    def test_generates_without_template_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(localscript_candidate_count=1, localscript_repair_attempts=0)
            service = LocalScriptService(
                settings_manager=manager,
                llm_client=StubLLMClient(["return wf.vars.emails[#wf.vars.emails]"]),
            )
            generation = service.generate(
                LAST_EMAIL_TASK,
                allow_clarification=False,
                interaction_mode="judged",
            )
        self.assertEqual(generation.code, "return wf.vars.emails[#wf.vars.emails]")
        self.assertEqual(generation.selected_strategy, "baseline")
        self.assertTrue(any(item.stage == "llm_cycle_started" for item in generation.trace))

    def test_judged_generation_uses_dedicated_localscript_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            manager.update_settings(
                localscript_candidate_count=1,
                localscript_repair_attempts=0,
                localscript_model="contest-model:1b",
            )
            llm = StubLLMClient(["return wf.vars.emails[#wf.vars.emails]"])
            service = LocalScriptService(settings_manager=manager, llm_client=llm)
            generation = service.generate(
                LAST_EMAIL_TASK,
                allow_clarification=False,
                interaction_mode="judged",
            )
        self.assertEqual(generation.code, "return wf.vars.emails[#wf.vars.emails]")
        self.assertEqual(llm.requested_models, ["contest-model:1b"])

    def test_generation_prompt_includes_self_check_and_reference_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = SettingsManager(Path(tmp_dir) / "settings.json")
            service = LocalScriptService(settings_manager=manager, llm_client=StubLLMClient([]))
            messages = service._build_generation_messages(
                LAST_EMAIL_TASK,
                context_messages=[],
                strategy="baseline",
                assumptions=[],
                feedback_hints=[],
            )
        system_prompt = messages[0]["content"]
        self.assertIn("Internal self-check", system_prompt)
        self.assertIn("Reference implementations for similar task shapes", system_prompt)
        self.assertIn("Expected output:", system_prompt)
        self.assertIn("return wf.vars.inboxEmails[#wf.vars.inboxEmails]", system_prompt)


class LocalScriptEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_eval_suite_produces_machine_readable_report(self) -> None:
        class _EvalStubOrchestrator:
            async def generate_localscript_response(self, prompt: str, **_: object):
                if "emails" in prompt:
                    text = "return wf.vars.emails[#wf.vars.emails]"
                elif "try_count_n" in prompt:
                    text = "return wf.vars.try_count_n + 1"
                elif "RESTbody" in prompt:
                    text = (
                        "local result = wf.vars.RESTbody.result\n"
                        "for _, filtered_entry in ipairs(result) do\n"
                        "    filtered_entry.extra = nil\n"
                        "end\n"
                        "return result"
                    )
                elif "parsedCsv" in prompt:
                    text = (
                        "local result = _utils.array.new()\n"
                        "for _, item in ipairs(wf.vars.parsedCsv) do\n"
                        "    table.insert(result, item)\n"
                        "end\n"
                        "return result"
                    )
                else:
                    text = (
                        "local timestamp = os.time({year = 2023, month = 10, day = 15, hour = 15, min = 30, sec = 0})\n"
                        "return timestamp"
                    )
                return type(
                    "StubResponse",
                    (),
                    {
                        "text": text,
                        "metrics": {
                            "validation_checks": 8,
                            "validation_errors": 0,
                            "candidate_count": 1,
                            "selected_strategy": "baseline",
                            "luac_status": "skipped_with_reason",
                            "repair_attempts_used": 0,
                            "assumptions": [],
                        },
                    },
                )()

        class _EvalStubMemory:
            def create_session(self, *_: object, **__: object) -> None:
                return None

            def add_message(self, *_: object, **__: object) -> None:
                return None

        class _EvalStubBackend:
            def __init__(self) -> None:
                self.memory_manager = _EvalStubMemory()
                self.orchestrator = _EvalStubOrchestrator()

        with patch("assistant.localscript.evaluator.build_backend", return_value=_EvalStubBackend()):
            report = await run_eval_suite(smoke_only=True)
        self.assertEqual(report["suite"], "smoke")
        self.assertEqual(report["cases_total"], 5)
        self.assertIn("selected_strategy_distribution", report)
        self.assertIn("baseline", report["selected_strategy_distribution"])
        self.assertIn("luac_status_distribution", report)


if __name__ == "__main__":
    unittest.main()
