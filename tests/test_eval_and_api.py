from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import request as urllib_request

from assistant.api.server import LocalScriptAPIHandler
from assistant.config.settings import SettingsManager
from assistant.localscript.self_check import run_self_check
from assistant.models import AssistantResponse


class _StubOrchestrator:
    async def generate_localscript_response(self, *_: object, **__: object) -> AssistantResponse:
        return AssistantResponse(
            text="return wf.vars.orderId",
            metrics={
                "validation_checks": 8,
                "validation_errors": 0,
                "candidate_count": 1,
                "selected_strategy": "template",
                "luac_status": "skipped_with_reason",
            },
        )


class _StubBackend:
    def __init__(self, settings_path: Path) -> None:
        self.settings_manager = SettingsManager(settings_path)
        self.orchestrator = _StubOrchestrator()


class EvalAndAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_self_check_returns_structured_statuses(self) -> None:
        report = await run_self_check(run_full_eval=False)
        statuses = {item["name"]: item["status"] for item in report["checks"]}
        self.assertIn(statuses["smoke_eval"], {"passed", "failed"})
        self.assertEqual(statuses["full_eval"], "skipped_with_reason")
        self.assertIn("pinned_python_dependencies", statuses)

    async def test_http_api_generate_endpoint_returns_code_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = _StubBackend(Path(tmp_dir) / "settings.json")
            LocalScriptAPIHandler.backend = backend  # type: ignore[assignment]
            server = ThreadingHTTPServer(("127.0.0.1", 0), LocalScriptAPIHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                payload = json.dumps({"prompt": "Верни orderId"}).encode("utf-8")
                req = urllib_request.Request(
                    f"http://127.0.0.1:{server.server_address[1]}/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib_request.urlopen(req, timeout=5) as response:
                    body = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                LocalScriptAPIHandler.backend = None

        self.assertEqual(body["code"], "return wf.vars.orderId")


class OptionalLiveOllamaTests(unittest.TestCase):
    def test_live_ollama_generate_is_opt_in(self) -> None:
        if os.getenv("ASSISTANT_RUN_LIVE_OLLAMA_TESTS") != "1":
            self.skipTest("Live Ollama tests are disabled. Set ASSISTANT_RUN_LIVE_OLLAMA_TESTS=1 to enable.")

        async def _run() -> AssistantResponse:
            from assistant.app import build_backend

            with tempfile.TemporaryDirectory() as tmp_dir:
                manager = SettingsManager(Path(tmp_dir) / "settings.json")
                backend = build_backend(settings_manager=manager, history_path=Path(tmp_dir) / "history.json")
                return await backend.orchestrator.generate_localscript_response(
                    'Верни orderId из workflow контекста. {"wf":{"vars":{"orderId":"123"}}}',
                    allow_clarification=False,
                    persist_memory=False,
                    use_memory_context=False,
                )

        response = asyncio.run(_run())
        self.assertIn("wf.vars.orderId", response.text)
        self.assertEqual(response.metrics.get("validation_errors"), 0)


if __name__ == "__main__":
    unittest.main()
