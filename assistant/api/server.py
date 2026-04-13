from __future__ import annotations

import asyncio
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from assistant.app import AssistantBackend, build_backend
from assistant.config.settings import Settings
from assistant.llm.client import LLMClientError
from assistant.models import Message


class LocalScriptAPIHandler(BaseHTTPRequestHandler):
    backend: AssistantBackend | None = None
    openapi_path = Path(__file__).with_name("openapi.yaml")

    server_version = "LocalScriptAPI/1.0"

    def do_OPTIONS(self) -> None:  # type: ignore[override]
        self.send_response(HTTPStatus.NO_CONTENT)
        self._write_common_headers(content_type="text/plain; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:  # type: ignore[override]
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if parsed.path == "/openapi.yaml":
            payload = self.openapi_path.read_text(encoding="utf-8")
            self.send_response(HTTPStatus.OK)
            self._write_common_headers(content_type="application/yaml; charset=utf-8")
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # type: ignore[override]
        parsed = urlparse(self.path)
        if parsed.path != "/generate":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        settings = self._settings()
        if not self._authorize(settings):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
            return

        try:
            request_payload = self._read_json_body(settings)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        prompt = request_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Field 'prompt' must be a non-empty string."})
            return
        try:
            context_messages = self._parse_context_messages(request_payload.get("context_messages"))
            allow_clarification = self._coerce_bool(request_payload.get("allow_clarification"), default=False)
            persist_memory = self._coerce_bool(request_payload.get("persist_memory"), default=False)
            use_memory_context = self._coerce_bool(request_payload.get("use_memory_context"), default=False)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        backend = self.backend or build_backend()
        try:
            response = asyncio.run(
                backend.orchestrator.generate_localscript_response(
                    prompt.strip(),
                    allow_clarification=allow_clarification,
                    persist_memory=persist_memory,
                    use_memory_context=use_memory_context,
                    context_messages_override=context_messages,
                )
            )
        except LLMClientError as exc:
            print(f"[api] model error: {exc}")
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Local model request failed."})
            return
        except Exception as exc:  # pragma: no cover - defensive API guard
            print(f"[api] internal error: {exc}")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Internal server error."})
            return

        selected_strategy = str(response.metrics.get("selected_strategy", ""))
        clarification_question = response.text if selected_strategy == "clarification" else None
        code = "" if clarification_question is not None else response.text
        self._send_json(
            HTTPStatus.OK,
            {
                "code": code,
                "clarification_question": clarification_question,
                "metrics": response.metrics,
            },
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(f"[api] {self.address_string()} - {format % args}")

    def _read_json_body(self, settings: Settings) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is required.")
        if content_length > settings.max_request_bytes:
            raise ValueError("Request body is too large.")
        raw_payload = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON body: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def _authorize(self, settings: Settings) -> bool:
        if not settings.api_token:
            return True
        auth_header = self.headers.get("Authorization", "")
        api_key = self.headers.get("X-API-Key", "")
        if auth_header == f"Bearer {settings.api_token}":
            return True
        if api_key == settings.api_token:
            return True
        return False

    def _coerce_bool(self, value: object, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        raise ValueError("Boolean API fields must be true or false.")

    def _parse_context_messages(self, value: object) -> list[Message] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("Field 'context_messages' must be an array of {role, content} objects.")
        parsed: list[Message] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Each context message must be an object.")
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
                raise ValueError("Each context message must contain role=system|user|assistant.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Each context message must contain non-empty string content.")
            parsed.append(Message(role=role, content=content.strip()))
        return parsed

    def _settings(self) -> Settings:
        backend = self.backend or build_backend()
        return backend.settings_manager.get_settings()

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._write_common_headers(content_type="application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_common_headers(self, *, content_type: str) -> None:
        settings = self._settings()
        self.send_header("Content-Type", content_type)
        origin = self.headers.get("Origin", "").strip()
        allowed_origins = {item.strip() for item in settings.api_allowed_origins.split(",") if item.strip()}
        if origin and origin in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")


def serve() -> None:
    backend = build_backend()
    settings = backend.settings_manager.get_settings()
    LocalScriptAPIHandler.backend = backend
    server = ThreadingHTTPServer((settings.api_host, settings.api_port), LocalScriptAPIHandler)
    print(f"LocalScript API listening on http://{settings.api_host}:{settings.api_port}")
    server.serve_forever()


def main() -> None:
    serve()
