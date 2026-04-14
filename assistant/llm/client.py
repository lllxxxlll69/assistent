from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, AsyncIterator, Iterable

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs
    from urllib import error as urllib_error
    from urllib import request as urllib_request

    class _CompatRequestException(Exception):
        pass

    class _CompatResponse:
        def __init__(self, body: bytes, status_code: int) -> None:
            self._body = body
            self.status_code = status_code
            self.text = body.decode("utf-8", errors="replace")

        def json(self) -> dict[str, Any]:
            return json.loads(self.text)

        def iter_lines(self, decode_unicode: bool = False):
            for line in self.text.splitlines():
                yield line if decode_unicode else line.encode("utf-8")

        def close(self) -> None:
            return None

    class _CompatSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def post(
            self,
            url: str,
            *,
            json: dict[str, Any] | None = None,
            stream: bool = False,
            timeout: int | float | None = None,
        ) -> _CompatResponse:
            del stream
            payload = b""
            headers = dict(self.headers)
            if json is not None:
                payload = __import__("json").dumps(json).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            request = urllib_request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(request, timeout=timeout) as response:
                    body = response.read()
                    return _CompatResponse(body=body, status_code=response.status)
            except urllib_error.URLError as exc:
                raise _CompatRequestException(str(exc)) from exc

    class _CompatRequestsModule:
        RequestException = _CompatRequestException
        Session = _CompatSession

    requests = _CompatRequestsModule()

from assistant.config.settings import FIXED_BATCH_SIZE, FIXED_CONTEXT_SIZE, FIXED_NUM_PREDICT, Settings


class LLMClientError(RuntimeError):
    """Raised when the configured LLM endpoint fails."""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        stream: bool | None = None,
        max_tokens_override: int | None = None,
        context_size_override: int | None = None,
        temperature_override: float | None = None,
    ) -> str:
        payload = self._build_payload(
            messages=messages,
            model=model or self.settings.model,
            stream=stream if stream is not None else False,
            max_tokens_override=max_tokens_override,
            context_size_override=context_size_override,
            temperature_override=temperature_override,
        )
        try:
            response = self._session.post(
                self.settings.api_url,
                json=payload,
                timeout=self.settings.request_timeout,
            )
        except requests.RequestException as exc:
            raise LLMClientError(f"Не удалось подключиться к LLM endpoint: {exc}") from exc

        if response.status_code >= 400:
            raise LLMClientError(f"LLM request failed: {response.status_code} {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError("LLM endpoint returned invalid JSON.") from exc

        content = self._extract_content(data)
        if content is None or not content.strip():
            raise LLMClientError(
                f"Model {model or self.settings.model} returned an empty response. "
                "Try resetting settings or increasing timeout/context size."
            )
        return content.strip()

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens_override: int | None = None,
        context_size_override: int | None = None,
        temperature_override: float | None = None,
    ) -> Iterable[str]:
        payload = self._build_payload(
            messages=messages,
            model=model or self.settings.model,
            stream=True,
            max_tokens_override=max_tokens_override,
            context_size_override=context_size_override,
            temperature_override=temperature_override,
        )
        try:
            response = self._session.post(
                self.settings.api_url,
                json=payload,
                stream=True,
                timeout=self.settings.request_timeout,
            )
        except requests.RequestException as exc:
            raise LLMClientError(f"Не удалось открыть поток к LLM endpoint: {exc}") from exc

        if response.status_code >= 400:
            response.close()
            raise LLMClientError(f"LLM stream failed: {response.status_code} {response.text}")

        saw_chunk = False
        saw_invalid_frame = False
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    saw_invalid_frame = True
                    continue

                if error_text := data.get("error"):
                    raise LLMClientError(f"Ollama stream error: {error_text}")

                chunk = self._extract_content(data, strip=False)
                if chunk is not None and chunk != "":
                    saw_chunk = True
                    yield chunk

                if data.get("done"):
                    break
        except requests.RequestException as exc:
            raise LLMClientError(f"Потоковый ответ был прерван: {exc}") from exc
        finally:
            response.close()

        if not saw_chunk:
            if saw_invalid_frame:
                raise LLMClientError("LLM stream returned malformed data.")
            raise LLMClientError("Model returned an empty streamed response.")

    async def achat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens_override: int | None = None,
        context_size_override: int | None = None,
        temperature_override: float | None = None,
    ) -> AsyncIterator[str]:
        for chunk in self.chat_stream(
            messages,
            model=model,
            max_tokens_override=max_tokens_override,
            context_size_override=context_size_override,
            temperature_override=temperature_override,
        ):
            yield chunk

    def vision_chat(
        self,
        prompt: str,
        *,
        image_base64: str,
        model: str | None = None,
        max_tokens_override: int | None = None,
        context_size_override: int | None = None,
        temperature_override: float | None = None,
    ) -> str:
        vision_model = model or self.settings.vision_model
        payload = self._build_payload(
            messages=[{"role": "user", "content": prompt, "images": [image_base64]}],
            model=vision_model,
            stream=False,
            max_tokens_override=max_tokens_override,
            context_size_override=context_size_override,
            temperature_override=temperature_override,
        )
        try:
            response = self._session.post(
                self.settings.api_url,
                json=payload,
                timeout=self.settings.request_timeout,
            )
        except requests.RequestException as exc:
            raise LLMClientError(f"Vision request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMClientError(f"Vision request failed: {response.status_code} {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError("Vision endpoint returned invalid JSON.") from exc

        content = self._extract_content(data)
        if content is not None and content.strip():
            return content.strip()

        thinking = self._extract_thinking(data)
        if thinking:
            recovered = self._recover_vision_response(prompt, thinking, vision_model=vision_model)
            if recovered and recovered.strip():
                return recovered.strip()

        raise LLMClientError(f"Vision model {vision_model} returned an empty response.")

    def warm_up(
        self,
        model: str | None = None,
        *,
        max_tokens_override: int | None = None,
        context_size_override: int | None = None,
        temperature_override: float | None = None,
    ) -> float:
        warm_model = model or self.settings.model
        payload = self._build_payload(
            messages=[{"role": "user", "content": "."}],
            model=warm_model,
            stream=False,
            max_tokens_override=max_tokens_override or 8,
            context_size_override=context_size_override or min(self.settings.context_size, 4096),
            temperature_override=temperature_override,
        )
        started = perf_counter()
        try:
            response = self._session.post(
                self.settings.api_url,
                json=payload,
                timeout=max(60, self.settings.request_timeout),
            )
        except requests.RequestException as exc:
            raise LLMClientError(f"Warm-up failed for {warm_model}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMClientError(f"Warm-up failed for {warm_model}: {response.status_code} {response.text}")
        return perf_counter() - started

    def _build_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool,
        max_tokens_override: int | None = None,
        context_size_override: int | None = None,
        temperature_override: float | None = None,
    ) -> dict[str, object]:
        del max_tokens_override, context_size_override
        options: dict[str, Any] = {
            "temperature": self.settings.temperature if temperature_override is None else temperature_override,
            # These values are fixed by the runtime contract and must not drift.
            "num_predict": FIXED_NUM_PREDICT,
            "num_ctx": FIXED_CONTEXT_SIZE,
            "num_batch": FIXED_BATCH_SIZE,
        }
        if self.settings.gpu_layers != 0:
            options["num_gpu"] = self.settings.gpu_layers
        if self.settings.main_gpu >= 0:
            options["main_gpu"] = self.settings.main_gpu
        if self.settings.cpu_threads > 0:
            options["num_thread"] = self.settings.cpu_threads
        if self.settings.low_vram:
            options["low_vram"] = True

        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "keep_alive": self.settings.keep_alive,
        }

    def _extract_content(self, data: dict[str, Any], *, strip: bool = True) -> str | None:
        if error_text := data.get("error"):
            raise LLMClientError(str(error_text))
        content: Any | None = None
        message_payload = data.get("message")
        if isinstance(message_payload, dict) and "content" in message_payload:
            content = message_payload.get("content")
        elif isinstance(data.get("response"), str):
            content = data["response"]

        if content is None:
            return None

        text = str(content)
        return text.strip() if strip else text

    def _extract_thinking(self, data: dict[str, Any]) -> str | None:
        message_payload = data.get("message")
        if isinstance(message_payload, dict):
            thinking = message_payload.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                return thinking.strip()

        root_thinking = data.get("thinking")
        if isinstance(root_thinking, str) and root_thinking.strip():
            return root_thinking.strip()
        return None

    def _recover_vision_response(self, prompt: str, thinking: str, *, vision_model: str) -> str:
        summary_model = (self.settings.model or "").strip()
        if summary_model and summary_model.lower() != vision_model.strip().lower():
            try:
                return self.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Ты превращаешь черновик анализа изображения в короткий итоговый ответ для пользователя. "
                                "Не пересказывай внутренние рассуждения и отвечай только на русском языке."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Запрос пользователя:\n{prompt}\n\n"
                                f"Черновик анализа изображения:\n{thinking}\n\n"
                                "Сформируй только итоговый ответ: 1 краткое summary и до 4 коротких пунктов."
                            ),
                        },
                    ],
                    model=summary_model,
                    stream=False,
                    temperature_override=0.0,
                )
            except LLMClientError:
                pass

        return self._draft_visible_response_from_thinking(thinking)

    def _draft_visible_response_from_thinking(self, thinking: str) -> str:
        normalized = re.sub(r"\s+", " ", thinking).strip()
        if not normalized:
            return ""

        meta_markers = (
            "мне нужно",
            "надо",
            "сначала",
            "начну",
            "проверю",
            "хорошо",
            "i need",
            "let me",
            "first,",
            "first ",
        )
        sentences = [item.strip(" -") for item in re.split(r"(?<=[.!?])\s+", normalized) if item.strip()]
        visible_sentences = [
            sentence
            for sentence in sentences
            if not any(marker in sentence.lower() for marker in meta_markers)
        ]
        selected = visible_sentences[:3] or sentences[:2]
        drafted = " ".join(selected).strip()
        return drafted[:400].rstrip(" ,;:")
