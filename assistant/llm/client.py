from __future__ import annotations

import json
from time import perf_counter
from typing import AsyncIterator, Iterable

import requests

from assistant.config.settings import Settings


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
    ) -> str:
        payload = self._build_payload(
            messages=messages,
            model=model or self.settings.model,
            stream=stream if stream is not None else False,
            max_tokens_override=max_tokens_override,
        )
        response = self._session.post(
            self.settings.api_url,
            json=payload,
            timeout=self.settings.request_timeout,
        )
        if response.status_code >= 400:
            raise LLMClientError(f"LLM request failed: {response.status_code} {response.text}")

        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMClientError(
                f"Model {model or self.settings.model} returned an empty response. "
                "Try resetting settings or increasing timeout/context size."
            )
        return content

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens_override: int | None = None,
    ) -> Iterable[str]:
        payload = self._build_payload(
            messages=messages,
            model=model or self.settings.model,
            stream=True,
            max_tokens_override=max_tokens_override,
        )
        response = self._session.post(
            self.settings.api_url,
            json=payload,
            stream=True,
            timeout=self.settings.request_timeout,
        )
        if response.status_code >= 400:
            raise LLMClientError(f"LLM stream failed: {response.status_code} {response.text}")

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = data.get("message", {}).get("content")
            if chunk:
                yield chunk
            if data.get("done"):
                break

    async def achat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens_override: int | None = None,
    ) -> AsyncIterator[str]:
        for chunk in self.chat_stream(messages, model=model, max_tokens_override=max_tokens_override):
            yield chunk

    def vision_chat(
        self,
        prompt: str,
        *,
        image_base64: str,
        model: str | None = None,
        max_tokens_override: int | None = None,
    ) -> str:
        payload = self._build_payload(
            messages=[{"role": "user", "content": prompt, "images": [image_base64]}],
            model=model or self.settings.vision_model,
            stream=False,
            max_tokens_override=max_tokens_override,
        )
        response = self._session.post(
            self.settings.api_url,
            json=payload,
            timeout=self.settings.request_timeout,
        )
        if response.status_code >= 400:
            raise LLMClientError(f"Vision request failed: {response.status_code} {response.text}")

        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMClientError(f"Vision model {model or self.settings.vision_model} returned an empty response.")
        return content

    def warm_up(self, model: str | None = None) -> float:
        warm_model = model or self.settings.model
        payload = self._build_payload(
            messages=[{"role": "user", "content": "."}],
            model=warm_model,
            stream=False,
            max_tokens_override=8,
        )
        started = perf_counter()
        response = self._session.post(
            self.settings.api_url,
            json=payload,
            timeout=max(60, self.settings.request_timeout),
        )
        if response.status_code >= 400:
            raise LLMClientError(f"Warm-up failed for {warm_model}: {response.status_code} {response.text}")
        return perf_counter() - started

    def _build_payload(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        stream: bool,
        max_tokens_override: int | None = None,
    ) -> dict[str, object]:
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": max_tokens_override or self.settings.max_tokens,
                "num_ctx": self.settings.context_size,
            },
            "keep_alive": "30m",
        }
