from __future__ import annotations

import base64
from pathlib import Path

from assistant.config.settings import Settings
from assistant.llm.client import LLMClient
from assistant.llm.prompts import build_vision_prompt
from assistant.models import VisionRequest, VisionResult


class VisionTools:
    def __init__(self, llm_client: LLMClient, settings: Settings) -> None:
        self.llm_client = llm_client
        self.settings = settings

    def analyze_image(self, request: VisionRequest) -> VisionResult:
        image_base64 = request.image_base64 or self._load_image_as_base64(request.image_path)
        prompt = build_vision_prompt(request.prompt)
        raw_response = self.llm_client.vision_chat(prompt, image_base64=image_base64)
        lines = [line.strip("- ").strip() for line in raw_response.splitlines() if line.strip()]
        summary = lines[0] if lines else raw_response
        details = lines[1:] if len(lines) > 1 else []
        return VisionResult(summary=summary, details=details, raw_response=raw_response)

    def _load_image_as_base64(self, image_path: str | None) -> str:
        if not image_path:
            raise ValueError("Either image_path or image_base64 must be provided.")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")
        max_size_bytes = self.settings.max_image_size_mb * 1024 * 1024
        if path.stat().st_size > max_size_bytes:
            raise ValueError(f"Image is too large: {path.name}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")
