from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from assistant.config.settings import Settings
from assistant.models import RetrievalChunk, ToolResult


class SearchTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search_local_files(self, query: str, root: str | None = None) -> ToolResult:
        search_root = Path(root or self.settings.search_root)
        chunks = self.retrieve_chunks(query=query, root=search_root)
        rendered = "\n\n".join(chunk.to_prompt_block() for chunk in chunks) or "No relevant files found."
        return ToolResult(
            content=rendered,
            structured_data={"chunks": [asdict(chunk) for chunk in chunks]},
        )

    def retrieve_chunks(self, query: str, root: Path) -> list[RetrievalChunk]:
        if not root.exists():
            return []

        query_terms = self._tokenize(query)
        scored_chunks: list[RetrievalChunk] = []

        for path in root.rglob("*"):
            if not path.is_file() or self._should_skip(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for chunk in self._chunk_text(text):
                score = self._score(query_terms, self._tokenize(chunk))
                if score <= 0:
                    continue
                scored_chunks.append(
                    RetrievalChunk(path=str(path), snippet=chunk[: self.settings.search_chunk_size], score=score)
                )

        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        return scored_chunks[: self.settings.max_search_results]

    def _chunk_text(self, text: str) -> list[str]:
        chunk_size = self.settings.search_chunk_size
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)] or [text]

    def _score(self, query_terms: list[str], chunk_terms: list[str]) -> float:
        if not query_terms or not chunk_terms:
            return 0.0
        query_counter = Counter(query_terms)
        chunk_counter = Counter(chunk_terms)
        numerator = sum(query_counter[token] * chunk_counter[token] for token in query_counter)
        query_norm = math.sqrt(sum(value * value for value in query_counter.values()))
        chunk_norm = math.sqrt(sum(value * value for value in chunk_counter.values()))
        if query_norm == 0 or chunk_norm == 0:
            return 0.0
        return numerator / (query_norm * chunk_norm)

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in text.replace("\n", " ").split() if token.strip()]

    def _should_skip(self, path: Path) -> bool:
        skipped_parts = {".git", ".venv", "__pycache__", "node_modules", ".idea"}
        return any(part in skipped_parts for part in path.parts)
