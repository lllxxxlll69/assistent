from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from assistant.config.settings import Settings
from assistant.models import ActionLogEntry, RetrievalChunk, ToolResult


class SearchTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.workspace_root = Path(settings.workspace_root).resolve(strict=False)

    def search_local_files(self, query: str, root: str | None = None) -> ToolResult:
        search_root, error = self._resolve_search_root(root)
        if error:
            return ToolResult(
                content=error,
                logs=[ActionLogEntry(message=error, success=False)],
                structured_data={"chunks": []},
            )

        chunks = self.retrieve_chunks(query=query, root=search_root)
        rendered = "\n\n".join(chunk.to_prompt_block() for chunk in chunks) or "No relevant files found."
        return ToolResult(
            content=rendered,
            logs=[ActionLogEntry(message=f"Local search completed in {search_root}")],
            structured_data={"chunks": [asdict(chunk) for chunk in chunks]},
        )

    def retrieve_chunks(self, query: str, root: Path) -> list[RetrievalChunk]:
        if not root.exists():
            return []

        query_terms = self._tokenize(query)
        scored_chunks: list[RetrievalChunk] = []
        max_file_size = self.settings.max_search_file_size_kb * 1024

        for path in root.rglob("*"):
            if not path.is_file() or self._should_skip(path):
                continue
            try:
                if path.stat().st_size > max_file_size:
                    continue
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

    def _resolve_search_root(self, root: str | None) -> tuple[Path, str | None]:
        candidate = Path(root or self.settings.search_root)
        resolved = candidate.resolve(strict=False) if candidate.is_absolute() else (self.workspace_root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            return self.workspace_root, f"Search root is outside workspace: {resolved}"
        return resolved, None

    def _should_skip(self, path: Path) -> bool:
        skipped_parts = {".git", ".venv", "__pycache__", "node_modules", ".idea", ".assistant_data"}
        return any(part in skipped_parts for part in path.parts)
