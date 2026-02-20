"""Base classes for source parsers used in RAG ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rag_common.llm import OpenAIClientAdapter
from rag_common.prompts import PromptRegistry
from rag_common.types import DocumentChunk, ParsedDocument


class BaseDocumentParser(ABC):
    """Base parser with optional LLM-based cleanup stage."""

    prompt_key: str

    def __init__(
        self,
        *,
        openai_client: Any | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
        normalize_with_llm: bool = False,
        prompt_registry: PromptRegistry | None = None,
        prompt_override: str | None = None,
    ) -> None:
        self._prompts = prompt_registry if prompt_registry else PromptRegistry()
        if prompt_override:
            self._prompts.set(self.prompt_key, prompt_override)
        self._normalize_with_llm = normalize_with_llm and openai_client is not None
        self._llm = (
            OpenAIClientAdapter(
                openai_client,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if openai_client is not None
            else None
        )

    def update_default_prompt(self, prompt: str) -> None:
        """Update the default parser cleanup prompt."""
        self._prompts.set(self.prompt_key, prompt)

    @abstractmethod
    def parse(self, source_path: str) -> ParsedDocument:
        """Parse and normalize a source file."""

    def _resolve_source_path(self, source_path: str) -> Path:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        if not path.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")
        return path

    def _normalize_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if not self._normalize_with_llm or self._llm is None:
            return chunks

        normalized: list[DocumentChunk] = []
        for chunk in chunks:
            clean_text = self._normalize_text(chunk.text)
            normalized.append(
                DocumentChunk(
                    text=clean_text,
                    order=chunk.order,
                    metadata=dict(chunk.metadata),
                )
            )
        return normalized

    def _normalize_text(self, text: str) -> str:
        if not self._llm:
            return text

        prompt = self._prompts.get(self.prompt_key).format(text=text)
        payload = self._llm.complete_json(
            system_prompt="You normalize parsed enterprise documents for retrieval.",
            user_prompt=prompt,
        )
        clean_text = payload.get("clean_text")
        if isinstance(clean_text, str) and clean_text.strip():
            return clean_text.strip()
        return text
