"""Base classes for source parsers used in RAG ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rag_common.llm import OpenAIClientAdapter
from rag_common.prompts import PromptRegistry
from rag_common.types import (
    DocumentChunk,
    JSONDict,
    ParsedDocument,
    VisualExtractionItem,
    VisualExtractionResult,
)


class BaseDocumentParser(ABC):
    """Base parser with optional LLM-based cleanup stage."""

    prompt_key: str
    vision_prompt_key: str = ""

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

    def update_prompt(self, key: str, prompt: str) -> None:
        """Update any prompt key used by this parser."""
        self._prompts.set(key, prompt)

    @abstractmethod
    def parse(self, source_path: str) -> ParsedDocument:
        """Parse and normalize a source file."""

    def iter_visual_items(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> Iterator[VisualExtractionItem]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement visual item streaming."
        )

    def extract_visual_result(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> VisualExtractionResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement visual result extraction."
        )

    def extract_visual_json(self, source_path: str, *, dpi: int = 170) -> JSONDict:
        """Process all items and return a full JSON payload."""
        return self.extract_visual_result(source_path, dpi=dpi).to_dict()

    def stream_visual_json_items(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> Iterator[JSONDict]:
        """Yield one extracted item JSON payload at a time."""
        for item in self.iter_visual_items(source_path, dpi=dpi):
            yield item.to_dict()

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

    def _require_llm(self) -> OpenAIClientAdapter:
        if self._llm is None:
            raise ValueError(
                "This parser requires an OpenAI client for vision extraction. "
                "Initialize with openai_client=..."
            )
        return self._llm

    def _extract_visual_item(
        self,
        *,
        source_path: str,
        index: int,
        item_type: str,
        image_bytes: bytes,
        prompt_key: str,
    ) -> VisualExtractionItem:
        llm = self._require_llm()
        prompt = self._prompts.get(prompt_key).format(
            source_path=source_path,
            index=index,
            item_type=item_type,
        )
        payload = llm.complete_json_with_image(
            system_prompt=(
                "You extract faithful markdown and concise captions from document screenshots."
            ),
            user_prompt=prompt,
            image_bytes=image_bytes,
        )
        markdown = _safe_string(payload.get("markdown")) or ""
        caption = _safe_string(payload.get("caption")) or ""
        return VisualExtractionItem(
            index=index,
            markdown=markdown,
            caption=caption,
            metadata={"item_type": item_type},
            raw_response=payload,
        )


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None
