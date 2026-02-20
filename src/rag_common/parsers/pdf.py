"""PDF parser component."""

from __future__ import annotations

from collections.abc import Iterator

from rag_common.exceptions import MissingDependencyError
from rag_common.parsers.base import BaseDocumentParser
from rag_common.parsers.rendering import iter_pdf_page_images
from rag_common.prompts import PROMPT_PDF_CLEANUP, PROMPT_PDF_VISION_EXTRACTION
from rag_common.types import (
    DocumentChunk,
    FileManifest,
    ParsedDocument,
    VisualExtractionItem,
    VisualExtractionResult,
)


class PdfParser(BaseDocumentParser):
    """Parses PDF files into chunked text suitable for RAG indexing."""

    prompt_key = PROMPT_PDF_CLEANUP
    vision_prompt_key = PROMPT_PDF_VISION_EXTRACTION

    def parse(self, source_path: str) -> ParsedDocument:
        path = self._resolve_source_path(source_path)
        pdf_reader_cls = _load_pdf_reader()
        reader = pdf_reader_cls(str(path))

        chunks: list[DocumentChunk] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            chunks.append(
                DocumentChunk(
                    text=page_text,
                    order=page_number,
                    metadata={"page_number": page_number},
                )
            )

        chunks = self._normalize_chunks(chunks)
        content = "\n\n".join(chunk.text for chunk in chunks).strip()
        return ParsedDocument(
            source_path=str(path),
            content=content,
            chunks=chunks,
            metadata={
                "parser": "pdf",
                "page_count": len(reader.pages),
                "chunk_count": len(chunks),
            },
        )

    def iter_visual_items(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> Iterator[VisualExtractionItem]:
        path = self._resolve_source_path(source_path)
        for page_index, image_bytes in iter_pdf_page_images(path, dpi=dpi):
            yield self._extract_visual_item(
                source_path=str(path),
                index=page_index,
                item_type="page",
                image_bytes=image_bytes,
                prompt_key=self.vision_prompt_key,
            )

    def extract_visual_result(
        self,
        source_path: str,
        *,
        dpi: int = 170,
        continue_on_error: bool | None = None,
        max_errors: int | None = None,
    ) -> VisualExtractionResult:
        path = self._resolve_source_path(source_path)
        items, errors, attempted = self._collect_visual_items(
            source_path=str(path),
            item_type="page",
            prompt_key=self.vision_prompt_key,
            images=iter_pdf_page_images(path, dpi=dpi),
            continue_on_error=continue_on_error,
            max_errors=max_errors,
        )
        manifest = FileManifest(
            source_path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            item_type="page",
            item_count=len(items),
            metadata={
                "dpi": dpi,
                "attempted_items": attempted,
                "successful_items": len(items),
                "failed_items": len(errors),
            },
        )
        return VisualExtractionResult(file_manifest=manifest, items=items, errors=errors)


def _load_pdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingDependencyError(
            "PdfParser requires 'pypdf'. Install with: pip install pypdf"
        ) from exc
    return PdfReader
