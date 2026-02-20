"""PDF parser component."""

from __future__ import annotations

from rag_common.exceptions import MissingDependencyError
from rag_common.parsers.base import BaseDocumentParser
from rag_common.prompts import PROMPT_PDF_CLEANUP
from rag_common.types import DocumentChunk, ParsedDocument


class PdfParser(BaseDocumentParser):
    """Parses PDF files into chunked text suitable for RAG indexing."""

    prompt_key = PROMPT_PDF_CLEANUP

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


def _load_pdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingDependencyError(
            "PdfParser requires 'pypdf'. Install with: pip install pypdf"
        ) from exc
    return PdfReader
