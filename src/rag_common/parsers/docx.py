"""Word document parser component."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from rag_common.exceptions import MissingDependencyError
from rag_common.parsers.base import BaseDocumentParser
from rag_common.prompts import PROMPT_DOCX_CLEANUP
from rag_common.types import DocumentChunk, ParsedDocument


class DocxParser(BaseDocumentParser):
    """Parses DOCX files into paragraph/table chunks."""

    prompt_key = PROMPT_DOCX_CLEANUP

    def parse(self, source_path: str) -> ParsedDocument:
        path = self._resolve_source_path(source_path)
        document_ctor, paragraph_cls, table_cls = _load_docx_types()
        document = document_ctor(str(path))

        chunks: list[DocumentChunk] = []
        order = 1

        for block in _iter_blocks(document, paragraph_cls, table_cls):
            if isinstance(block, paragraph_cls):
                text = block.text.strip()
                if not text:
                    continue
                style_name = getattr(getattr(block, "style", None), "name", None)
                metadata: dict[str, Any] = {"block_type": "paragraph"}
                if isinstance(style_name, str) and style_name.strip():
                    metadata["style"] = style_name.strip()
                chunks.append(DocumentChunk(text=text, order=order, metadata=metadata))
                order += 1
                continue

            if isinstance(block, table_cls):
                table_lines = _table_to_lines(block)
                for row_number, row_text in enumerate(table_lines, start=1):
                    chunks.append(
                        DocumentChunk(
                            text=row_text,
                            order=order,
                            metadata={
                                "block_type": "table_row",
                                "row_number": row_number,
                            },
                        )
                    )
                    order += 1

        chunks = self._normalize_chunks(chunks)
        content = "\n\n".join(chunk.text for chunk in chunks).strip()
        return ParsedDocument(
            source_path=str(path),
            content=content,
            chunks=chunks,
            metadata={
                "parser": "docx",
                "chunk_count": len(chunks),
            },
        )


def _iter_blocks(document: Any, paragraph_cls: type[Any], table_cls: type[Any]) -> Iterator[Any]:
    parent = document.element.body
    for child in parent.iterchildren():
        if child.tag.endswith("}p"):
            yield paragraph_cls(child, document)
        elif child.tag.endswith("}tbl"):
            yield table_cls(child, document)


def _table_to_lines(table: Any) -> list[str]:
    lines: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _load_docx_types():
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingDependencyError(
            "DocxParser requires 'python-docx'. Install with: pip install python-docx"
        ) from exc
    return Document, Paragraph, Table
