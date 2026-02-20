"""Routing helpers for parser selection by file extension."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rag_common.exceptions import UnsupportedFileTypeError
from rag_common.parsers.base import BaseDocumentParser
from rag_common.parsers.docx import DocxParser
from rag_common.parsers.pdf import PdfParser
from rag_common.parsers.pptx import PptxParser
from rag_common.types import JSONDict, ParsedDocument, VisualExtractionItem, VisualExtractionResult


class DocumentParserRouter:
    """Dispatches files to parser implementations based on extension."""

    def __init__(
        self,
        *,
        pdf_parser: BaseDocumentParser | None = None,
        pptx_parser: BaseDocumentParser | None = None,
        docx_parser: BaseDocumentParser | None = None,
    ) -> None:
        self._parsers: dict[str, BaseDocumentParser] = {
            ".pdf": pdf_parser or PdfParser(),
            ".pptx": pptx_parser or PptxParser(),
            ".docx": docx_parser or DocxParser(),
        }

    def register_parser(self, extension: str, parser: BaseDocumentParser) -> None:
        normalized_ext = extension.lower().strip()
        if not normalized_ext.startswith("."):
            normalized_ext = f".{normalized_ext}"
        self._parsers[normalized_ext] = parser

    def parse(self, source_path: str) -> ParsedDocument:
        parser = self._resolve_parser(source_path)
        return parser.parse(source_path)

    def stream_visual_items(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> Iterator[VisualExtractionItem]:
        parser = self._resolve_parser(source_path, for_visual=True)
        yield from parser.iter_visual_items(source_path, dpi=dpi)

    def stream_visual_json_items(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> Iterator[JSONDict]:
        for item in self.stream_visual_items(source_path, dpi=dpi):
            yield item.to_dict()

    def extract_visual_result(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> VisualExtractionResult:
        parser = self._resolve_parser(source_path, for_visual=True)
        return parser.extract_visual_result(source_path, dpi=dpi)

    def extract_visual_json(self, source_path: str, *, dpi: int = 170) -> JSONDict:
        return self.extract_visual_result(source_path, dpi=dpi).to_dict()

    def _resolve_parser(self, source_path: str, *, for_visual: bool = False) -> BaseDocumentParser:
        ext = Path(source_path).suffix.lower()
        if for_visual and ext in {".doc", ".docm"}:
            ext = ".docx"
        if for_visual and ext in {".ppt", ".pptm"}:
            ext = ".pptx"

        parser = self._parsers.get(ext)
        if parser is None:
            if ext in {".doc", ".ppt"}:
                raise UnsupportedFileTypeError(
                    f"Legacy Office format '{ext}' is not directly supported. "
                    "Please convert to .docx or .pptx first."
                )
            raise UnsupportedFileTypeError(
                f"No parser registered for extension '{ext}'. "
                "Supported extensions: "
                + ", ".join(sorted(self._parsers))
            )
        return parser
