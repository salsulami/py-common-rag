"""Rendering helpers for converting documents into screenshot images."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rag_common.exceptions import DocumentConversionError, MissingDependencyError


def iter_pdf_page_images(source_path: Path, *, dpi: int = 170) -> Iterator[tuple[int, bytes]]:
    """Yield PNG bytes for each PDF page."""
    fitz = _load_pymupdf()
    scale = max(float(dpi), 72.0) / 72.0

    with fitz.open(str(source_path)) as document:
        transform = fitz.Matrix(scale, scale)
        for page_index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=transform, alpha=False)
            yield page_index, pixmap.tobytes("png")


def iter_office_page_images(source_path: Path, *, dpi: int = 170) -> Iterator[tuple[int, bytes]]:
    """
    Convert Office file to PDF with LibreOffice, then render each page to PNG bytes.

    Used for DOCX/PPTX visual extraction where page/slide screenshots are required.
    """
    soffice_binary = shutil.which("soffice")
    if not soffice_binary:
        raise MissingDependencyError(
            "Visual extraction for DOCX/PPTX requires LibreOffice ('soffice') in PATH."
        )

    with tempfile.TemporaryDirectory(prefix="rag_common_render_") as temp_dir:
        output_dir = Path(temp_dir)
        pdf_path = _convert_to_pdf_with_soffice(
            source_path=source_path,
            output_dir=output_dir,
            soffice_binary=soffice_binary,
        )
        yield from iter_pdf_page_images(pdf_path, dpi=dpi)


def visual_runtime_diagnostics() -> dict[str, bool]:
    """Return runtime availability flags used by visual extraction."""
    return {
        "pymupdf_available": _has_pymupdf(),
        "soffice_available": _has_soffice(),
    }


def _convert_to_pdf_with_soffice(
    *,
    source_path: Path,
    output_dir: Path,
    soffice_binary: str,
) -> Path:
    command = [
        soffice_binary,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(source_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DocumentConversionError(
            f"Failed to convert '{source_path}' to PDF. "
            f"stderr={result.stderr.strip() or 'N/A'}"
        )

    generated_pdf = output_dir / f"{source_path.stem}.pdf"
    if generated_pdf.exists():
        return generated_pdf

    all_pdfs = sorted(output_dir.glob("*.pdf"))
    if len(all_pdfs) == 1:
        return all_pdfs[0]
    raise DocumentConversionError(
        f"LibreOffice did not produce a PDF for '{source_path}'."
    )


def _load_pymupdf() -> Any:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingDependencyError(
            "Visual screenshot rendering requires 'pymupdf'. Install with: pip install pymupdf"
        ) from exc
    return fitz


def _has_pymupdf() -> bool:
    try:
        import fitz  # noqa: F401
    except ImportError:
        return False
    return True


def _has_soffice() -> bool:
    return shutil.which("soffice") is not None
