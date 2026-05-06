"""
Document loading utilities for ArgusAI.

Supported inputs:
- text-like files: .txt, .md, .py, .json, .csv, .yaml, ...
- PDF files: direct text extraction first, OCR fallback if needed
- image files: delegated to OCRManager

PDF OCR is intentionally limited by Config.pdf_ocr_max_pages for MVP stability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Set, cast

from .config import Config
from .utils import clean_text

if TYPE_CHECKING:
    from .ocr import OCRManager


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore[assignment]

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None  # type: ignore[assignment]

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None  # type: ignore[assignment]


class DocumentLoader:
    """Text and OCR loader for text files, PDFs, and images."""

    TEXT_SUFFIXES: Set[str] = {
        ".txt",
        ".md",
        ".py",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
        ".log",
        ".toml",
        ".ini",
        ".cfg",
        ".sql",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".xml",
    }

    IMAGE_SUFFIXES: Set[str] = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    }

    def __init__(self, config: Config) -> None:
        self.config = config

    def load_document_text(self, path: str, ocr_manager: "OCRManager") -> str:
        """
        Load text from a supported document path.

        Strategy:
        1. Read text-like files directly.
        2. Extract PDF text using pypdf.
        3. If PDF extraction fails or returns empty text, fallback to OCR.
        4. For image files, delegate to OCRManager.
        """

        file_path = Path(path)

        if not file_path.exists():
            logging.warning("Document not found: %s", file_path)
            return ""

        suffix = file_path.suffix.lower()

        if suffix in self.TEXT_SUFFIXES:
            return self._read_text_file(file_path)

        if suffix == ".pdf":
            extracted = self.extract_pdf_text(file_path)

            if extracted:
                return extracted

            return self.extract_pdf_text_with_ocr(file_path, ocr_manager)

        if suffix in self.IMAGE_SUFFIXES:
            return ocr_manager.run_ocr_on_path(str(file_path), self)

        logging.warning("Unsupported document type: %s", suffix)
        return ""

    def _read_text_file(self, path: Path) -> str:
        """Read a text-like file safely."""

        try:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            logging.warning("Unable to read text file %s: %s", path, exc)
            return ""

    def extract_pdf_text(self, path: Path) -> str:
        """
        Extract embedded text from a PDF using pypdf.

        This works for digital PDFs. Scanned PDFs usually need OCR fallback.
        """

        if PdfReader is None:
            logging.warning("pypdf is not installed; PDF text extraction disabled.")
            return ""

        try:
            reader = PdfReader(str(path))
            parts: List[str] = []

            for page in reader.pages:
                parts.append(page.extract_text() or "")

            return clean_text("\n".join(parts))

        except Exception as exc:
            logging.warning("PDF text extraction failed for %s: %s", path, exc)
            return ""

    def extract_pdf_text_with_ocr(self, path: Path, ocr_manager: "OCRManager") -> str:
        """
        Extract text from a scanned PDF through OCR.

        PDF pages are rendered to images, then passed to OCRManager.
        """

        if Image is None:
            return "OCR unavailable because Pillow is not installed."

        if not (ocr_manager.has_surya() or ocr_manager.has_tesseract()):
            return "OCR unavailable for PDF in this environment."

        images = self.render_pdf_to_images(
            path,
            max_pages=self.config.pdf_ocr_max_pages,
        )

        if not images:
            return "OCR unavailable for PDF in this environment."

        texts: List[str] = []

        for image in images:
            page_text = ocr_manager.ocr_pil_image(image)

            if page_text:
                texts.append(page_text)

        combined = clean_text("\n\n".join(texts))

        if not combined:
            return "OCR unavailable for PDF in this environment."

        return combined

    def render_pdf_to_images(self, path: Path, max_pages: int) -> List[Any]:
        """
        Render PDF pages to PIL images.

        Priority:
        1. pypdfium2
        2. PyMuPDF / fitz
        """

        images = self._render_pdf_with_pdfium(path, max_pages=max_pages)

        if images:
            return images

        return self._render_pdf_with_pymupdf(path, max_pages=max_pages)

    def _render_pdf_with_pdfium(self, path: Path, max_pages: int) -> List[Any]:
        """Render PDF pages using pypdfium2."""

        images: List[Any] = []

        if Image is None or pdfium is None:
            return images

        try:
            pdf = pdfium.PdfDocument(str(path))
            limit = min(len(pdf), max_pages)

            for page_index in range(limit):
                page = pdf[page_index]
                render_scale = cast(Any, float(self.config.pdf_ocr_scale))
                bitmap = cast(Any, page).render(scale=render_scale)
                pil_image = bitmap.to_pil()
                images.append(pil_image)

            return images

        except Exception as exc:
            logging.warning("pypdfium2 PDF rendering failed for %s: %s", path, exc)
            return []

    def _render_pdf_with_pymupdf(self, path: Path, max_pages: int) -> List[Any]:
        """Render PDF pages using PyMuPDF / fitz."""

        images: List[Any] = []

        if Image is None or fitz is None:
            return images

        try:
            document = fitz.open(str(path))
            limit = min(document.page_count, max_pages)
            matrix = fitz.Matrix(
                self.config.pdf_ocr_scale,
                self.config.pdf_ocr_scale,
            )

            for page_index in range(limit):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix)

                mode = "RGBA" if pixmap.alpha else "RGB"

                pil_image = Image.frombytes(
                    mode,
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )

                images.append(pil_image)

            return images

        except Exception as exc:
            logging.warning("PyMuPDF PDF rendering failed for %s: %s", path, exc)
            return []

    def pdf_ocr_backend_status(self) -> str:
        """Return available PDF rendering backend status."""

        if pdfium is not None:
            return "pypdfium2"

        if fitz is not None:
            return "pymupdf"

        return "off"