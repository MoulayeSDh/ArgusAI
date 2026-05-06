"""
OCR management for ArgusAI.

Default OCR backend:
- pytesseract/Tesseract for MVP stability, especially in Docker

Optional fallback backend:
- Surya OCR, attempted automatically when Tesseract is not usable and Surya is
  installed. Surya remains defensive because its Python API changes frequently.

This module does not handle full document loading. PDF rendering and document
text extraction are implemented in documents.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

from .config import Config
from .utils import clean_text

if TYPE_CHECKING:
    from .documents import DocumentLoader


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment]


# Surya API changes across versions; keep imports defensive.
try:
    from surya.detection import DetectionPredictor  # type: ignore
except Exception:
    DetectionPredictor = None  # type: ignore[assignment]

try:
    from surya.recognition import RecognitionPredictor  # type: ignore
except Exception:
    RecognitionPredictor = None  # type: ignore[assignment]

try:
    from surya.foundation import FoundationPredictor  # type: ignore
except Exception:
    FoundationPredictor = None  # type: ignore[assignment]


SURYA_AVAILABLE = DetectionPredictor is not None and RecognitionPredictor is not None


class OCRManager:
    """
    OCR manager with Tesseract as the stable MVP backend.

    Surya is optional and used as a best-effort fallback. This lets Windows
    users continue without manually editing PATH when a working Surya install is
    present, while Docker keeps the more predictable Tesseract runtime.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        self.surya_foundation: Optional[Any] = None
        self.surya_detection: Optional[Any] = None
        self.surya_recognition: Optional[Any] = None
        self._tesseract_runtime_ok: Optional[bool] = None

        self._configure_tesseract_cmd()
        self._tesseract_runtime_ok = self._probe_tesseract_runtime()

        should_init_surya = self.config.use_surya or (
            self.config.auto_surya_fallback and not self._tesseract_runtime_ok
        )

        if should_init_surya:
            logging.warning("Surya OCR support is best-effort because its API changes often.")
            self._try_init_surya()

        if self.config.use_tesseract_fallback and pytesseract is None:
            logging.warning("pytesseract is not installed; OCR fallback disabled.")

        if (
            self.config.use_tesseract_fallback
            and pytesseract is not None
            and not self._tesseract_runtime_ok
        ):
            logging.warning(
                "Tesseract runtime is unavailable. Install it, set ARGUSAI_TESSERACT_CMD, "
                "or use the Docker image where Tesseract is bundled."
            )

    def _configure_tesseract_cmd(self) -> None:
        """Apply an explicit Tesseract binary path when configured."""

        if pytesseract is None or not self.config.tesseract_cmd:
            return

        try:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
        except Exception as exc:
            logging.warning("Unable to configure Tesseract command: %s", exc)

    def _probe_tesseract_runtime(self) -> bool:
        """Return True only when pytesseract can reach a Tesseract binary."""

        if not self.config.use_tesseract_fallback or pytesseract is None:
            return False

        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as exc:
            logging.info("Tesseract runtime probe failed: %s", exc)
            return False

    def has_surya(self) -> bool:
        """Return True if Surya OCR components are initialized."""

        return self.surya_detection is not None and self.surya_recognition is not None

    def has_tesseract(self) -> bool:
        """Return True if pytesseract and the Tesseract runtime are usable."""

        return bool(
            self.config.use_tesseract_fallback
            and pytesseract is not None
            and self._tesseract_runtime_ok
        )

    def available_backend(self) -> str:
        """Return the active OCR backend name."""

        if self.has_tesseract() and self.has_surya():
            return "tesseract+surya"

        if self.has_tesseract():
            return "tesseract"

        if self.has_surya():
            return "surya"

        return "off"

    def _instantiate_surya_component(
        self,
        cls: Any,
        *possible_args: Any,
    ) -> Optional[Any]:
        """
        Try several constructor signatures because Surya APIs change often.

        This avoids hard-failing ArgusAI startup when Surya changes its API.
        """

        if cls is None:
            return None

        attempts = [
            tuple(),
            possible_args,
            tuple(arg for arg in possible_args if arg is not None),
        ]

        for args in attempts:
            try:
                return cls(*args)
            except TypeError:
                continue
            except Exception as exc:
                logging.warning("Surya component initialization failed: %s", exc)
                return None

        return None

    def _try_init_surya(self) -> None:
        """Initialize Surya components if available."""

        if not SURYA_AVAILABLE:
            logging.warning("Surya is not available.")
            return

        try:
            self.surya_foundation = self._instantiate_surya_component(
                FoundationPredictor,
            )

            self.surya_detection = self._instantiate_surya_component(
                DetectionPredictor,
                self.surya_foundation,
            )

            self.surya_recognition = self._instantiate_surya_component(
                RecognitionPredictor,
                self.surya_foundation,
            )

            if self.has_surya():
                logging.info("Surya OCR initialized.")
                return

            logging.warning(
                "Surya partially unavailable or constructor mismatch; "
                "using pytesseract fallback when available."
            )
            self.surya_detection = None
            self.surya_recognition = None

        except Exception as exc:
            logging.warning("Surya initialization failed: %s", exc)
            self.surya_foundation = None
            self.surya_detection = None
            self.surya_recognition = None

    def ocr_pil_image(self, image: Any) -> str:
        """
        Run OCR on a PIL image.

        Priority:
        1. pytesseract/Tesseract for MVP stability
        2. Surya if enabled and initialized
        """

        if Image is None:
            return ""

        if self.has_tesseract():
            text = self._ocr_with_tesseract(image)
            if text:
                return text

        if self.has_surya():
            text = self._ocr_with_surya(image)
            if text:
                return text

        return ""

    def _ocr_with_surya(self, image: Any) -> str:
        """Run OCR with Surya if available."""

        try:
            if getattr(image, "mode", None) != "RGB":
                image = image.convert("RGB")

            recognition = cast(Any, self.surya_recognition)

            try:
                predictions = recognition(
                    [image],
                    det_predictor=self.surya_detection,
                )
            except TypeError:
                predictions = recognition([image])

            lines: list[str] = []

            for prediction in predictions:
                text_lines = (
                    getattr(prediction, "text_lines", None)
                    or getattr(prediction, "lines", None)
                    or []
                )

                for line in text_lines:
                    lines.append(getattr(line, "text", str(line)))

            return clean_text("\n".join(lines))

        except Exception as exc:
            logging.warning("Surya OCR failed: %s", exc)
            return ""

    def _ocr_with_tesseract(self, image: Any) -> str:
        """Run OCR with pytesseract if available."""

        if pytesseract is None:
            return ""

        try:
            return clean_text(pytesseract.image_to_string(image))
        except Exception as exc:
            logging.warning("Tesseract OCR failed: %s", exc)
            return ""

    def run_ocr_on_path(self, path: str, pdf_loader: "DocumentLoader") -> str:
        """
        Run OCR on an image path or PDF path.

        PDF OCR is delegated to DocumentLoader because rendering PDF pages
        belongs to documents.py.
        """

        file_path = Path(path)

        if not file_path.exists():
            return "OCR error: file not found."

        if file_path.suffix.lower() == ".pdf":
            return pdf_loader.extract_pdf_text_with_ocr(file_path, self)

        return self.run_ocr_on_image_path(file_path)

    def run_ocr_on_image_path(self, path: Path) -> str:
        """Run OCR on an image file."""

        if Image is None:
            return "OCR unavailable because Pillow is not installed."

        try:
            image = Image.open(str(path))
        except Exception as exc:
            logging.warning("Unable to open image for OCR: %s", exc)
            return "OCR unavailable."

        text = self.ocr_pil_image(image)

        return text or "OCR unavailable."
