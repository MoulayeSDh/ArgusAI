from argusai.config import Config
from argusai import ocr as ocr_module
from argusai.ocr import OCRManager


class FallbackOCRManager(OCRManager):
    def __init__(self):
        self.config = Config(use_tesseract_fallback=True)
        self.surya_detection = object()
        self.surya_recognition = object()
        self.tesseract_text = ""
        self.surya_text = ""
        self.calls = []

    def has_tesseract(self):
        return True

    def has_surya(self):
        return True

    def _ocr_with_tesseract(self, image):
        self.calls.append("tesseract")
        return self.tesseract_text

    def _ocr_with_surya(self, image):
        self.calls.append("surya")
        return self.surya_text


def test_ocr_uses_tesseract_before_surya():
    ocr = FallbackOCRManager()
    ocr.tesseract_text = "text from tesseract"
    ocr.surya_text = "text from surya"

    assert ocr.ocr_pil_image(object()) == "text from tesseract"
    assert ocr.calls == ["tesseract"]


def test_ocr_falls_back_to_surya_when_tesseract_returns_no_text():
    ocr = FallbackOCRManager()
    ocr.tesseract_text = ""
    ocr.surya_text = "text from surya"

    assert ocr.ocr_pil_image(object()) == "text from surya"
    assert ocr.calls == ["tesseract", "surya"]


def test_available_backend_reports_tesseract_then_surya_chain():
    ocr = FallbackOCRManager()

    assert ocr.available_backend() == "tesseract+surya"


def test_tesseract_requires_runtime_probe(monkeypatch):
    class FakePyTesseract:
        class pytesseract:
            tesseract_cmd = "tesseract"

        @staticmethod
        def get_tesseract_version():
            raise RuntimeError("missing binary")

    monkeypatch.setattr(ocr_module, "pytesseract", FakePyTesseract)

    manager = OCRManager(Config(auto_surya_fallback=False))

    assert manager.has_tesseract() is False


def test_auto_surya_fallback_initializes_when_tesseract_runtime_missing(monkeypatch):
    class FakePyTesseract:
        class pytesseract:
            tesseract_cmd = "tesseract"

        @staticmethod
        def get_tesseract_version():
            raise RuntimeError("missing binary")

    calls = []

    monkeypatch.setattr(ocr_module, "pytesseract", FakePyTesseract)
    monkeypatch.setattr(ocr_module, "SURYA_AVAILABLE", True)
    monkeypatch.setattr(OCRManager, "_try_init_surya", lambda self: calls.append("surya"))

    OCRManager(Config(auto_surya_fallback=True))

    assert calls == ["surya"]
