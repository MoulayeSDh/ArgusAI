"""
ArgusAI.

Local-first multimodal agentic AI assistant.

ArgusAI is designed as a CLI-first and Docker-friendly project using:
- Ollama for local LLM inference
- Qdrant for optional semantic memory
- OCR for image/document text extraction
- Optional web crawling with explicit user control

This package intentionally keeps __init__.py lightweight.
Heavy components such as Ollama, Qdrant, OCR, and Crawl4AI are imported only
inside their dedicated modules.
"""

__app_name__ = "ArgusAI"
__version__ = "0.1.0-beta"
__author__ = "Moulaye S. Dahi"
__license__ = "PolyForm Noncommercial 1.0.0"

__all__ = [
    "__app_name__",
    "__version__",
    "__author__",
    "__license__",
]
