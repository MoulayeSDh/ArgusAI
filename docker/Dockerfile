# ArgusAI v0.1.0-beta
# Local-first multimodal agentic AI assistant

FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV PIP_NO_CACHE_DIR=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /app

# ---------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------
# tesseract-ocr      -> OCR engine used by pytesseract
# tesseract-ocr-eng  -> English OCR language
# tesseract-ocr-fra  -> French OCR language
# curl               -> health/debug utility
# build-essential    -> useful for Python packages that need compilation
# ---------------------------------------------------------------------

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Crawl4AI uses Playwright for URL scraping. Installing Chromium in the image
# keeps Docker usage self-contained instead of requiring host browser setup.
RUN python -m playwright install --with-deps chromium

# ---------------------------------------------------------------------
# Project files
# ---------------------------------------------------------------------

COPY pyproject.toml .
COPY LICENSE.md .
COPY README.md .
COPY src ./src

# Install ArgusAI as an editable package.
RUN pip install -e .

# ---------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------
# In Docker Compose:
# - Qdrant will be reachable at http://qdrant:6333
# - Ollama is expected to run on the host at http://host.docker.internal:11434
# ---------------------------------------------------------------------

ENV ARGUSAI_QDRANT_URL=http://qdrant:6333
ENV ARGUSAI_OLLAMA_URL=http://host.docker.internal:11434
ENV ARGUSAI_ARTIFACTS_DIR=/app/outputs

VOLUME ["/app/outputs"]

CMD ["argusai", "--no-memory-prompt", "--request-timeout", "900"]
