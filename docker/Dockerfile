# ArgusAI v0.1.0-beta
# Local-first multimodal agentic AI assistant

FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# ---------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------
# tesseract-ocr      -> OCR engine used by pytesseract
# tesseract-ocr-eng  -> English OCR language data
# tesseract-ocr-fra  -> French OCR language data
# tesseract-ocr-ara  -> Arabic OCR language data
# curl               -> health/debug utility and compose wait loop
# build-essential    -> useful for Python packages that compile wheels
# ca-certificates    -> HTTPS access for Python packages and web tools
# ---------------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
        tesseract-ocr-ara \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
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

# Install ArgusAI package without re-resolving dependencies already installed
# from requirements.txt.
RUN pip install --no-deps -e . \
    && python -c "import argusai; print('ArgusAI', argusai.__version__)"

# ---------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------
# In Docker Compose:
# - Qdrant is reachable at http://qdrant:6333
# - Ollama is expected to run on the host at http://host.docker.internal:11434
# ---------------------------------------------------------------------

ENV ARGUSAI_QDRANT_URL=http://qdrant:6333 \
    ARGUSAI_OLLAMA_URL=http://host.docker.internal:11434 \
    ARGUSAI_ARTIFACTS_DIR=/app/outputs \
    ARGUSAI_TESSERACT_CMD=/usr/bin/tesseract

VOLUME ["/app/outputs"]

CMD ["argusai", "--no-memory-prompt", "--request-timeout", "900"]
