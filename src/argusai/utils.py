"""
General utility functions for ArgusAI.

This module contains lightweight helpers only:
- text cleaning
- JSON parsing
- filename/path helpers
- logging setup
- JSONL trace writing
- basic CLI colors

Heavy dependencies must not be imported globally here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import sys
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, cast

from .config import Config, RouteType


class Colors:
    """Minimal ANSI color palette for the CLI."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Text / JSON helpers
# ---------------------------------------------------------------------------


def clean_model_text(text: str) -> str:
    """
    Remove hidden reasoning blocks like <think>...</think> safely.

    Handles both closed and unclosed thinking blocks.
    """

    if not text:
        return ""

    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)

    return text.strip()


def safe_json_loads(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from a model response.

    If the response contains extra text, try to extract the first JSON object.
    """

    raw = (raw or "").strip()
    if not raw:
        return {}

    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float and reject NaN/inf."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def normalize_text(text: str) -> str:
    """Normalize text for lightweight matching."""

    return re.sub(r"\s+", " ", (text or "").strip().lower())


def clean_text(text: str) -> str:
    """Clean extracted OCR/PDF/web text."""

    text = (text or "").replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def has_arabic(text: str) -> bool:
    """Return True if text contains Arabic script characters."""

    if not isinstance(text, str):
        return False

    return any(
        "\u0600" <= char <= "\u06FF"
        or "\u0750" <= char <= "\u077F"
        or "\u08A0" <= char <= "\u08FF"
        for char in text
    )


def format_cli_text(text: str) -> str:
    """
    Fix Arabic RTL display only for terminal output.

    Never use this before memory writes, Qdrant ingestion, artifact saving,
    document exports, prompts, logs, or traces.
    """

    if not isinstance(text, str):
        return text

    if not has_arabic(text):
        return text

    try:
        import arabic_reshaper  # type: ignore[import-untyped]
        from bidi.algorithm import get_display  # type: ignore[import-untyped]

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Return True if any keyword is present in text."""

    normalized = normalize_text(text)
    return any(keyword.lower() in normalized for keyword in keywords)


def is_affirmative(text: str) -> bool:
    """Accept English and French yes-like confirmations."""

    return str(text or "").strip().lower() in {
        "y",
        "yes",
        "o",
        "oui",
        "true",
        "1",
    }


# ---------------------------------------------------------------------------
# Web / URL helpers
# ---------------------------------------------------------------------------


def extract_urls(text: str) -> List[str]:
    """Extract HTTP/HTTPS URLs from text."""

    return re.findall(r"https?://[^\s]+", text or "")


def has_explicit_web_request(text: str) -> bool:
    """
    Detect explicit URL-based web intent.

    Semantic web-search routing is handled by the router model.
    This helper only detects explicit HTTP/HTTPS URLs.
    """

    return bool(extract_urls(text))


# ---------------------------------------------------------------------------
# File / path helpers
# ---------------------------------------------------------------------------


def encode_file_base64(path: str) -> str:
    """Encode a file as base64 for Ollama vision requests."""

    with open(path, "rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")


def document_signature(path: Path) -> str:
    """Create a stable signature for document ingestion deduplication."""

    stat = path.stat()
    raw = f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def detect_requested_filename(text: str) -> Optional[str]:
    """
    Detect whether the user requested generation of a specific file.

    Examples:
    - "génère un fichier app.py"
    - "create file Dockerfile"
    - "nommé config.json"
    """

    if not text:
        return None

    patterns = [
        r"(?:fichier|file)\s+[\"']?([A-Za-z0-9._\-]+(?:\.[A-Za-z0-9._\-]+)?|Dockerfile|\.dockerignore|\.gitignore)[\"']?",
        r"(?:nomm[ée]|named|appel[ée])\s+[\"']?([A-Za-z0-9._\-]+(?:\.[A-Za-z0-9._\-]+)?|Dockerfile|\.dockerignore|\.gitignore)[\"']?",
        r"(?:g[eé]n[eè]re(?:r)?|cr[eé]e(?:r)?|create|make|write)\s+(?:moi\s+)?(?:un\s+|une\s+|a\s+)?(?:fichier|file)?\s*[\"']?([A-Za-z0-9._\-]+(?:\.[A-Za-z0-9._\-]+)?|Dockerfile|\.dockerignore|\.gitignore)[\"']?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        candidate = match.group(1).strip()
        lower_candidate = candidate.lower()

        if "." in candidate or lower_candidate in {
            "dockerfile",
            ".dockerignore",
            ".gitignore",
        }:
            return candidate

    return None


def get_image_info(path: Optional[str]) -> Dict[str, Any]:
    """
    Return basic image metadata.

    Pillow is imported lazily to keep utils lightweight.
    """

    if not path:
        return {}

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return {}

    try:
        with Image.open(path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
            }
    except Exception:
        return {}


def get_doc_info(path: Optional[str]) -> Dict[str, Any]:
    """Return basic document metadata."""

    if not path:
        return {}

    p = Path(path)

    return {
        "exists": p.exists(),
        "suffix": p.suffix.lower(),
        "is_pdf": p.suffix.lower() == ".pdf",
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def parse_route_list(values: Any) -> List[RouteType]:
    """Parse and validate a list of route names returned by an LLM."""

    allowed: Set[str] = {
        "reasoning",
        "code",
        "vision",
        "ocr",
        "memory",
        "web",
        "web_search",
        "web_scrape",
        "hybrid",
    }

    if not isinstance(values, list):
        return []

    routes: List[RouteType] = []

    for item in values:
        route = str(item).strip().lower()
        if route in allowed:
            routes.append(cast(RouteType, route))

    return routes[:3]


def model_installed(requested: str, installed: Sequence[str]) -> bool:
    """
    Check whether an Ollama model tag is installed.

    Accepts exact tag match or base model match.
    Example:
    requested='deepseek-r1:8b'
    installed=['deepseek-r1:latest']
    """

    req = requested.strip().lower()
    installed_lower = [name.strip().lower() for name in installed]

    if req in installed_lower:
        return True

    req_base = req.split(":", 1)[0]

    for model_name in installed_lower:
        model_base = model_name.split(":", 1)[0]
        if req_base == model_base:
            return True

    return False


# ---------------------------------------------------------------------------
# Logging / tracing
# ---------------------------------------------------------------------------


def setup_logging(config: Config) -> Path:
    """
    Configure ArgusAI logging.

    Logs are written to outputs/logs/argusai.log by default.
    Console output is kept quiet to avoid polluting the CLI UX.
    """

    config.ensure_runtime_dirs()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = RotatingFileHandler(
        config.log_file_path,
        maxBytes=1_500_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.CRITICAL)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    for noisy_logger in ("httpx", "httpcore", "qdrant_client", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return config.log_file_path


def new_run_id() -> str:
    """Create a compact unique run ID."""

    return str(uuid.uuid4())


def utc_timestamp() -> float:
    """Return current Unix timestamp."""

    return time.time()


def write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append a dictionary to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_trace(config: Config, payload: Dict[str, Any]) -> None:
    """
    Write one execution trace to outputs/traces/runs.jsonl.

    Tracing must never crash the main agent.
    """

    if not config.enable_traces:
        return

    try:
        trace_payload = {
            "run_id": payload.get("run_id") or new_run_id(),
            "timestamp": payload.get("timestamp") or utc_timestamp(),
            **payload,
        }
        write_jsonl(config.traces_file_path, trace_payload)
    except Exception as exc:
        logging.warning("Failed to write ArgusAI trace: %s", exc)
