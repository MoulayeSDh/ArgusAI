"""
Runtime configuration and shared schemas for ArgusAI.

This module must stay lightweight:
- no Ollama import
- no Qdrant import
- no OCR import
- no Crawl4AI import

Heavy dependencies are loaded only inside their dedicated modules.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set


RouteType = Literal[
    "reasoning",
    "code",
    "vision",
    "ocr",
    "memory",
    "web",
    "web_search",
    "web_scrape",
    "hybrid",
]


InputType = Literal[
    "text",
    "image",
    "document",
    "multimodal",
]


@dataclass
class Config:
    """
    Runtime configuration for ArgusAI.

    Defaults are optimized for local development.
    Docker Compose can override URLs through environment variables.
    """

    # ------------------------------------------------------------------
    # Runtime identity
    # ------------------------------------------------------------------
    app_name: str = "ArgusAI"
    version: str = "0.1.0-beta"

    # ------------------------------------------------------------------
    # Local services
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "argusai_v2_memory"

    # ------------------------------------------------------------------
    # Ollama models
    # ------------------------------------------------------------------
    router_vision_model: str = "qwen3-vl:8b"
    reasoning_model: str = "deepseek-r1:8b"
    judge_model: str = "qwen3.5:9b"
    coder_model: str = "qwen2.5-coder:7b"
    embedding_model: str = "qwen3-embedding:latest"

    # ------------------------------------------------------------------
    # Ollama request behavior
    # ------------------------------------------------------------------
    request_timeout: int = 900
    keep_alive: str = "10m"
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Memory / Qdrant
    # ------------------------------------------------------------------
    use_memory: bool = True
    ask_before_disable_memory: bool = True
    memory_top_k: int = 4
    buffer_k: int = 4

    # ------------------------------------------------------------------
    # Document chunking
    # ------------------------------------------------------------------
    chunk_size: int = 700
    chunk_overlap: int = 100
    max_doc_chars: int = 120_000
    max_memory_write_chars: int = 4_000

    # ------------------------------------------------------------------
    # Web access
    # ------------------------------------------------------------------
    web_enabled: bool = False
    crawl4ai_enabled: bool = False

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    use_surya: bool = False
    auto_surya_fallback: bool = True
    use_tesseract_fallback: bool = True
    tesseract_cmd: Optional[str] = None
    pdf_ocr_max_pages: int = 5
    pdf_ocr_scale: float = 2.0

    # ------------------------------------------------------------------
    # Judge / self-improvement
    # ------------------------------------------------------------------
    max_judge_rounds: int = 2
    judge_accept_threshold: float = 7.0
    enable_route_recheck: bool = True

    # ------------------------------------------------------------------
    # Artifacts and runtime outputs
    # ------------------------------------------------------------------
    artifacts_dir: str = "outputs"
    artifact_max_chars: int = 200_000
    artifact_note_in_answer: bool = True

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------
    enable_traces: bool = True
    traces_filename: str = "runs.jsonl"

    @classmethod
    def from_env(cls) -> "Config":
        """
        Build Config from environment variables.

        Useful for Docker Compose while keeping local defaults simple.
        """

        return cls(
            ollama_base_url=os.getenv("ARGUSAI_OLLAMA_URL", "http://localhost:11434"),
            qdrant_url=os.getenv("ARGUSAI_QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("ARGUSAI_QDRANT_COLLECTION", "argusai_v2_memory"),
            router_vision_model=os.getenv("ARGUSAI_ROUTER_MODEL", "qwen3-vl:8b"),
            reasoning_model=os.getenv("ARGUSAI_REASONING_MODEL", "deepseek-r1:8b"),
            judge_model=os.getenv("ARGUSAI_JUDGE_MODEL", "qwen3.5:9b"),
            coder_model=os.getenv("ARGUSAI_CODER_MODEL", "qwen2.5-coder:7b"),
            embedding_model=os.getenv("ARGUSAI_EMBEDDING_MODEL", "qwen3-embedding:latest"),
            artifacts_dir=os.getenv("ARGUSAI_ARTIFACTS_DIR", "outputs"),
            log_level=os.getenv("ARGUSAI_LOG_LEVEL", "INFO"),
            tesseract_cmd=os.getenv("ARGUSAI_TESSERACT_CMD") or None,
        )

    @property
    def artifacts_path(self) -> Path:
        """Base runtime output directory."""
        return Path(self.artifacts_dir)

    @property
    def logs_dir(self) -> Path:
        """Directory for rotating logs."""
        return self.artifacts_path / "logs"

    @property
    def traces_dir(self) -> Path:
        """Directory for JSONL execution traces."""
        return self.artifacts_path / "traces"

    @property
    def log_file_path(self) -> Path:
        """Main ArgusAI log file path."""
        return self.logs_dir / "argusai.log"

    @property
    def traces_file_path(self) -> Path:
        """JSONL trace file path."""
        return self.traces_dir / self.traces_filename

    def ensure_runtime_dirs(self) -> None:
        """Create runtime directories if they do not exist."""
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_traces:
            self.traces_dir.mkdir(parents=True, exist_ok=True)

    def required_models(self) -> Set[str]:
        """Return the set of Ollama model tags required by the current config."""
        return {
            self.router_vision_model,
            self.reasoning_model,
            self.judge_model,
            self.coder_model,
            self.embedding_model,
        }


@dataclass
class Message:
    """Short-term conversation message."""

    role: str
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class PreprocessedInput:
    """Normalized user request after CLI parsing."""

    input_type: InputType
    user_text: str
    normalized_text: str
    image_path: Optional[str] = None
    doc_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutePlan:
    """Execution plan chosen for the current request."""

    primary_route: RouteType
    reason: str
    confidence: float
    use_memory: bool = True
    use_ocr: bool = False
    use_web: bool = False
    requires_reasoning: bool = True
    secondary_routes: List[RouteType] = field(default_factory=list)


@dataclass
class ExecutionBundle:
    """Collected evidence and metadata before answer generation."""

    route: RouteType
    memory_context: str
    evidence: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeVerdict:
    """Judge model output."""

    score: float
    accept: bool
    critique: str
    improvement_instructions: str
    route_ok: bool = True


@dataclass
class ArtifactResult:
    """Information about a generated artifact."""

    requested: bool
    filename: Optional[str] = None
    saved_path: Optional[str] = None
    saved_content: Optional[str] = None
    validation_ok: Optional[bool] = None
    validation_error: Optional[str] = None
    error: Optional[str] = None
