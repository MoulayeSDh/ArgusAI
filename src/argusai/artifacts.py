"""
Artifact management for ArgusAI.

This module handles generated files safely:
- sanitize requested filenames
- extract file content from model answers
- save only inside the configured outputs directory
- validate basic file formats after generation
"""

from __future__ import annotations

import json
import logging
import py_compile
import re
from pathlib import Path
from typing import Optional, Set

from .config import ArtifactResult


class FileArtifactManager:
    """Write and validate generated text/code artifacts."""

    def __init__(self, base_dir: str = "outputs", max_chars: int = 200_000) -> None:
        self.base_dir = Path(base_dir)
        self.max_chars = max_chars

        self.allowed_extensions: Set[str] = {
            ".py",
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".sh",
            ".bash",
            ".ps1",
            ".sql",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".xml",
            ".csv",
        }

        self.allowed_special_names: Set[str] = {
            "dockerfile",
            ".dockerignore",
            ".gitignore",
            ".env.example",
        }

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a user/model-provided filename.

        Rules:
        - remove directory traversal
        - keep only safe characters
        - allow known special files such as Dockerfile and .gitignore
        - reject unsupported extensions
        """

        candidate = (filename or "").strip().strip('"').strip("'")
        candidate = candidate.replace("\\", "/").split("/")[-1]
        candidate = re.sub(r"\s+", "_", candidate)
        candidate = re.sub(r"[^A-Za-z0-9._\-]", "", candidate)

        if not candidate or candidate in {".", ".."} or ".." in candidate:
            raise ValueError("Invalid filename.")

        lower_name = candidate.lower()

        if lower_name in self.allowed_special_names:
            return candidate

        suffix = Path(candidate).suffix.lower()

        if not suffix or suffix not in self.allowed_extensions:
            raise ValueError(f"Unsupported filename or extension: {candidate}")

        return candidate

    def extract_file_content(self, answer: str) -> str:
        """
        Extract artifact content from an LLM answer.

        If the answer contains fenced code blocks, use the first block.
        Otherwise, save the raw answer.
        """

        if not answer or not answer.strip():
            raise ValueError("Generated answer is empty; nothing to save.")

        code_blocks = re.findall(
            r"```(?:[A-Za-z0-9_+\-.#]*)\n([\s\S]*?)```",
            answer,
        )

        content = code_blocks[0].strip() if code_blocks else answer.strip()

        if len(content) > self.max_chars:
            raise ValueError(f"Generated content is too large (> {self.max_chars} chars).")

        if content and not content.endswith("\n"):
            content += "\n"

        return content

    def save_text(self, filename: str, content: str) -> Path:
        """
        Save content into the artifacts directory.

        The final path is always base_dir / sanitized_filename.
        """

        safe_name = self.sanitize_filename(filename)
        path = self.base_dir / safe_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return path

    def save_from_answer(self, filename: str, answer: str) -> ArtifactResult:
        """
        Extract content from a model answer, save it, then validate it.

        Returns:
            ArtifactResult with saved path and validation status.
        """

        try:
            content = self.extract_file_content(answer)
            saved_path = self.save_text(filename, content)
            validation_ok, validation_error = self.validate_artifact(saved_path)

            return ArtifactResult(
                requested=True,
                filename=filename,
                saved_path=saved_path.as_posix(),
                saved_content=content,
                validation_ok=validation_ok,
                validation_error=validation_error,
                error=None,
            )

        except Exception as exc:
            logging.warning("Artifact generation failed: %s", exc)

            return ArtifactResult(
                requested=True,
                filename=filename,
                saved_path=None,
                saved_content=None,
                validation_ok=False,
                validation_error=None,
                error=str(exc),
            )

    def validate_artifact(self, path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate a generated artifact based on its file type.

        Current validation:
        - Python: py_compile
        - JSON: json.loads
        - YAML/YML: yaml.safe_load if PyYAML is installed
        - Shell/PowerShell: simple dangerous command scan
        """

        suffix = path.suffix.lower()
        lower_name = path.name.lower()

        try:
            if suffix == ".py":
                py_compile.compile(str(path), doraise=True)
                return True, None

            if suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                return True, None

            if suffix in {".yaml", ".yml"}:
                return self._validate_yaml(path)

            if suffix in {".sh", ".bash", ".ps1"}:
                return self._validate_shell_like_script(path)

            if lower_name in {"dockerfile", ".dockerignore", ".gitignore", ".env.example"}:
                return True, None

            return True, None

        except Exception as exc:
            return False, str(exc)

    def _validate_yaml(self, path: Path) -> tuple[bool, Optional[str]]:
        """Validate YAML if PyYAML is available."""

        try:
            import yaml  # type: ignore
        except ImportError:
            return True, "PyYAML is not installed; YAML syntax was not validated."

        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _validate_shell_like_script(self, path: Path) -> tuple[bool, Optional[str]]:
        """
        Run a conservative scan for obviously dangerous shell commands.

        This is not a full security sandbox. It only catches common high-risk
        patterns before the user executes generated scripts.
        """

        content = path.read_text(encoding="utf-8", errors="ignore").lower()

        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r"rm\s+-rf\s+\*",
            r"del\s+/s",
            r"format\s+[a-z]:",
            r"mkfs\.",
            r"shutdown\s+",
            r"reboot\s*",
            r"curl\s+.*\|\s*sh",
            r"wget\s+.*\|\s*sh",
            r"invoke-expression",
            r"iex\s*\(",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, content):
                return False, f"Potentially dangerous command detected: {pattern}"

        return True, None


def append_artifact_note(answer: str, artifact: ArtifactResult, enabled: bool = True) -> str:
    """
    Append a short artifact status note to the final answer.

    This keeps agent.py cleaner.
    """

    if not enabled or not artifact.requested:
        return answer

    if artifact.saved_path:
        note = f"\n\n[artifact saved: {artifact.saved_path}]"

        if artifact.validation_ok is False:
            note += f"\n[artifact validation failed: {artifact.validation_error}]"
        elif artifact.validation_error:
            note += f"\n[artifact validation warning: {artifact.validation_error}]"

        if note not in answer:
            return answer.rstrip() + note

    if artifact.error:
        note = f"\n\n[artifact generation failed: {artifact.error}]"

        if note not in answer:
            return answer.rstrip() + note

    return answer