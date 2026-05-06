"""
Ollama HTTP client for ArgusAI.

This module contains a small wrapper around the local Ollama API.
It is intentionally minimal and does not depend on the rest of the agent logic.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import requests

from .utils import clean_model_text, safe_json_loads


class OllamaClient:
    """Small HTTP wrapper around the local Ollama API."""

    def __init__(
        self,
        base_url: str,
        request_timeout: int = 180,
        keep_alive: str = "10m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.keep_alive = keep_alive

    def list_models(self) -> List[str]:
        """
        Return locally installed Ollama model tags.

        Raises:
            requests.HTTPError: If Ollama returns an error response.
            requests.RequestException: If Ollama is unreachable.
        """

        url = f"{self.base_url}/api/tags"
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        return [str(model.get("name", "")) for model in models if model.get("name")]

    def generate_text(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[str]] = None,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
        temperature: float = 0.2,
        think: bool = True,
    ) -> str:
        """
        Generate plain text with Ollama.

        Args:
            model: Ollama model tag.
            prompt: Prompt text.
            images: Optional base64-encoded images for vision models.
            stream: Whether to stream chunks.
            callback: Optional callback called for each streamed chunk.
            temperature: Sampling temperature.
            think: Whether to allow thinking mode when supported.

        Returns:
            Cleaned model response.
        """

        if stream and think:
            raise ValueError(
                "Streaming with think=True is disabled to avoid leaking partial <think> blocks."
            )

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "think": bool(think),
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
            },
        }

        if images:
            payload["images"] = images

        url = f"{self.base_url}/api/generate"

        if not stream:
            response = requests.post(
                url,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            data = response.json()
            return clean_model_text(str(data.get("response", "")))

        return self._stream_generate(url=url, payload=payload, callback=callback, think=think)

    def generate_json(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        *,
        images: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON with Ollama.

        Args:
            model: Ollama model tag.
            prompt: Prompt text.
            schema: JSON schema sent through Ollama's format parameter.
            images: Optional base64-encoded images for vision-capable models.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dictionary. Returns an empty dict if parsing fails.
        """

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
            },
        }

        if images:
            payload["images"] = images

        url = f"{self.base_url}/api/generate"

        response = requests.post(
            url,
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        data = response.json()
        raw = str(data.get("response") or data.get("thinking") or "{}")
        return safe_json_loads(raw)

    def ping(self) -> bool:
        """
        Check whether Ollama is reachable.

        Returns:
            True if /api/tags responds successfully, False otherwise.
        """

        try:
            self.list_models()
            return True
        except requests.RequestException:
            return False

    def _stream_generate(
        self,
        *,
        url: str,
        payload: Dict[str, Any],
        callback: Optional[Callable[[str], None]],
        think: bool,
    ) -> str:
        """
        Internal streaming generation helper.

        Ollama streaming responses are JSONL-like chunks.
        """

        parts: List[str] = []

        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=self.request_timeout,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk = str(data.get("response", ""))

                if not think:
                    chunk = clean_model_text(chunk)

                if chunk:
                    parts.append(chunk)

                    if callback is not None:
                        callback(chunk)

                if data.get("done"):
                    break

        return "".join(parts).strip()
