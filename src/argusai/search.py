"""
Web search utilities for ArgusAI.

This module provides a lightweight, optional web search tool based on `ddgs`.
It is designed for general web search queries, not URL scraping.

Design:
- SearchWebTool -> general web search, no API key, no Docker required.
- WebCrawler/Crawl4AI -> URL-specific scraping, handled separately in web.py.

If `ddgs` is not installed, the tool fails gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast


try:
    from ddgs import DDGS as DDGSClient  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    DDGSClient = None  # type: ignore[assignment]


@dataclass
class WebSearchResult:
    """Normalized web search result."""

    title: str
    url: str
    snippet: str
    source: str = "ddgs"
    raw: Dict[str, Any] = field(default_factory=dict)


class SearchWebTool:
    """
    Lightweight web search tool using ddgs.

    This tool is for general internet search:
    - current weather
    - recent news
    - latest software versions
    - current public information

    It is not a scraper. URL crawling/scraping should stay in web.py with Crawl4AI.
    """

    def __init__(
        self,
        max_results: int = 5,
        region: str = "fr-fr",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        timeout: int = 10,
        backend: str = "auto",
        proxy: Optional[str] = None,
        verify: bool = True,
    ) -> None:
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch
        self.timelimit = timelimit
        self.timeout = timeout
        self.backend = backend
        self.proxy = proxy
        self.verify = verify

    def is_available(self) -> bool:
        """Return True if ddgs is installed and importable."""
        return DDGSClient is not None

    def status(self) -> str:
        """Return human-readable tool status."""
        if self.is_available():
            return "available"
        return "unavailable: install ddgs"

    def search(self, query: str, max_results: Optional[int] = None) -> List[WebSearchResult]:
        """
        Run a general web search.

        Args:
            query: User search query.
            max_results: Optional override for number of results.

        Returns:
            List of normalized WebSearchResult objects.
        """
        query = (query or "").strip()
        if not query:
            return []

        if not self.is_available():
            logging.warning("SearchWebTool unavailable: ddgs is not installed.")
            return []

        limit = max_results or self.max_results

        try:
            ddgs_cls = cast(Any, DDGSClient)
            client = ddgs_cls(
                proxy=self.proxy,
                timeout=self.timeout,
                verify=self.verify,
            )

            raw_results = client.text(
                query=query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.timelimit,
                max_results=limit,
                backend=self.backend,
            )

            return [self._normalize_result(item) for item in raw_results]

        except Exception as exc:
            logging.warning("Web search failed: %s", exc)
            return []

    def search_context(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Run search and return a compact text context for the LLM.

        This context should be injected into the final answer prompt.
        """
        results = self.search(query=query, max_results=max_results)

        if not results:
            return (
                "Web search produced no usable results. "
                "Do not invent current facts. Explain that web search failed or returned no results."
            )

        lines = [
            "Web search results:",
            "",
        ]

        for idx, item in enumerate(results, start=1):
            lines.extend(
                [
                    f"[{idx}] {item.title}",
                    f"URL: {item.url}",
                    f"Snippet: {item.snippet}",
                    "",
                ]
            )

        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_result(item: Dict[str, Any]) -> WebSearchResult:
        """Normalize ddgs raw result dictionary."""
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = str(item.get("body") or item.get("snippet") or "").strip()

        return WebSearchResult(
            title=title or "Untitled result",
            url=url,
            snippet=snippet,
            raw=dict(item),
        )
