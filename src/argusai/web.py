"""
Optional web crawling utilities for ArgusAI.

Current MVP behavior:
- Web access is disabled by default.
- Web access must be enabled through config/CLI.
- This module only crawls explicit URLs.
- It does not perform general web search yet.

Future improvement:
- Add a SearchWebTool separate from CrawlUrlTool.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional

from .config import Config
from .utils import clean_text, extract_urls


# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # type: ignore
    from crawl4ai.async_logger import AsyncLogger  # type: ignore

    CRAWL4AI_AVAILABLE = True
except Exception:
    AsyncWebCrawler = None  # type: ignore[assignment]
    BrowserConfig = None  # type: ignore[assignment]
    CrawlerRunConfig = None  # type: ignore[assignment]
    AsyncLogger = None  # type: ignore[assignment]
    CRAWL4AI_AVAILABLE = False


class WebCrawler:
    """Optional Crawl4AI-based URL crawler."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def is_available(self) -> bool:
        """Return True if Crawl4AI is installed."""

        return CRAWL4AI_AVAILABLE and AsyncWebCrawler is not None

    def is_enabled(self) -> bool:
        """Return True if web crawling is enabled and available."""

        return (
            self.config.web_enabled
            and self.config.crawl4ai_enabled
            and self.is_available()
        )

    def status(self) -> str:
        """Return web tool status for CLI display."""

        if not self.config.web_enabled:
            return "off"

        if not self.config.crawl4ai_enabled:
            return "off"

        if not self.is_available():
            return "unavailable"

        return "armed"

    def web_context(self, user_text: str, max_urls: int = 2) -> str:
        """
        Crawl explicit URLs found in the user request.

        Args:
            user_text: User request containing one or more URLs.
            max_urls: Maximum number of URLs to crawl.

        Returns:
            Text context extracted from the URLs.
        """

        if not self.is_enabled():
            return "Web tool disabled."

        urls = extract_urls(user_text)

        if not urls:
            return (
                "Web tool currently expects explicit URL(s) in the user request. "
                "General web search is not implemented in this MVP."
            )

        selected_urls = urls[:max_urls]

        try:
            return asyncio.run(self.crawl_urls(selected_urls))
        except RuntimeError as exc:
            logging.warning("Async runtime conflict during web crawl: %s", exc)
            return f"Web crawl failed because of an async runtime conflict: {exc}"
        except Exception as exc:
            logging.warning("Web crawl failed: %s", exc)
            return f"Web crawl failed: {exc}"

    async def crawl_urls(self, urls: List[str]) -> str:
        """
        Crawl multiple URLs asynchronously.

        Returns:
            Combined cleaned content snippets.
        """

        if not self.is_enabled():
            return "Web tool disabled."

        if AsyncWebCrawler is None:
            return "Crawl4AI unavailable."

        results: List[str] = []

        # Crawl4AI prints Unicode progress markers by default. On Windows
        # cp1252 consoles that can fail before crawling starts, so keep the
        # crawler quiet and prefer UTF-8 for subprocess IO.
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

        crawler_kwargs: dict[str, Any] = {}
        if BrowserConfig is not None:
            crawler_kwargs["config"] = BrowserConfig(headless=True, verbose=False)
        if AsyncLogger is not None:
            crawler_kwargs["logger"] = AsyncLogger(verbose=False)

        async with AsyncWebCrawler(**crawler_kwargs) as crawler:
            for url in urls:
                result = await self._crawl_single_url(crawler, url)
                results.append(result)

        return "\n\n".join(results)

    async def _crawl_single_url(self, crawler: Any, url: str) -> str:
        """Crawl one URL and return a cleaned text snippet."""

        try:
            run_kwargs: dict[str, Any] = {"url": url}
            if CrawlerRunConfig is not None:
                run_kwargs["config"] = CrawlerRunConfig(verbose=False, log_console=False)

            result = await crawler.arun(**run_kwargs)

            markdown = (
                getattr(result, "markdown", None)
                or getattr(result, "cleaned_html", None)
                or ""
            )

            snippet = clean_text(str(markdown))[:2500]

            if not snippet:
                return f"URL: {url}\nContent: <empty extraction>"

            return f"URL: {url}\nContent:\n{snippet}"

        except Exception as exc:
            logging.warning("Failed to crawl %s: %s", url, exc)
            return f"URL: {url}\nError: {exc}"
