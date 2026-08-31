"""Web search provider for routing online when workspace documentation is insufficient."""

from __future__ import annotations

import re
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    score: float = 1.0


class WebSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 4) -> list[WebSearchResult]:
        """Perform a web search and return structured results."""
        ...


class FakeWebSearch(WebSearchProvider):
    """Deterministic web search provider for offline development and tests.

    Returns synthetic but relevant snippets based on the query keywords
    without making any external network requests.
    """

    def search(self, query: str, max_results: int = 4) -> list[WebSearchResult]:
        cleaned = query.strip()
        if not cleaned:
            return []

        slug = re.sub(r"[^\w\s-]", "", cleaned).strip().lower()
        slug = re.sub(r"[-\s]+", "-", slug)
        encoded = urllib.parse.quote_plus(cleaned)

        results = [
            WebSearchResult(
                title=f"Online Reference for '{cleaned}'",
                url=f"https://www.example.org/search?q={encoded}",
                snippet=(
                    f"Comprehensive online knowledge and guide about {cleaned}. "
                    "Provides general specifications, industry best practices, "
                    "and standard protocols."
                ),
                score=0.95,
            ),
            WebSearchResult(
                title=f"Community Knowledge Base: {cleaned}",
                url=f"https://docs.example.org/kb/{slug or 'general'}",
                snippet=(
                    f"Frequently asked questions and public resources covering {cleaned}. "
                    f"Includes troubleshooting steps and external reference links."
                ),
                score=0.88,
            ),
        ]
        return results[:max_results]


class DuckDuckGoWebSearch(WebSearchProvider):
    """Real web search powered by DuckDuckGo (no API key required)."""

    def search(self, query: str, max_results: int = 4) -> list[WebSearchResult]:
        cleaned = query.strip()
        if not cleaned:
            return []

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw_results = list(ddgs.text(cleaned, max_results=max_results))

            results: list[WebSearchResult] = []
            for r in raw_results:
                title = r.get("title") or "Web Search Result"
                url = r.get("href") or r.get("link") or ""
                snippet = r.get("body") or r.get("snippet") or ""
                if snippet:
                    results.append(
                        WebSearchResult(
                            title=title.strip(),
                            url=url.strip(),
                            snippet=snippet.strip(),
                            score=0.9,
                        )
                    )
            return results
        except Exception as exc:
            log.warning("web_search.ddgs_failed", query=cleaned, error=str(exc))
            # Fallback to FakeWebSearch if offline / rate-limited during non-production
            if not settings.is_production:
                return FakeWebSearch().search(cleaned, max_results=max_results)
            return []


class TavilyWebSearch(WebSearchProvider):
    """Tavily search provider for production AI web search."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.TAVILY_API_KEY

    def search(self, query: str, max_results: int = 4) -> list[WebSearchResult]:
        cleaned = query.strip()
        if not cleaned:
            return []

        if not self.api_key:
            log.warning("web_search.tavily_missing_key", query=cleaned)
            return DuckDuckGoWebSearch().search(cleaned, max_results=max_results)

        try:
            import httpx

            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": cleaned,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            results: list[WebSearchResult] = []
            for r in data.get("results", []):
                results.append(
                    WebSearchResult(
                        title=r.get("title", "Web Result"),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                        score=float(r.get("score", 0.9)),
                    )
                )
            return results
        except Exception as exc:
            log.warning("web_search.tavily_failed", query=cleaned, error=str(exc))
            return DuckDuckGoWebSearch().search(cleaned, max_results=max_results)


@lru_cache
def get_web_search_provider() -> WebSearchProvider:
    """Return the configured web search provider singleton."""
    if settings.LLM_PROVIDER == "fake" or settings.WEB_SEARCH_PROVIDER == "fake":
        return FakeWebSearch()
    if settings.WEB_SEARCH_PROVIDER == "tavily" and settings.TAVILY_API_KEY:
        return TavilyWebSearch()
    return DuckDuckGoWebSearch()


def reset_web_search_provider_cache() -> None:
    """Drop cached search provider (used by tests)."""
    get_web_search_provider.cache_clear()
