"""Web search provider for routing online when workspace documentation is insufficient."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
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
                url=f"https://duckduckgo.com/?q={encoded}",
                snippet=(
                    f"Comprehensive online knowledge and guide about {cleaned}. "
                    "Provides general specifications, industry best practices, "
                    "and standard protocols."
                ),
                score=0.95,
            ),
            WebSearchResult(
                title=f"Community Knowledge Base: {cleaned}",
                url=f"https://en.wikipedia.org/wiki/Special:Search?search={encoded}",
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

        # 1. Try DuckDuckGo Lite endpoint (fast, zero-auth, live search results)
        results = self._search_lite(cleaned, max_results=max_results)
        if results:
            return results

        # 2. Try DuckDuckGo Instant Answer API (Wikipedia-grounded definitions)
        instant = self._search_instant_answer(cleaned)
        if instant:
            results.append(instant)

        # 3. Try ddgs news/text if available
        ddgs_results = self._search_ddgs(cleaned, max_results=max_results)
        if ddgs_results:
            results.extend(ddgs_results)

        if results:
            # Deduplicate by URL
            seen_urls: set[str] = set()
            unique_results: list[WebSearchResult] = []
            for r in results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)
            return unique_results[:max_results]

        # 4. Fallback to FakeWebSearch if offline / rate-limited during non-production
        if not settings.is_production:
            return FakeWebSearch().search(cleaned, max_results=max_results)
        return []

    def _search_lite(self, query: str, max_results: int = 4) -> list[WebSearchResult]:
        """Fetch search results from DuckDuckGo Lite HTML interface."""
        try:
            from bs4 import BeautifulSoup

            url = "https://lite.duckduckgo.com/lite/"
            data = urllib.parse.urlencode({"q": query}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            soup = BeautifulSoup(html, "html.parser")
            links = soup.find_all("a", class_="result-link")
            snippets = soup.find_all("td", class_="result-snippet")

            results: list[WebSearchResult] = []
            for link_el, snip_el in zip(links, snippets, strict=False):
                href = link_el.get("href", "").strip()
                if "uddg=" in href:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]

                title = link_el.get_text().strip()
                snippet = snip_el.get_text().strip()
                if href and title and snippet:
                    results.append(
                        WebSearchResult(
                            title=title,
                            url=href,
                            snippet=snippet,
                            score=0.92,
                        )
                    )
                if len(results) >= max_results:
                    break

            return results
        except Exception as exc:
            log.warning("web_search.ddg_lite_failed", query=query, error=str(exc))
            return []

    def _search_instant_answer(self, query: str) -> WebSearchResult | None:
        """Fetch DuckDuckGo Instant Answer abstract if available."""
        import json
        import urllib.request

        try:
            encoded = urllib.parse.quote_plus(query)
            api_url = (
                f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
            )
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))

            abstract = (data.get("AbstractText") or "").strip()
            heading = (data.get("Heading") or query).strip()
            source_url = (data.get("AbstractURL") or "").strip()
            if abstract and source_url:
                return WebSearchResult(
                    title=heading,
                    url=source_url,
                    snippet=abstract,
                    score=0.98,
                )
        except Exception as exc:
            log.warning("web_search.ddg_instant_failed", query=query, error=str(exc))
        return None

    def _search_ddgs(self, query: str, max_results: int = 4) -> list[WebSearchResult]:
        """Fallback to duckduckgo_search library news/text."""
        try:
            from duckduckgo_search import DDGS

            results: list[WebSearchResult] = []
            with DDGS() as ddgs:
                # Try news first as it returns current, direct articles
                news_items = list(ddgs.news(query, max_results=max_results))
                for item in news_items:
                    title = item.get("title") or "News Article"
                    url = item.get("url") or item.get("link") or ""
                    body = item.get("body") or item.get("snippet") or ""
                    if url and body:
                        results.append(
                            WebSearchResult(
                                title=title.strip(),
                                url=url.strip(),
                                snippet=body.strip(),
                                score=0.90,
                            )
                        )
                    if len(results) >= max_results:
                        return results
            return results
        except Exception as exc:
            log.warning("web_search.ddgs_failed", query=query, error=str(exc))
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
    if settings.WEB_SEARCH_PROVIDER == "fake":
        return FakeWebSearch()
    if settings.WEB_SEARCH_PROVIDER == "tavily" and settings.TAVILY_API_KEY:
        return TavilyWebSearch()
    return DuckDuckGoWebSearch()


def reset_web_search_provider_cache() -> None:
    """Drop cached search provider (used by tests)."""
    get_web_search_provider.cache_clear()

