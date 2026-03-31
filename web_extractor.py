"""Web page content extractor and search for Discord LLM Bot"""

import asyncio
import base64
import re
import logging
from typing import Dict, List, Optional, Tuple

import aiohttp
import trafilatura

logger = logging.getLogger("WebExtractor")

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

MAX_CONTENT_LENGTH = 2000
MAX_CONTENT_LENGTH_WIKI = 6000

# Minimum chars of extracted content to consider a page "successfully scraped".
# Below this threshold, we retry with JS rendering if available.
_MIN_CONTENT_THRESHOLD = 100

_VISION_EXTRACT_PROMPT = (
    "Extract ALL text content, data, and tables from this webpage screenshot. "
    "For tables, preserve the structure using plain text columns. "
    "Include all numbers, names, statistics, and details exactly as shown. "
    "Output ONLY the extracted content, no commentary."
)


class JSRenderer:
    """Headless Chromium renderer using Playwright for JS-heavy pages."""

    def __init__(self):
        self._pw = None
        self._browser = None

    async def start(self):
        """Launch the headless browser. Call once at application startup."""
        try:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=True)
            logger.info("Playwright JS renderer started")
        except Exception as e:
            logger.warning(f"Failed to start Playwright renderer: {e}")
            self._pw = None
            self._browser = None

    async def stop(self):
        """Shut down the browser. Call once at application shutdown."""
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        self._browser = None
        self._pw = None
        logger.info("Playwright JS renderer stopped")

    @property
    def available(self) -> bool:
        return self._browser is not None

    async def render(self, url: str, timeout: int = 15000) -> Optional[str]:
        """Render a page with JS and return the fully rendered HTML."""
        if not self.available:
            return None
        try:
            page = await self._browser.new_page()
            try:
                await page.goto(url, wait_until="load", timeout=timeout)
                # Give JS frameworks time to render content
                await asyncio.sleep(2)
                html = await page.content()
                return html
            finally:
                await page.close()
        except Exception as e:
            logger.warning(f"JS render failed for {url}: {e}")
            return None

    async def screenshot(self, url: str, timeout: int = 15000) -> Optional[str]:
        """Render a page and return a base64-encoded full-page PNG screenshot."""
        if not self.available:
            return None
        try:
            page = await self._browser.new_page(viewport={"width": 1280, "height": 900})
            try:
                await page.goto(url, wait_until="load", timeout=timeout)
                await asyncio.sleep(2)
                png_bytes = await page.screenshot(full_page=True)
                return base64.b64encode(png_bytes).decode("ascii")
            finally:
                await page.close()
        except Exception as e:
            logger.warning(f"Screenshot failed for {url}: {e}")
            return None


# Module-level singleton — call js_renderer.start() at app startup
js_renderer = JSRenderer()


async def _extract_with_vision(screenshot_b64: str) -> str:
    """Send a screenshot to the vision model and extract text content."""
    from config import OLLAMA_API_URL, IMAGE_RECOGNITION_MODEL

    payload = {
        "model": IMAGE_RECOGNITION_MODEL,
        "prompt": _VISION_EXTRACT_PROMPT,
        "images": [screenshot_b64],
        "stream": False,
        "options": {"num_ctx": 32768},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()
                content = data.get("response", "")
                logger.info(f"Vision extraction returned {len(content)} chars")
                return content
    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        return ""


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[^\s\")*\]]*)?)?'
    urls = re.findall(url_pattern, text)
    urls = [url.rstrip('.,;!?"\'()[]') for url in urls]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    logger.debug(f"Extracted URLs: {unique}")
    return unique


def _is_wiki_url(url: str) -> bool:
    """Check if a URL is a Wikipedia page."""
    return 'wikipedia.org/wiki/' in url


async def _extract_with_trafilatura(html: str) -> Tuple[str, str]:
    """Extract title and content from HTML using trafilatura (CPU-bound, runs in thread)."""
    content = await asyncio.to_thread(
        trafilatura.extract,
        html,
        include_comments=False,
        include_tables=True,
        output_format='txt',
    )
    metadata = await asyncio.to_thread(trafilatura.extract_metadata, html)
    title = metadata.title if metadata and metadata.title else ""
    return title, content or ""


async def fetch_webpage_content(url: str, timeout: int = 10, max_length: int = None) -> Tuple[str, str]:
    """
    Fetch and extract main content from a webpage.

    Uses aiohttp + trafilatura as the fast path. If the result is too thin
    (< _MIN_CONTENT_THRESHOLD chars) and the JS renderer is available,
    retries with headless Chromium to handle JS-rendered pages.

    Returns:
        Tuple of (title, content)
    """
    if max_length is None:
        max_length = MAX_CONTENT_LENGTH_WIKI if _is_wiki_url(url) else MAX_CONTENT_LENGTH

    try:
        # --- Fast path: aiohttp + trafilatura ---
        logger.info(f"Fetching webpage: {url}")
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url, headers=DEFAULT_HEADERS) as response:
                response.raise_for_status()
                html = await response.text()

        title, content = await _extract_with_trafilatura(html)

        # --- JS fallback: if content is thin, retry with Playwright ---
        if len(content) < _MIN_CONTENT_THRESHOLD and js_renderer.available:
            logger.info(f"Thin content ({len(content)} chars), retrying with JS renderer: {url}")

            # Try trafilatura on rendered HTML first (cheaper)
            rendered_html = await js_renderer.render(url)
            if rendered_html:
                js_title, js_content = await _extract_with_trafilatura(rendered_html)
                if len(js_content) > len(content):
                    content = js_content
                    title = js_title or title
                    logger.info(f"JS+trafilatura improved content to {len(content)} chars")

            # If still thin, screenshot + vision model (most expensive, most capable)
            if len(content) < _MIN_CONTENT_THRESHOLD:
                logger.info(f"Still thin after JS+trafilatura, trying screenshot+vision: {url}")
                screenshot_b64 = await js_renderer.screenshot(url)
                if screenshot_b64:
                    vision_content = await _extract_with_vision(screenshot_b64)
                    if len(vision_content) > len(content):
                        content = vision_content
                        logger.info(f"Vision extraction improved content to {len(content)} chars")

        if not content:
            logger.warning(f"No content extracted for {url}")
            return title, ""

        # Truncate to prevent overwhelming the LLM context
        if len(content) > max_length:
            content = content[:max_length] + "..."

        logger.info(f"Extracted content from {url} (length: {len(content)})")
        return title, content
    except Exception as e:
        logger.error(f"Error fetching webpage {url}: {e}")
        return "", f"Error fetching webpage: {str(e)}"


async def extract_webpage_context(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Extract content from URLs in text and return formatted context.

    Returns:
        Tuple of (context_string, list of dicts with 'url' and 'title' for each fetched page)
    """
    urls = extract_urls(text)

    if not urls:
        return "", []

    context_parts = []
    fetched_sources = []

    for url in urls:
        title, content = await fetch_webpage_content(url)
        if content:
            context_parts.append(f"Web Page Content (from {url}):\nTitle: {title}\nContent: {content}\n")
            fetched_sources.append({"url": url, "title": title or url})

    return "\n".join(context_parts) if context_parts else "", fetched_sources


async def web_search(query: str, max_results: int = 5, use_ddg: bool = False, enrich_top: int = 2) -> List[dict]:
    """
    Search the web and return results, optionally enriching top results
    with full page content via trafilatura.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        use_ddg: If True, use DuckDuckGo instead of Tavily.
        enrich_top: Number of top results to fetch full page content for (0 to disable).

    Returns:
        List of dicts with 'title', 'url', and 'content' keys.
    """
    if use_ddg:
        results = await _ddg_search(query, max_results)
    else:
        results = await _tavily_search(query, max_results)

    # Always enrich Wikipedia results (server-rendered, high-quality structured data)
    for i in range(len(results)):
        if _is_wiki_url(results[i]["url"]):
            logger.info(f"Enriching Wikipedia result: {results[i]['url']}")
            title, full_content = await fetch_webpage_content(results[i]["url"])
            if full_content and len(full_content) > len(results[i]["content"]):
                results[i]["content"] = full_content
                if title and not results[i]["title"]:
                    results[i]["title"] = title

    # Move Wikipedia results to the front (highest quality data)
    wiki_results = [r for r in results if _is_wiki_url(r["url"])]
    other_results = [r for r in results if not _is_wiki_url(r["url"])]
    results = wiki_results + other_results

    # Enrich top non-wiki results with short snippets
    if results and enrich_top > 0:
        enriched = 0
        for i in range(len(results)):
            if enriched >= enrich_top:
                break
            if _is_wiki_url(results[i]["url"]) or len(results[i]["content"]) >= 500:
                continue
            url = results[i]["url"]
            logger.info(f"Enriching search result: {url}")
            title, full_content = await fetch_webpage_content(url)
            if full_content and len(full_content) > len(results[i]["content"]):
                results[i]["content"] = full_content
                if title and not results[i]["title"]:
                    results[i]["title"] = title
                enriched += 1

    return results


async def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Search using Tavily API."""
    from config import TAVILY_API_KEY

    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set, web search unavailable")
        return []

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        response = await client.search(query, max_results=max_results)

        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
            })

        logger.info(f"Tavily search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error during Tavily search: {e}")
        return []


async def _ddg_search(query: str, max_results: int = 5) -> List[dict]:
    """Search using DuckDuckGo (news + text combined)."""
    try:
        from duckduckgo_search import DDGS

        def _search():
            ddgs = DDGS()
            # Use news search for better current-events results
            news = ddgs.news(query, max_results=max_results)
            results = []
            for r in news:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("body", ""),
                })
            return results

        results = await asyncio.to_thread(_search)
        logger.info(f"DDG search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error during DDG search: {e}")
        return []


def format_search_results(results: List[dict], max_per_result: int = 1500) -> str:
    """Format search results into a context string for the LLM.

    Args:
        results: List of search result dicts.
        max_per_result: Max content chars per result (Wikipedia gets 4x this limit).
    """
    if not results:
        return ""

    parts = ["Web Search Results:"]
    for i, result in enumerate(results, 1):
        content = result['content']
        limit = max_per_result * 4 if _is_wiki_url(result.get('url', '')) else max_per_result
        if len(content) > limit:
            content = content[:limit] + "..."
        parts.append(f"\n[{i}] {result['title']}\n{content}")

    return "\n".join(parts)
