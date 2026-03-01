# ============================================
# Web Scraper Tool
# ============================================

import logging
import requests
from bs4 import BeautifulSoup
from src.guardrails import validate_url

logger = logging.getLogger(__name__)


def scrape_url(url: str, max_chars: int = 3000) -> dict:
    """Extract text content from a URL.

    Validates URL first, then extracts title + text.
    Returns {title, snippet, url, tool} or empty dict on failure.
    """
    if not validate_url(url):
        logger.warning(f"URL not reachable: {url}")
        return {}
    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "ResearchBot/1.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
        text = soup.get_text(separator=" ", strip=True)[:max_chars]

        logger.info(f"Scraped {url}: {len(text)} chars")
        return {
            "title": title,
            "url": url,
            "snippet": text[:500],
            "date": "",
            "tool": "scraper",
        }
    except Exception as e:
        logger.error(f"Scraper error for {url}: {e}")
        return {}


def scrape_urls(urls: list[str], max_chars: int = 3000) -> list[dict]:
    """Scrape multiple URLs, skipping failures."""
    results = []
    for url in urls:
        result = scrape_url(url, max_chars)
        if result:
            results.append(result)
    return results
