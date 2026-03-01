# ============================================
# Custom Tool Definitions for Research Agents
# ============================================

import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from src.guardrails import validate_url


# === Web Search Tool ===
_search = TavilySearch(max_results=5)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information using Tavily.

    Returns list of {title, url, snippet, date} dicts.
    """
    try:
        raw = _search.invoke(query)
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        return [
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:500],
                "date": r.get("date", ""),
                "status": "success",
            }
            for r in results[:max_results]
        ]
    except Exception as e:
        return [{"status": "error", "error": str(e)}]


# === Content Extraction Tool ===
def extract_content(url: str, max_chars: int = 3000) -> dict:
    """Extract text content from a URL.

    Validates URL first, then extracts title + text + links.
    """
    if not validate_url(url):
        return {"status": "error", "error": f"URL not reachable: {url}"}
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "ResearchBot/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        title = soup.title.string if soup.title else "Untitled"
        text = soup.get_text(separator="\n", strip=True)[:max_chars]
        links = [a.get("href") for a in soup.find_all("a", href=True)[:10]]

        return {
            "status": "success",
            "title": title,
            "content": text,
            "links": links,
            "source_url": url,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "source_url": url}


# === Summarization Tool ===
_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def summarize(text: str, max_tokens: int = 500) -> dict:
    """Summarize a long text using LLM. Tracks token usage."""
    try:
        response = _llm.invoke(
            f"Summarize the following text in {max_tokens} tokens or less. "
            f"Focus on key facts and findings:\n\n{text[:4000]}"
        )
        tokens_used = response.response_metadata.get(
            "token_usage", {}
        ).get("total_tokens", 0)
        return {
            "status": "success",
            "summary": response.content,
            "tokens_used": tokens_used,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "tokens_used": 0}