# ============================================
# Guardrails: PII, URL Validation, Budget, Rate Limiting
# ============================================
# Production safety checks for the research pipeline.
# Teaches: layered guardrails, rate limiting for API quotas.

import re
import time
import ipaddress
import logging
import requests
import threading
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# PII Patterns (compiled for performance)
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

# Domains the scraper is allowed to fetch when the allowlist is switched on.
# Off by default: the researcher feeds this scraper URLs that Tavily chose, and
# four domains would reject almost all of them. Kept as an opt-in because a
# closed allowlist is the right posture once you know your source set -- and
# because this constant used to be defined, documented as a guardrail, and never
# read by a single line of code.
ALLOWED_DOMAINS = ["wikipedia.org", "arxiv.org", "github.com", "tavily.com"]

TOKEN_BUDGET = 50000  # Default hard limit


# === PII Detection & Scrubbing ===

def detect_pii(text: str) -> dict:
    """Scan text for PII. Returns dict of PII type -> matches."""
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[pii_type] = matches
    return found


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Remove PII from text. Returns (cleaned_text, list of PII types found)."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
    return text, found


# === URL Validation ===

def host_of(url: str) -> str:
    """Lowercased hostname, or "" if the URL will not parse."""
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_allowed_domain(url: str, allowlist: list[str] | None = None) -> bool:
    """Is this URL on the allowlist, matched by hostname suffix?

    Suffix on the parsed HOSTNAME, never a substring of the whole URL.
    `"wikipedia.org" in url` -- the shortcut this project used elsewhere --
    happily accepts https://evil.example/wikipedia.org/, because the string is
    right there in the path.
    """
    host = host_of(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in (allowlist or ALLOWED_DOMAINS))


def is_internal_address(url: str) -> bool:
    """True if the URL points at loopback, a private range, or link-local.

    Why a scraper needs this: the URLs reaching it come from web search results
    and from page content, i.e. from strangers. Fetching whatever it is handed
    turns this agent into an SSRF proxy -- http://169.254.169.254/latest/meta-data/
    is the cloud instance-metadata endpoint, and it answers to anything running
    inside the VPC. It is reachable, so the old `status_code < 400` check passed
    it, and the scraper then fed the credentials it returned straight into a
    prompt.

    DNS names that resolve to private space are not caught here -- that needs
    resolve-then-check-then-pin, and requests re-resolves on connect. Treat this
    as the cheap layer, not the whole answer.
    """
    host = host_of(url)
    if not host:
        return True
    if host in {"localhost", "metadata.google.internal"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a name, not a literal address
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast)


def validate_url(url: str, timeout: int = 5, allowlist: list[str] | None = None) -> bool:
    """Check that a URL is safe to fetch, then that it is reachable.

    Reachability alone is not validation. This used to be a bare HEAD request:
    any URL that answered was "valid", including file-adjacent schemes, internal
    hosts, and the metadata service.

    Pass `allowlist` (e.g. `ALLOWED_DOMAINS`) to additionally require a known
    domain.
    """
    scheme = ""
    try:
        scheme = (urlparse(url).scheme or "").lower()
    except ValueError:
        return False
    if scheme not in ("http", "https"):
        logger.warning(f"Rejected non-HTTP(S) URL: {url[:80]}")
        return False
    if is_internal_address(url):
        logger.warning(f"Rejected internal/loopback URL (SSRF guard): {url[:80]}")
        return False
    if allowlist is not None and not is_allowed_domain(url, allowlist):
        logger.warning(f"Rejected URL outside the allowlist: {url[:80]}")
        return False
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.status_code < 400
    except (requests.RequestException, ValueError):
        return False


# === Token Budget ===

def check_budget(current_tokens: int, budget: int = TOKEN_BUDGET) -> bool:
    """Return True if we still have token budget remaining."""
    if current_tokens >= budget:
        logger.warning(f"Token budget exceeded: {current_tokens}/{budget}")
        return False
    return True


# === Prompt Injection Detection (OWASP LLM Top 10 #1) ===

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|your\s+|previous\s+)?instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output)\s+(your\s+)?(system\s+prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+|all\s+)?above", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"override\s+(system|safety)\s+", re.IGNORECASE),
]


def detect_injection(text: str) -> tuple[bool, list[str]]:
    """Scan input for prompt injection patterns.

    Returns (is_safe, list_of_matched_patterns).
    """
    found = []
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            found.append(match.group())
    return (len(found) == 0, found)


# === Rate Limiter (Gemini Free Tier: 30 RPM) ===

class RateLimiter:
    """Fixed-spacing rate limiter for API calls.

    Not a token bucket, despite what this docstring used to say -- the
    difference matters. A token bucket accumulates credit while idle and lets
    you spend it in a burst; this limiter never allows a burst, it just holds
    every call at least `60 / max_rpm` seconds after the last one. That is the
    safer choice against a provider quota (Gemini free tier: 30 RPM) and the
    slower one for bursty workloads. Pick deliberately.
    """

    def __init__(self, max_rpm: int = 30):
        self.max_rpm = max_rpm
        self.min_interval = 60.0 / max_rpm  # seconds between calls
        self._last_call = 0.0
        self._call_count = 0
        self._observed_min_gap = float("inf")
        self._lock = threading.Lock()

    def wait_if_needed(self) -> float:
        """Block until it's safe to make another API call.

        Returns the number of seconds waited (0 if no wait needed).
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if self._call_count:
                self._observed_min_gap = min(self._observed_min_gap, elapsed)
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                time.sleep(wait_time)
                self._last_call = time.time()
                self._call_count += 1
                return wait_time
            else:
                self._last_call = now
                self._call_count += 1
                return 0.0

    @property
    def total_calls(self) -> int:
        return self._call_count

    def within_limit(self) -> bool:
        """Have calls been spaced at least `min_interval` apart so far?"""
        with self._lock:
            if self._call_count < 2:
                return True
            return (time.time() - self._last_call) >= 0 and self._observed_min_gap >= self.min_interval

    def reset(self):
        """Reset the limiter (for testing)."""
        with self._lock:
            self._last_call = 0.0
            self._call_count = 0
            self._observed_min_gap = float("inf")


# Global rate limiter instance
rate_limiter = RateLimiter(max_rpm=30)


# === Guardrail Summary ===

def check_all_guardrails(state: dict) -> dict:
    """Run every guardrail over the current state and return a status summary.

    Returns {budget_ok, pii_clean, pii_types, injection_safe, rate_ok,
             token_count, rate_limiter_calls, issues}.

    This docstring used to promise `pii_clean` and `rate_ok` and return neither:
    PII was never checked here at all, so a caller writing
    `if summary["pii_clean"]:` got a KeyError, and one writing
    `summary.get("pii_clean", True)` got a silent pass. Both keys are now real.
    """
    token_count = state.get("token_count", 0)
    budget_ok = check_budget(token_count)

    issues = []
    if not budget_ok:
        issues.append(f"Token budget exceeded: {token_count}/{TOKEN_BUDGET}")

    # Injection: scan the user query AND every retrieved snippet. The query is
    # the obvious channel; the snippets are the dangerous one, because the user
    # never sees the page the search tool pulled them from.
    query = state.get("query", "")
    injection_safe, injection_patterns = detect_injection(query)
    if not injection_safe:
        issues.append(f"Prompt injection in query: {injection_patterns}")
    for source in state.get("sources", []):
        snippet_safe, patterns = detect_injection(str(source.get("snippet", "")))
        if not snippet_safe:
            injection_safe = False
            issues.append(
                f"Prompt injection in retrieved source "
                f"{source.get('url', '?')[:60]}: {patterns}")

    # PII across everything the pipeline is holding, not just the final draft.
    pii_types = sorted({
        t for text in [query, state.get("current_draft", ""), state.get("synthesis", "")]
        for t in detect_pii(str(text))
    })
    if pii_types:
        issues.append(f"PII present in state: {pii_types}")

    # The limiter only ever delays; "not ok" means it is being driven past its
    # configured rate, which is worth surfacing before the provider 429s.
    rate_ok = rate_limiter.total_calls == 0 or rate_limiter.within_limit()

    return {
        "budget_ok": budget_ok,
        "pii_clean": not pii_types,
        "pii_types": pii_types,
        "injection_safe": injection_safe,
        "rate_ok": rate_ok,
        "token_count": token_count,
        "rate_limiter_calls": rate_limiter.total_calls,
        "issues": issues,
    }


if __name__ == "__main__":
    # Smoke tests
    test = "Contact john@example.com or 555-123-4567"
    print(f"PII detected: {detect_pii(test)}")

    cleaned, pii_types = scrub_pii(test)
    print(f"Scrubbed: {cleaned}")
    print(f"PII types found: {pii_types}")

    print(f"Budget check (1000/50000): {check_budget(1000)}")
    print(f"Budget check (60000/50000): {check_budget(60000)}")

    print(f"URL valid (wikipedia): {validate_url('https://en.wikipedia.org')}")

    # Rate limiter test
    limiter = RateLimiter(max_rpm=60)  # 1 per second for testing
    for i in range(3):
        wait = limiter.wait_if_needed()
        print(f"Call {i+1}: waited {wait:.2f}s")