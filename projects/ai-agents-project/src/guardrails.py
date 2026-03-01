# ============================================
# Guardrails: PII, URL Validation, Budget, Rate Limiting
# ============================================
# Production safety checks for the research pipeline.
# Teaches: layered guardrails, rate limiting for API quotas.

import re
import time
import logging
import requests
import threading

logger = logging.getLogger(__name__)

# PII Patterns (compiled for performance)
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

# Allowed domains for URL validation
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

def validate_url(url: str, timeout: int = 5) -> bool:
    """Check if a URL is reachable via HEAD request."""
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
    """Token bucket rate limiter for API calls.

    Gemini free tier allows 30 requests per minute.
    This limiter enforces spacing between calls to stay within quota.
    """

    def __init__(self, max_rpm: int = 30):
        self.max_rpm = max_rpm
        self.min_interval = 60.0 / max_rpm  # seconds between calls
        self._last_call = 0.0
        self._call_count = 0
        self._lock = threading.Lock()

    def wait_if_needed(self) -> float:
        """Block until it's safe to make another API call.

        Returns the number of seconds waited (0 if no wait needed).
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
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

    def reset(self):
        """Reset the limiter (for testing)."""
        with self._lock:
            self._last_call = 0.0
            self._call_count = 0


# Global rate limiter instance
rate_limiter = RateLimiter(max_rpm=30)


# === Guardrail Summary ===

def check_all_guardrails(state: dict) -> dict:
    """Run all guardrails and return a status summary.

    Returns {budget_ok, pii_clean, injection_safe, rate_ok, token_count, issues}.
    """
    token_count = state.get("token_count", 0)
    budget_ok = check_budget(token_count)

    issues = []
    if not budget_ok:
        issues.append(f"Token budget exceeded: {token_count}/{TOKEN_BUDGET}")

    # Check for prompt injection in query
    query = state.get("query", "")
    injection_safe, injection_patterns = detect_injection(query)
    if not injection_safe:
        issues.append(f"Prompt injection detected: {injection_patterns}")

    return {
        "budget_ok": budget_ok,
        "injection_safe": injection_safe,
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