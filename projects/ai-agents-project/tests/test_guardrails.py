"""
Tests for the Guardrails module.
Run: pytest tests/ -v
"""
import pytest


def test_detect_pii_email():
    from src.guardrails import detect_pii
    result = detect_pii("Contact john@example.com for info")
    assert "email" in result
    assert "john@example.com" in result["email"]


def test_detect_pii_phone():
    from src.guardrails import detect_pii
    result = detect_pii("Call 555-123-4567 today")
    assert "phone" in result


def test_detect_pii_ssn():
    from src.guardrails import detect_pii
    result = detect_pii("SSN: 123-45-6789")
    assert "ssn" in result


def test_detect_pii_clean():
    from src.guardrails import detect_pii
    result = detect_pii("No personal information here")
    assert len(result) == 0


def test_scrub_pii():
    from src.guardrails import scrub_pii
    text = "Email john@example.com or call 555-123-4567"
    cleaned, found = scrub_pii(text)
    assert "john@example.com" not in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "email" in found


def test_check_budget_within():
    from src.guardrails import check_budget
    assert check_budget(1000) is True


def test_check_budget_exceeded():
    from src.guardrails import check_budget
    assert check_budget(60000) is False


def test_check_budget_custom():
    from src.guardrails import check_budget
    assert check_budget(500, budget=1000) is True
    assert check_budget(1500, budget=1000) is False


def test_rate_limiter_init():
    from src.guardrails import RateLimiter
    limiter = RateLimiter(max_rpm=60)
    assert limiter.max_rpm == 60
    assert limiter.total_calls == 0


def test_rate_limiter_tracks_calls():
    from src.guardrails import RateLimiter
    limiter = RateLimiter(max_rpm=600)  # High RPM = no waiting
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    assert limiter.total_calls == 2


def test_rate_limiter_reset():
    from src.guardrails import RateLimiter
    limiter = RateLimiter(max_rpm=600)
    limiter.wait_if_needed()
    limiter.reset()
    assert limiter.total_calls == 0


def test_check_all_guardrails():
    from src.guardrails import check_all_guardrails
    result = check_all_guardrails({"token_count": 1000})
    assert result["budget_ok"] is True
    assert isinstance(result["issues"], list)

    result2 = check_all_guardrails({"token_count": 60000})
    assert result2["budget_ok"] is False
    assert len(result2["issues"]) > 0


def test_evaluation_imports():
    from evaluation.run_eval import TEST_QUESTIONS
    assert len(TEST_QUESTIONS) >= 5
    assert all("query" in q for q in TEST_QUESTIONS)
    assert all("type" in q for q in TEST_QUESTIONS)


def test_judge_prompts():
    from evaluation.judge_prompt import ACCURACY_PROMPT, COMPLETENESS_PROMPT, CITATION_PROMPT
    assert "{query}" in ACCURACY_PROMPT
    assert "{report}" in COMPLETENESS_PROMPT
    assert "0-3" in CITATION_PROMPT
