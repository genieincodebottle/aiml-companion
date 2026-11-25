"""
Guardrails manager - enterprise-grade safety layer for the entire pipeline.

Provides:
1. Budget enforcement (token + cost ceilings)
2. Loop detection (prevents infinite agent cycles)
3. Timeout enforcement (per-agent and pipeline-wide)
4. Hallucination detection (source grounding checks)
5. PII detection and masking
6. Confidence thresholding

Every agent call passes through guardrails BEFORE and AFTER execution.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import get_guardrails_config, get_budget_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII Detection Patterns
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone_us": re.compile(r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "phone_intl": re.compile(r'\b\+\d{1,3}[-.\s]?\d{6,14}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
}

# URLs are left alone when masking: an SEC accession number such as
# /0001564590-21-004599 contains a ten-digit run that the phone pattern would
# happily redact, which would break the citation it belongs to.
_URL_SPLIT = re.compile(r'(https?://\S+)')

# State keys that hold AgentFinding lists (see src/models/state.py).
FINDING_KEYS = ("financial_findings", "news_findings", "competitive_findings", "risk_findings")


@dataclass
class GuardrailState:
    """Mutable state tracking for guardrail enforcement."""
    total_agent_calls: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    pipeline_start_time: float = 0.0
    agent_call_history: list[dict] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    tokens_by_agent: dict[str, int] = field(default_factory=dict)
    seen_entries: set = field(default_factory=set)  # (agent, state fingerprint) pairs


class GuardrailManager:
    """Central guardrail enforcement for the pipeline.

    Usage:
        gm = GuardrailManager()
        gm.start_pipeline()

        # Before each agent call
        check = gm.pre_check(agent_name="financial_analyst")
        if not check["passed"]:
            return fallback_output(check["reason"])

        # Execute agent...

        # After each agent call
        gm.post_check(agent_name, output_text, tokens_used, cost)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or get_guardrails_config()
        self.budget = get_budget_config()
        self.state = GuardrailState()
        self._enabled = True

    def start_pipeline(self) -> None:
        """Reset state for a new pipeline run."""
        self.state = GuardrailState(pipeline_start_time=time.time())

    def disable(self) -> None:
        """Disable guardrails (for testing)."""
        self._enabled = False

    # -------------------------------------------------------------------
    # Pre-Execution Checks
    # -------------------------------------------------------------------

    def pre_check(self, agent_name: str, state_fingerprint: str | None = None) -> dict:
        """Run all pre-execution guardrails. Returns {"passed": bool, "reason": str}.

        `state_fingerprint` (a hash of the state the agent is about to receive)
        enables repeat-entry detection: the same agent entered twice with an
        identical state is a cycle, because it will compute the same thing again.
        """
        if not self._enabled:
            return {"passed": True, "reason": ""}

        checks = [
            self._check_agent_call_limit(agent_name),
            self._check_token_budget(),
            self._check_cost_budget(),
            self._check_timeout(),
            self._check_loop_detection(agent_name),
            self._check_repeat_entry(agent_name, state_fingerprint),
            self._check_agent_token_budget(agent_name),
        ]

        for check in checks:
            if not check["passed"]:
                self.state.violations.append({
                    "agent": agent_name,
                    "type": "pre_check",
                    "reason": check["reason"],
                    "time": time.time(),
                })
                logger.warning("[Guardrail] Pre-check FAILED for %s: %s", agent_name, check["reason"])
                return check

        self.state.total_agent_calls += 1
        if state_fingerprint is not None:
            self.state.seen_entries.add((agent_name, state_fingerprint))
        self.state.agent_call_history.append({
            "agent": agent_name,
            "time": time.time(),
            "type": "pre_check_passed",
        })
        return {"passed": True, "reason": ""}

    def _check_agent_call_limit(self, agent_name: str) -> dict:
        max_calls = self.config.get("max_agent_calls", 30)
        if self.state.total_agent_calls >= max_calls:
            return {"passed": False, "reason": f"Agent call limit reached ({max_calls}). Pipeline is taking too many steps."}
        return {"passed": True, "reason": ""}

    def _check_token_budget(self) -> dict:
        max_tokens = self.budget.get("max_total_tokens", 100000)
        warning_pct = self.budget.get("warning_threshold_pct", 80)
        used = self.state.total_tokens_used

        if used >= max_tokens:
            return {"passed": False, "reason": f"Token budget exhausted ({used}/{max_tokens})"}

        if used >= max_tokens * warning_pct / 100:
            logger.warning("[Guardrail] Token budget at %d%% (%d/%d)", int(used / max_tokens * 100), used, max_tokens)

        return {"passed": True, "reason": ""}

    def _check_cost_budget(self) -> dict:
        max_cost = self.budget.get("max_cost_usd", 0.50)
        if self.state.total_cost_usd >= max_cost:
            return {"passed": False, "reason": f"Cost budget exhausted (${self.state.total_cost_usd:.4f} / ${max_cost:.2f})"}
        return {"passed": True, "reason": ""}

    def _check_timeout(self) -> dict:
        max_time = self.config.get("max_execution_time_seconds", 300)
        elapsed = time.time() - self.state.pipeline_start_time
        if elapsed >= max_time:
            return {"passed": False, "reason": f"Pipeline timeout ({elapsed:.0f}s / {max_time}s)"}
        return {"passed": True, "reason": ""}

    def _check_loop_detection(self, agent_name: str) -> dict:
        """Detect if the same agent is being called too many times."""
        max_iterations = self.config.get("max_reasoning_iterations", 10)
        recent = [h for h in self.state.agent_call_history if h["agent"] == agent_name]
        if len(recent) >= max_iterations:
            return {"passed": False, "reason": f"Agent '{agent_name}' hit loop limit ({max_iterations} calls)"}
        return {"passed": True, "reason": ""}

    def _check_repeat_entry(self, agent_name: str, fingerprint: str | None) -> dict:
        """Detect the same agent being entered again with an identical state."""
        if fingerprint is not None and (agent_name, fingerprint) in self.state.seen_entries:
            return {"passed": False,
                    "reason": f"Agent '{agent_name}' re-entered with an identical state (loop detected)"}
        return {"passed": True, "reason": ""}

    def agent_token_limit(self) -> int | None:
        """Per-agent token ceiling (`budget.max_tokens_per_agent`), or None if unset/disabled."""
        if not self._enabled:
            return None
        return self.budget.get("max_tokens_per_agent") or None

    def _check_agent_token_budget(self, agent_name: str) -> dict:
        limit = self.agent_token_limit()
        used = self.state.tokens_by_agent.get(agent_name, 0)
        if limit and used >= limit:
            return {"passed": False, "reason": f"Agent '{agent_name}' token ceiling reached ({used}/{limit})"}
        return {"passed": True, "reason": ""}

    # -------------------------------------------------------------------
    # Post-Execution Checks
    # -------------------------------------------------------------------

    def post_check(
        self,
        agent_name: str,
        output_text: str,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> dict:
        """Run post-execution guardrails. Returns {"passed": bool, "warnings": list}."""
        if not self._enabled:
            return {"passed": True, "warnings": []}

        # Update tracking
        self.state.total_tokens_used += tokens_used
        self.state.total_cost_usd += cost_usd
        self.state.tokens_by_agent[agent_name] = self.state.tokens_by_agent.get(agent_name, 0) + tokens_used

        warnings = []

        # Per-agent ceiling. The scope stops an agent before its NEXT call, so a
        # single call can still overshoot; say so instead of staying silent.
        limit = self.agent_token_limit()
        if limit and tokens_used >= limit:
            warnings.append(f"{agent_name} used {tokens_used} tokens, at or over the per-agent ceiling of {limit}")
            self.state.violations.append({
                "agent": agent_name, "type": "agent_token_ceiling",
                "reason": f"{tokens_used}/{limit} tokens", "time": time.time(),
            })

        # PII check
        if self.config.get("pii_detection_enabled", True):
            pii_found = self.detect_pii(output_text)
            if pii_found:
                warnings.append(f"PII detected in {agent_name} output: {', '.join(pii_found.keys())}")
                logger.warning("[Guardrail] PII detected in %s: %s", agent_name, list(pii_found.keys()))

        # Source grounding check
        if self.config.get("source_grounding_required", True):
            if not self._has_sources(output_text):
                warnings.append(f"No sources cited in {agent_name} output")

        # Confidence threshold: applied to structured findings by
        # apply_confidence_threshold(), which needs the result dict, not its text.

        self.state.agent_call_history.append({
            "agent": agent_name,
            "time": time.time(),
            "type": "post_check",
            "tokens": tokens_used,
            "cost": cost_usd,
            "warnings": warnings,
        })

        return {"passed": True, "warnings": warnings}

    # -------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------

    @staticmethod
    def detect_pii(text: str) -> dict[str, list[str]]:
        """Detect PII patterns in text. Returns {pattern_name: [matches]}."""
        found = {}
        for name, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found[name] = matches
        return found

    @staticmethod
    def mask_pii(text: str) -> str:
        """Replace PII with masked versions. URLs are left untouched."""
        parts = _URL_SPLIT.split(text)
        for i in range(0, len(parts), 2):  # even indexes are the text between URLs
            masked = parts[i]
            for name, pattern in PII_PATTERNS.items():
                if name == "email":
                    masked = pattern.sub("[EMAIL_REDACTED]", masked)
                elif name.startswith("phone"):
                    masked = pattern.sub("[PHONE_REDACTED]", masked)
                elif name == "ssn":
                    masked = pattern.sub("[SSN_REDACTED]", masked)
                elif name == "credit_card":
                    masked = pattern.sub("[CC_REDACTED]", masked)
                elif name == "ip_address":
                    masked = pattern.sub("[IP_REDACTED]", masked)
            parts[i] = masked
        return "".join(parts)

    def mask_result(self, result: dict) -> tuple[dict, list[str]]:
        """Mask PII in every string of an agent's output before it enters pipeline state.

        Returns (masked_result, pii_types_found). A no-op when guardrails are
        disabled or `pii_detection_enabled` is false. Detection alone only
        warned; the PII still reached the report, which is what this prevents.
        """
        if not self._enabled or not self.config.get("pii_detection_enabled", True):
            return result, []

        found: set[str] = set()

        def walk(value: Any) -> Any:
            if isinstance(value, str):
                found.update(self.detect_pii(_URL_SPLIT.sub(" ", value)))
                return self.mask_pii(value)
            if isinstance(value, dict):
                return {k: walk(v) for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v) for v in value]
            return value

        masked = walk(result)
        if found:
            logger.warning("[Guardrail] Masked PII (%s) in agent output", ", ".join(sorted(found)))
        return masked, sorted(found)

    def apply_confidence_threshold(self, agent_name: str, result: dict) -> list[str]:
        """Flag findings below `confidence_threshold` for human review.

        Sets `low_confidence: True` on each such finding (the report prints it)
        and returns warnings for the pipeline. Findings are flagged, never
        dropped: in due diligence a weak finding is information, and removing it
        silently would hide the thing the reader most needs to check.
        """
        if not self._enabled:
            return []
        threshold = self.config.get("confidence_threshold", 0.6)
        low = []
        for key in FINDING_KEYS:
            for finding in result.get(key) or []:
                confidence = finding.get("confidence")
                if isinstance(confidence, (int, float)) and confidence < threshold:
                    finding["low_confidence"] = True
                    low.append(finding)
        if not low:
            return []
        titles = "; ".join(str(f.get("title", "Untitled")) for f in low[:3])
        more = f" (+{len(low) - 3} more)" if len(low) > 3 else ""
        logger.warning("[Guardrail] %d low-confidence finding(s) from %s", len(low), agent_name)
        return [
            f"{len(low)} finding(s) from {agent_name} below the {threshold:.2f} confidence "
            f"threshold, flagged for review: {titles}{more}"
        ]

    @staticmethod
    def _has_sources(text: str) -> bool:
        """Check if the output contains source references (URLs or citations)."""
        url_pattern = re.compile(r'https?://\S+')
        citation_pattern = re.compile(r'\[(?:source|ref|citation|\d+)\]', re.IGNORECASE)
        return bool(url_pattern.search(text) or citation_pattern.search(text))

    def get_budget_status(self) -> dict:
        """Get current budget usage status."""
        max_tokens = self.budget.get("max_total_tokens", 100000)
        max_cost = self.budget.get("max_cost_usd", 0.50)
        return {
            "tokens_used": self.state.total_tokens_used,
            "tokens_limit": max_tokens,
            "tokens_pct": round(self.state.total_tokens_used / max_tokens * 100, 1) if max_tokens else 0,
            "cost_usd": round(self.state.total_cost_usd, 6),
            "cost_limit": max_cost,
            "cost_pct": round(self.state.total_cost_usd / max_cost * 100, 1) if max_cost else 0,
            "agent_calls": self.state.total_agent_calls,
            "violations": len(self.state.violations),
            "elapsed_seconds": round(time.time() - self.state.pipeline_start_time, 1),
        }

    def get_violations(self) -> list[dict]:
        """Get all recorded violations."""
        return self.state.violations.copy()
