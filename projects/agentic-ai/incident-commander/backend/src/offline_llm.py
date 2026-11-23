"""The offline agent brain: deterministic, free, and clearly labelled.

Used in two places:

* **Offline mode** (`IC_LLM_MODE=offline`, or automatically when no Gemini key is
  set). The console and API run end to end with no key, so a first-time reader can
  see the loop, the approval gate and the postmortem before spending any quota.
  `/api/health` reports `llm_mode: "offline"` and the console shows a banner, so
  nobody mistakes these runs for model reasoning (ADR-0009).
* **Tests and `run_eval --stub`**, through `tests/_fake_llm.py`.

It satisfies the same `structured()` contract as `LLMClient`, and it plays a
competent SRE by reading the telemetry text in the prompt, the same input the real
model receives. It never looks up the answer by scenario id. If a prompt stops
carrying its evidence, the offline runs (and the tests) fail, which is the point.

What it is NOT: reasoning. It is a hand-written runbook. Use it to learn the loop,
then add a key to see what a model does with the same evidence.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .models.schemas import (
    CriticVerdict,
    DiagnosisResult,
    Hypothesis,
    PostmortemResult,
    RemediationProposal,
    SupervisorDecision,
    TriageResult,
    VerificationResult,
)
from .models.state import ActionRisk, ActionType, Severity, TelemetryTool

_HIGH = {ActionType.ROLLBACK_DEPLOY, ActionType.FAILOVER}
_MEDIUM = {ActionType.RESTART_SERVICE, ActionType.SCALE_OUT}


def _number(text: str, key: str) -> float | None:
    """Pull a numeric metric out of the JSON telemetry block."""
    m = re.search(rf'"{key}":\s*([0-9.]+)', text)
    return float(m.group(1)) if m else None


def _error_rate(text: str) -> float | None:
    return _number(text, "error_rate")


def _infer(text: str) -> tuple[str, ActionType, float, list[TelemetryTool]]:
    """Map telemetry to (cause, action, confidence, tools_needed) like a runbook.

    Ordered most specific first. Also returns the tools an unsure agent would ask for.
    """
    # --- external / unfixable: recognizing this is a skill, not a failure ---
    if "sslhandshakeexception" in text or "certificate expired" in text or "pkix path" in text:
        return (
            "An upstream partner's CA certificate has expired. The failure is in a third party's "
            "trust chain and cannot be remediated by any action available to us.",
            ActionType.NO_OP,
            0.92,
            [],
        )

    # --- compound fault, phase 2 -------------------------------------------
    if "restart required" in text or "not draining" in text or "leaked connections still held" in text:
        return (
            "Stale leaked connections are still held after the rollback; a restart is needed to drain the pool",
            ActionType.RESTART_SERVICE,
            0.85,
            [],
        )

    # --- config drift: invisible until the config diff is fetched ----------
    if "db.pool.maxsize" in text or ('"was": 50' in text and '"now": 10' in text):
        return (
            "A configuration change cut db.pool.maxSize from 50 to 10, starving the service of connections",
            ActionType.SCALE_OUT,
            0.88,
            [],
        )
    if "queue wait time" in text and "no application errors" in text:
        # The honest answer here is "I cannot tell yet": metrics moved, nothing was
        # deployed, and there are no errors. This is what needs_more_data is for.
        return (
            "Latency rose with no deploy and no errors; the cause is not identifiable from the default telemetry",
            ActionType.NO_OP,
            0.35,
            [TelemetryTool.CONFIG_DIFF, TelemetryTool.POOL_STATS],
        )

    # --- classic single-step faults ----------------------------------------
    if "outofmemoryerror" in text or "oomkilled" in text:
        return ("Memory leak exhausting heap; pods are being OOM-killed", ActionType.RESTART_SERVICE, 0.86, [])
    if "connection is not available" in text or "hikaripool" in text:
        return ("Database connection pool exhausted under load", ActionType.SCALE_OUT, 0.84, [])
    if "upstreamtimeout" in text or "thread pool exhausted" in text:
        return (
            "A slow downstream dependency is causing cascading timeouts and thread-pool starvation",
            ActionType.FAILOVER,
            0.82,
            [],
        )
    if "cache hit ratio" in text or "thundering herd" in text:
        return ("Cache stampede hammering the origin database", ActionType.CLEAR_CACHE, 0.80, [])
    if "release" in text or "exception" in text:
        return ("A regression introduced by a recent deploy", ActionType.ROLLBACK_DEPLOY, 0.83, [])

    # --- healthy ------------------------------------------------------------
    err = _error_rate(text)
    if err is not None and err < 0.02:
        return (
            "No fault is present. Every real signal is at baseline; the alert appears to be spurious.",
            ActionType.NO_OP,
            0.90,
            [],
        )
    return ("Unclear from the current telemetry", ActionType.NO_OP, 0.40, [TelemetryTool.SLO_BURN])


def _service(text: str) -> str:
    """The service under investigation, from the prompt's trusted context line."""
    m = re.search(r"- service: ([a-z0-9][a-z0-9-]*)", text)
    return m.group(1) if m else "affected-service"


def _looks_healthy(text: str) -> bool:
    err = _error_rate(text)
    latency = _number(text, "p99_latency_ms")
    mem = _number(text, "mem")
    return err is not None and err < 0.02 and (latency is None or latency < 800) and (mem is None or mem < 0.9)


def _no_effect_count(text: str) -> int:
    return text.count("no effect") + text.count("changed nothing")


class OfflineLLM:
    """Stands in for `LLMClient`; returns validated schema objects deterministically."""

    def __init__(self, step_delay_s: float = 0.0) -> None:
        self.calls: list[tuple[str, str]] = []  # (agent, schema name), for assertions
        #: Pause per call so a human can watch the console animate. 0 in tests.
        self.step_delay_s = step_delay_s

    def model_name_for(self, agent: str) -> str:  # pragma: no cover - parity with the real client
        return "offline"

    def structured(
        self,
        schema,
        system: str,
        user: str,
        incident_id: str = "global",
        max_retries: int | None = None,
        *,
        agent: str = "unknown",
        budget_remaining_usd: float | None = None,
        **_: Any,
    ):
        self.calls.append((agent, schema.__name__))
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)
        text = user.lower()

        if schema is SupervisorDecision:
            return self._supervise(text)
        if schema is TriageResult:
            return self._triage(text)
        if schema is DiagnosisResult:
            return self._diagnose(text)
        if schema is RemediationProposal:
            return self._propose(text)
        if schema is CriticVerdict:
            return self._critique(text)
        if schema is VerificationResult:
            return self._verify(text)
        if schema is PostmortemResult:
            return self._postmortem(text)

        raise ValueError(f"OfflineLLM has no canned output for {schema}")

    # --- per-agent behaviour ------------------------------------------------

    def _supervise(self, text: str) -> SupervisorDecision:
        # Mirrors what a sensible router would do from the state block. The loop
        # clamps this through `sanitize_route` regardless, so a wrong guess here is
        # repaired rather than fatal - which is itself worth exercising.
        if "triage done: false" in text:
            return SupervisorDecision(next_agent="triage", reason="incident not yet classified", confidence=0.9)
        if "diagnoses so far: 0" in text:
            return SupervisorDecision(next_agent="diagnose", reason="no diagnosis yet", confidence=0.9)
        if "latest needs_more_data: true" in text and "tools requested by the diagnostician: []" not in text:
            return SupervisorDecision(
                next_agent="gather_evidence", reason="diagnostician asked for specific tools", confidence=0.8
            )
        return SupervisorDecision(next_agent="propose", reason="the diagnosis is actionable", confidence=0.8)

    def _triage(self, text: str) -> TriageResult:
        if _looks_healthy(text):
            return TriageResult(
                severity=Severity.SEV4,
                summary="All real signals are nominal; the alert appears to be spurious.",
                blast_radius="No customer impact observed.",
                affected_service=_service(text),
                customer_facing=False,
            )
        err = _error_rate(text) or 0.0
        sev = Severity.SEV1 if err >= 0.30 else Severity.SEV2 if err >= 0.10 else Severity.SEV3
        return TriageResult(
            severity=sev,
            summary="Elevated errors and latency affecting customer requests.",
            blast_radius="Requests routed through the affected service.",
            affected_service=_service(text),
            customer_facing=True,
        )

    def _diagnose(self, text: str) -> DiagnosisResult:
        cause, _action, conf, tools = _infer(text)
        needs_more = conf < 0.6
        # If evidence was already gathered and confidence is still low, stop asking:
        # an agent that requests the same tools forever never converges.
        if needs_more and "additional evidence you previously requested" in text:
            tools = []
        return DiagnosisResult(
            hypotheses=[
                Hypothesis(
                    cause=cause,
                    confidence=conf,
                    evidence_needed="post-remediation metrics",
                    tool_to_check=tools[0] if tools else None,
                )
            ],
            leading_cause=cause,
            confidence=conf,
            evidence=[f"error_rate={_error_rate(text)}", f"p99_latency_ms={_number(text, 'p99_latency_ms')}"],
            needs_more_data=needs_more,
            requested_tools=tools if needs_more else [],
            used_recalled_episode="similar past incidents" in text,
        )

    def _propose(self, text: str) -> RemediationProposal:
        cause, action, _conf, _tools = _infer(text)
        # Never act on a healthy service.
        if _looks_healthy(text):
            action = ActionType.NO_OP
            cause = "The service is healthy; no remediation is warranted."
        risk = ActionRisk.HIGH if action in _HIGH else ActionRisk.MEDIUM if action in _MEDIUM else ActionRisk.LOW
        return RemediationProposal(
            action_type=action,
            target=_service(text),
            risk=risk,
            rationale=cause,
            expected_signal="error_rate drops below 0.01 and p99_latency_ms below 500",
            fallback_if_ineffective=ActionType.RESTART_SERVICE if action != ActionType.RESTART_SERVICE else None,
            reversible=action != ActionType.ROLLBACK_DEPLOY,
        )

    def _critique(self, text: str) -> CriticVerdict:
        # Veto only the genuinely indefensible case: acting on a healthy service.
        acting_on_healthy = _looks_healthy(text) and "- action: no_op" not in text
        return CriticVerdict(
            endorse=not acting_on_healthy,
            concerns=(
                ["The telemetry shows a healthy service; this action would cause disruption for no benefit."]
                if acting_on_healthy
                else []
            ),
            strongest_objection=(
                "Every metric is at baseline, so this action treats a fault that is not present."
                if acting_on_healthy
                else "The diagnosis assumes the current telemetry snapshot is representative rather than transient."
            ),
            safer_alternative=ActionType.NO_OP if acting_on_healthy else None,
            confidence=0.85,
        )

    def _verify(self, text: str) -> VerificationResult:
        # Only the telemetry counts - never the action's self-report.
        post = text.split("post-remediation telemetry", 1)[-1]
        err = _error_rate(post)
        recovered = err is not None and err < 0.02
        improved = "partial progress" in text
        # Out of productive moves: repeated no-effect actions mean escalate, not loop.
        stuck = _no_effect_count(text) >= 2
        if recovered:
            next_step = "resolve"
        elif stuck:
            next_step = "escalate"
        else:
            next_step = "loop"
        return VerificationResult(
            recovered=recovered,
            confidence=0.9,
            observations=[f"error_rate={err}"],
            expected_signal_met=recovered,
            signal_assessment=(
                f"Predicted error_rate below 0.01; observed {err}."
                if err is not None
                else "No error_rate present in the post-remediation telemetry."
            ),
            partial_progress=improved and not recovered,
            next_step=next_step,
        )

    def _postmortem(self, text: str) -> PostmortemResult:
        # Derive the lesson from what actually worked in THIS incident. A fixed
        # lesson ("roll the release back") would be written into runbook memory
        # after every offline run and then recalled as if it were experience.
        fixed = re.findall(r"(\w+) -> resolved the fault", text)
        if fixed:
            lesson = f"For this symptom pattern, {fixed[-1]} cleared the fault; confirm the cause in telemetry first."
            summary = f"The loop resolved the incident with {fixed[-1]}."
        else:
            lesson = "No available action cleared this fault; escalate early instead of repeating actions."
            summary = "The loop stopped without a fix and handed the incident to a human."
        return PostmortemResult(
            summary=summary,
            root_cause="See the diagnostics in the incident timeline.",
            contributing_factors=["Offline mode: written by the rule-based agent, not a model."],
            recommendations=["Re-run with a Gemini key to compare the model's postmortem."],
            runbook_lesson=lesson,
            detection_gap="No alert fired on the leading indicator.",
        )
