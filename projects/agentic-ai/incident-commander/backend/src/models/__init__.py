"""Domain models: incident state machine, event envelope, and LLM output schemas."""

from .schemas import (
    SCHEMA_VERSION,
    CriticVerdict,
    DiagnosisResult,
    Hypothesis,
    JudgeScore,
    PostmortemResult,
    RemediationProposal,
    RunJudgement,
    SupervisorDecision,
    TriageResult,
    VerificationResult,
)
from .state import (
    ActionRisk,
    ActionStatus,
    ActionType,
    AgentName,
    EventType,
    HaltReason,
    IncidentEvent,
    IncidentState,
    IncidentStatus,
    RemediationAction,
    Severity,
    TelemetryTool,
)

__all__ = [
    # state
    "ActionRisk",
    "ActionStatus",
    "ActionType",
    "AgentName",
    "EventType",
    "HaltReason",
    "IncidentEvent",
    "IncidentState",
    "IncidentStatus",
    "RemediationAction",
    "Severity",
    "TelemetryTool",
    # schemas
    "SCHEMA_VERSION",
    "CriticVerdict",
    "DiagnosisResult",
    "Hypothesis",
    "JudgeScore",
    "PostmortemResult",
    "RemediationProposal",
    "RunJudgement",
    "SupervisorDecision",
    "TriageResult",
    "VerificationResult",
]
