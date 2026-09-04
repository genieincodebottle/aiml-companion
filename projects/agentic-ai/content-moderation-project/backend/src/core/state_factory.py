"""One place that builds a complete `ContentState`.

`ContentState` has around sixty keys, and the agents read many of them without
a `.get` default. Building one by hand is therefore easy to get wrong in a way
that does not raise: a missing `content_metadata` surfaces as
``'NoneType' object has no attribute 'platform'`` inside an agent, which that
agent catches, logs, and swallows. The pipeline then completes and reports
"under review", and the run looks like it worked.

This factory used to be a sixty-line literal inlined in `main.py`, so the API
was the only thing that could construct a state. Anything else -- a demo
script, a test, a notebook -- had to copy it, and a copy drifts. Both callers
now use this, which means a field added here reaches both.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import ContentMetadata, ContentState, ContentStatus, UserProfile
from ..utils.tools import generate_content_id


def build_initial_state(
    content_text: str,
    *,
    user_profile: UserProfile,
    content_type: str = "story_comment",
    platform: str = "web",
    language: str = "en",
    content_id: Optional[str] = None,
    metadata: Optional[ContentMetadata] = None,
    image_urls: Optional[List[str]] = None,
    video_urls: Optional[List[str]] = None,
    is_appeal: bool = False,
    appeal_reason: Optional[str] = None,
    original_decision: Optional[str] = None,
) -> ContentState:
    """Return a fully populated `ContentState` ready for the workflow.

    Every field the agents read is present. Nothing here is optional-by-accident:
    if an agent needs a key, it is set to a real empty value rather than left
    out, because a missing key and an empty one fail very differently.
    """
    if not content_text or not content_text.strip():
        raise ValueError("content_text is required and cannot be blank")

    content_id = content_id or generate_content_id()
    now = datetime.now().isoformat()

    if metadata is None:
        metadata = ContentMetadata(
            content_id=content_id,
            content_type=content_type,
            platform=platform,
            created_at=now,
            language=language,
        )

    state: Dict[str, Any] = {
        # Core identifiers
        "content_id": content_id,
        "submission_id": f"SUB-{content_id}",
        "submission_timestamp": now,

        # Content details
        "content_text": content_text,
        "content_type": content_type,
        "content_metadata": metadata,

        # Image/video analysis (for multimodal content)
        "image_urls": image_urls or [],
        "video_urls": video_urls or [],
        "image_descriptions": [],
        "detected_objects": [],
        "detected_text_in_media": [],

        # User information
        "user_profile": user_profile,
        "user_id": user_profile.user_id,
        "username": user_profile.username,

        # Content Analysis (populated by Content Analysis Agent)
        "content_category": None,
        "content_sentiment": None,
        "content_topics": [],
        "contains_sensitive_content": False,
        "explicit_content_detected": False,

        # Toxicity Detection (populated by Toxicity Detection Agent)
        "toxicity_score": None,
        "toxicity_level": None,
        "toxicity_categories": [],
        "hate_speech_detected": False,
        "harassment_detected": False,

        # Policy Violation (populated by Policy Violation Agent)
        "policy_violations": [],
        "violation_severity": None,
        "policy_flags": [],
        "recommended_action": None,

        # Reputation Scoring (populated by Reputation Agent)
        "user_reputation_score": None,
        "user_reputation_tier": None,
        "user_risk_score": None,
        "user_history_flags": [],
        "similar_violations_count": 0,

        # Appeal Information (for Appeal Review Agent)
        "is_appeal": is_appeal,
        "appeal_reason": appeal_reason,
        "original_decision": original_decision,
        "appeal_timestamp": now if is_appeal else None,

        # Action Enforcement (populated by Action Enforcement Agent)
        "moderation_action": None,
        "action_reason": "",
        "action_timestamp": None,
        "user_notified": False,
        "content_removed": False,
        "user_suspended": False,
        "suspension_duration_days": None,

        # Agent decisions tracking
        "agent_decisions": [],
        "current_agent": None,

        # Workflow control
        "status": ContentStatus.SUBMITTED.value,
        "requires_human_review": False,
        "human_review_reason": None,
        "overall_confidence": 0.0,

        # Manual review
        "reviewer_name": None,
        "review_notes": None,
        "review_decision": None,
        "review_timestamp": None,

        # Timestamps
        "created_at": now,
        "processed_at": None,

        # Memory/learning
        "similar_content": None,
        "historical_patterns": None,

        # ReAct Loop (Think-Act-Observe synthesis)
        "react_think_output": None,
        "react_act_decision": None,
        "react_observe_result": None,
        "react_confidence": None,
        "react_reasoning": None,

        # Human-in-the-Loop (HITL) fields
        "hitl_required": False,
        "hitl_trigger_reasons": [],
        "hitl_checkpoint": None,
        "hitl_priority": None,
        "hitl_assigned_to": None,
        "hitl_queue_position": None,
        "hitl_waiting_since": None,
        "hitl_human_decision": None,
        "hitl_human_notes": None,
        "hitl_human_confidence_override": None,
        "hitl_resolution_timestamp": None,

        # Guardrails. The wrapper in workflow.py creates these lazily, so the
        # pipeline works without them, but a factory that claims to return a
        # complete ContentState should actually return one.
        "_guardrail_iteration": 0,
        "_guardrail_checks": [],
        "guardrail_violations": [],
        "guardrail_warnings": [],
    }
    return state  # type: ignore[return-value]


def demo_user(
    username: str = "demo_user",
    *,
    user_id: str = "demo-user-001",
    total_violations: int = 0,
    previous_warnings: int = 0,
    reputation_score: float = 0.75,
    account_age_days: int = 365,
    verified: bool = False,
    follower_count: int = 120,
) -> UserProfile:
    """A plausible user profile for demos and tests.

    Reputation and violation history genuinely change the outcome -- the
    reputation agent escalates repeat offenders -- so these are parameters
    rather than constants.
    """
    return UserProfile(
        user_id=user_id,
        username=username,
        account_age_days=account_age_days,
        total_posts=42,
        total_violations=total_violations,
        previous_warnings=previous_warnings,
        previous_suspensions=0,
        reputation_score=reputation_score,
        reputation_tier="new_user",
        verified=verified,
        follower_count=follower_count,
    )
