"""The Critic judges claims against what the investigation gathered, not just the draft.

Found on a live run: shown only the draft's own short citations, the Critic rejected
true claims (a note in the entity network, a near-name screening result) as
unsupported, and every correct case it rejected went to a person. These tests pin the
Critic's input: the SIGNALS line, the source record, and nothing un-scanned.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import get_settings
from src.control_plane import ControlPlane
from src.models.schemas import CritiqueResult
from src.tenants.fincrime import stub_brain


@pytest.fixture
def critic_inputs(monkeypatch):
    seen: list[str] = []
    real = stub_brain.synthesize

    def spy(schema, system, user):
        if schema is CritiqueResult:
            seen.append(user)
        return real(schema, system, user)

    monkeypatch.setattr(stub_brain, "synthesize", spy)
    return seen


def _run(alert_id: str):
    plane = ControlPlane(settings=dataclasses.replace(get_settings(), auto_approve=True))
    return plane.start_investigation(alert_id, actor="ana")[1]


def test_the_critic_sees_the_signals_and_the_source_record(critic_inputs):
    _run("ALRT-001")
    (user,) = critic_inputs
    assert "SIGNALS sanctions=true" in user
    assert "SOURCE RECORD" in user
    assert '"entity_network"' in user and '"screening"' in user
    # Everything the drafter was given, so a cited triage assessment can be checked.
    assert '"triage"' in user and '"graph_finding"' in user
    assert "every_claim_cited=" in user  # the offline brain still keys on this line


def test_the_critic_sees_a_near_name_screening_result(critic_inputs):
    """ALRT-005's draft says the alert came from a name resembling a sanctioned one.
    That fact lives in the screening data, so the Critic must see it."""
    result = _run("ALRT-005")
    (user,) = critic_inputs
    assert "Meridian Holdings" in user
    assert result.disposition == "close"


def test_the_critic_never_sees_the_planted_instruction(critic_inputs):
    """The source record is the injection-scanned copy, so a note's order is redacted."""
    _run("ALRT-004")
    (user,) = critic_inputs
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" not in user.upper()
    assert "[redacted-instruction]" in user
