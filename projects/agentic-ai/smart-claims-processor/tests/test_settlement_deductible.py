"""Fast mode skips the policy checker. The settlement agent used to treat the
deductible as 0 in that case, and live runs paid a $450 claim in full on a
policy with a $2,500 deductible. The deductible now comes from the policy
record, and the payout is capped in code, not only in the prompt.
"""

from types import SimpleNamespace

import src.agents.settlement_calculator as sc
from src.models.schemas import ClaimDecision, SettlementOutput


def _run(monkeypatch, llm_amount, deductible, estimated, human_decision=None):
    monkeypatch.setattr(sc, "lookup_policy", lambda n: {"policy_number": n})
    monkeypatch.setattr(sc, "get_coverage_for_claim_type",
                        lambda p, t: {"deductible": deductible, "coverage_limit": 100_000})
    monkeypatch.setattr(sc, "gather_with_tools", lambda *a, **k: ("", []))
    monkeypatch.setattr(sc, "get_llm", lambda *a, **k: None)
    monkeypatch.setattr(sc, "log_agent_action", lambda **k: None)
    reply = SettlementOutput(
        decision=ClaimDecision.APPROVED, settlement_amount_usd=llm_amount, gross_damage_usd=estimated,
        deductible_applied_usd=0, depreciation_applied_usd=0, calculation_breakdown=["model math"],
        confidence=0.9,
    )
    monkeypatch.setattr(sc, "get_structured_llm", lambda schema: SimpleNamespace(invoke=lambda msgs: reply))
    state = {
        "claim": {"claim_id": "CLM-D", "policy_number": "POL-HOME", "incident_type": "property_water",
                  "estimated_amount": estimated, "incident_description": "leak"},
        "masked_claim": {"incident_type": "property_water", "estimated_amount": estimated,
                         "incident_description": "leak"},
        "policy_output": None, "damage_output": None, "fraud_output": None,
        "human_decision": human_decision,
    }
    return sc.run_settlement_calculator(state)["settlement_output"]


def test_claim_below_its_deductible_is_denied_in_fast_mode(monkeypatch):
    out = _run(monkeypatch, llm_amount=450, deductible=2500, estimated=450)
    assert out.decision == ClaimDecision.DENIED
    assert out.settlement_amount_usd == 0
    assert "Deductible" in out.denial_reasons[0]


def test_payout_is_capped_at_damage_minus_deductible(monkeypatch):
    out = _run(monkeypatch, llm_amount=480, deductible=100, estimated=480)
    assert out.settlement_amount_usd == 380


def test_a_reviewer_approval_is_not_overturned_by_a_fraud_investigation_decision(monkeypatch):
    # Live run: India path B was approved by the reviewer, then the settlement
    # model returned fraud_investigation and the claim ended there.
    monkeypatch.setattr(sc, "lookup_policy", lambda n: {"policy_number": n})
    monkeypatch.setattr(sc, "get_coverage_for_claim_type",
                        lambda p, t: {"deductible": 1000, "coverage_limit": 100_000})
    monkeypatch.setattr(sc, "gather_with_tools", lambda *a, **k: ("", []))
    monkeypatch.setattr(sc, "get_llm", lambda *a, **k: None)
    monkeypatch.setattr(sc, "log_agent_action", lambda **k: None)
    reply = SettlementOutput(
        decision=ClaimDecision.FRAUD_INVESTIGATION, settlement_amount_usd=0, gross_damage_usd=20_000,
        deductible_applied_usd=0, depreciation_applied_usd=0, calculation_breakdown=["held"],
        confidence=0.9,
    )
    monkeypatch.setattr(sc, "get_structured_llm", lambda schema: SimpleNamespace(invoke=lambda msgs: reply))
    state = {
        "claim": {"claim_id": "CLM-H", "policy_number": "P", "incident_type": "theft",
                  "estimated_amount": 20_000, "incident_description": "stolen"},
        "masked_claim": {"incident_type": "theft", "estimated_amount": 20_000, "incident_description": "stolen"},
        "policy_output": None, "damage_output": None, "fraud_output": None,
        "human_decision": "approved",
    }
    out = sc.run_settlement_calculator(state)["settlement_output"]
    assert out.decision == ClaimDecision.APPROVED
    assert out.settlement_amount_usd > 0
    assert "fraud_investigation" in out.calculation_breakdown[-1]


def test_a_reviewer_settlement_override_survives_the_settlement_agent(monkeypatch):
    # Fraud-triggered pause, reviewer approves with an amount, pipeline then
    # runs settlement. The reviewer's amount used to be replaced.
    monkeypatch.setattr(sc, "lookup_policy", lambda n: {"policy_number": n})
    monkeypatch.setattr(sc, "get_coverage_for_claim_type",
                        lambda p, t: {"deductible": 500, "coverage_limit": 100_000})
    monkeypatch.setattr(sc, "gather_with_tools", lambda *a, **k: ("", []))
    monkeypatch.setattr(sc, "get_llm", lambda *a, **k: None)
    monkeypatch.setattr(sc, "log_agent_action", lambda **k: None)
    reply = SettlementOutput(
        decision=ClaimDecision.APPROVED, settlement_amount_usd=9_000, gross_damage_usd=10_000,
        deductible_applied_usd=500, depreciation_applied_usd=500, calculation_breakdown=["model"],
        confidence=0.9,
    )
    monkeypatch.setattr(sc, "get_structured_llm", lambda schema: SimpleNamespace(invoke=lambda msgs: reply))
    state = {
        "claim": {"claim_id": "CLM-O", "policy_number": "P", "incident_type": "auto_collision",
                  "estimated_amount": 10_000, "incident_description": "x"},
        "masked_claim": {"incident_type": "auto_collision", "estimated_amount": 10_000, "incident_description": "x"},
        "policy_output": None, "damage_output": None, "fraud_output": None,
        "human_decision": "approved", "human_settlement_override_usd": 7_250.0,
    }
    result = sc.run_settlement_calculator(state)
    assert result["final_amount_usd"] == 7_250.0
    assert "Reviewer set the settlement" in result["settlement_output"].calculation_breakdown[-1]
