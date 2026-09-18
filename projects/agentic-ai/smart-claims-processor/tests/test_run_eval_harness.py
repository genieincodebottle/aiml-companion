"""evaluation/run_eval.py reads what process_claim actually returns.

process_claim returns {"paused": bool, "state": {...}}. The harness used to
treat that wrapper as the state, so every decision read "unknown" and every
eval score was None. It also could not read the multi-claim sample files.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

import run_eval  # noqa: E402

from src.models.schemas import ClaimDecision  # noqa: E402

SAMPLES = Path(__file__).parent.parent / "data" / "sample_claims"


def test_multi_claim_files_expand_into_individual_claims():
    claims = run_eval._load_claims(SAMPLES / "test_all_paths_us.json")
    assert len(claims) >= 5
    for label, claim in claims:
        assert label.startswith("test_all_paths_us.json:")
        assert claim["claim_id"].startswith("EVAL-")
        assert not any(k.startswith("_") for k in claim)


def test_country_follows_the_file_name():
    assert run_eval._country_for(SAMPLES / "test_all_paths_india.json") == "india"
    assert run_eval._country_for(SAMPLES / "auto_accident.json") == "us"


def test_completed_and_paused_claims_are_reported_from_the_state(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_text(json.dumps({
        "claim_id": "C-1", "policy_number": "P", "incident_type": "auto_collision",
        "incident_date": "2026-01-01", "incident_description": "x", "estimated_amount": 100,
    }), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({
        "claim_id": "C-2", "policy_number": "P", "incident_type": "auto_theft",
        "incident_date": "2026-01-01", "incident_description": "y", "estimated_amount": 50000,
    }), encoding="utf-8")

    def fake_process(claim):
        if claim["claim_id"] == "C-1":
            return {"paused": False, "state": {
                "final_decision": ClaimDecision.APPROVED, "final_amount_usd": 80.0,
                "evaluation_output": SimpleNamespace(overall_score=0.91), "evaluation_passed": True,
                "pipeline_trace": [{"agent": "intake_agent"}],
            }}
        return {"paused": True, "state": {"pipeline_trace": [{"agent": "hitl_checkpoint"}]}, "interrupt": {}}

    monkeypatch.setattr(run_eval, "process_claim", fake_process)
    out = run_eval.run_batch_evaluation(str(tmp_path))
    by_id = {r["claim_id"]: r for r in out["results"]}
    assert by_id["C-1"]["decision"] == "approved"
    assert by_id["C-1"]["eval_score"] == 0.91
    assert by_id["C-2"]["decision"] == "paused_for_review"
    assert out["summary"]["paused_for_review"] == 1
