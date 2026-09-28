"""Run provenance: the model, config, prompts and reads behind a decision.

The audit chain proves a FINAL_DECISION line was not edited. It never said what
produced the line, so a claim could not be reconstructed from the log: the model
comes from the environment, the prompts are Python strings that get edited, the
thresholds are a YAML file that gets edited, and memory returns different
neighbours every week.

These tests pin the four fields and, more importantly, the two properties that
make them worth having: the record survives a HITL pause, and it never carries
prompt text or claim text.
"""

import json
import threading

import pytest

from src.guardrails.manager import guard_node
from src.llm import reset_token_tracking
from src.provenance import (
    collect_node,
    config_fingerprint,
    record_prompt,
    record_retrieval,
    run_identity,
    run_versions_update,
    start_node,
)


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    from src.security import audit_log

    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path))
    audit_log._last_hash_cache.clear()
    yield tmp_path
    audit_log._last_hash_cache.clear()


def _state(**extra):
    base = {
        "claim": {"claim_id": "CLM-P-1"},
        "agent_call_count": 0,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "processing_seconds": 0.0,
        "guardrails_violations": [],
        "pipeline_trace": [],
    }
    base.update(extra)
    return base


# ── The four fields ─────────────────────────────────────────────────────────

def test_run_identity_names_the_model_provider_and_config():
    ident = run_identity()
    assert ident["model"]
    assert ident["provider"]
    assert len(ident["config_sha256"]) == 16


def test_config_fingerprint_changes_when_a_threshold_changes(tmp_path, monkeypatch):
    import src.provenance as prov

    before = config_fingerprint()
    edited = tmp_path / "base.yaml"
    edited.write_text("hitl:\n  triggers:\n    low_confidence: 0.60\n", encoding="utf-8")
    prov.config_fingerprint.cache_clear()
    monkeypatch.setattr("src.config.CONFIG_PATH", edited)
    monkeypatch.setattr("src.config._COUNTRIES_DIR", tmp_path / "none")
    after = config_fingerprint()
    prov.config_fingerprint.cache_clear()

    assert after != before, "a YAML edit that moves a gate must move the fingerprint"


def test_prompt_hash_is_stable_for_the_same_prompt_and_differs_otherwise():
    start_node()
    record_prompt("Assess the damage on claim CLM-1")
    first, _ = collect_node()

    start_node()
    record_prompt("Assess the damage on claim CLM-1")
    same, _ = collect_node()

    start_node()
    record_prompt("Assess the damage on claim CLM-2")
    other, _ = collect_node()

    assert first == same
    assert first != other


def test_retrieved_ids_keep_their_order_and_do_not_repeat():
    start_node()
    record_retrieval(["CLM-9", "CLM-3"])
    record_retrieval(["CLM-3", "CLM-7"])
    _, ids = collect_node()
    assert ids == ["CLM-9", "CLM-3", "CLM-7"]


# ── The properties that make it useful ──────────────────────────────────────

def test_no_prompt_text_or_claim_text_reaches_the_record():
    secret = "Rajesh Kumar, policy POL-77, rear-ended at a red light"
    start_node()
    record_prompt(f"You are an intake agent. Claim text: {secret}")
    record_retrieval(["CLM-42"])
    record = run_versions_update(_state(), "intake_agent")

    blob = json.dumps(record)
    assert secret not in blob
    assert "Rajesh" not in blob and "POL-77" not in blob
    assert len(record["prompts"]["intake_agent"]) == 16


def test_guard_node_folds_provenance_into_state():
    reset_token_tracking()

    def node(state):
        record_prompt("intake prompt")
        record_retrieval(["CLM-100"])
        return {"intake_output": object()}

    update = guard_node("intake_agent", node, enforce_budget=False)(_state())
    record = update["run_versions"]

    assert record["model"] and record["config_sha256"]
    assert "intake_agent" in record["prompts"]
    assert record["retrieved"]["intake_agent"] == ["CLM-100"]


def test_a_node_does_not_inherit_the_previous_node_s_prompts():
    reset_token_tracking()
    state = _state()

    state["run_versions"] = guard_node(
        "intake_agent", lambda s: record_prompt("intake prompt") or {}, enforce_budget=False
    )(state)["run_versions"]
    state["run_versions"] = guard_node(
        "damage_assessor", lambda s: record_prompt("damage prompt") or {}, enforce_budget=False
    )(state)["run_versions"]

    prompts = state["run_versions"]["prompts"]
    assert set(prompts) == {"intake_agent", "damage_assessor"}
    assert prompts["intake_agent"] != prompts["damage_assessor"]


def test_the_record_survives_a_pause_and_a_resume_in_a_new_context():
    """A resume runs in a fresh ContextVar, exactly as it does after a restart.

    Token totals survive a pause because guard_node carries them in state. The
    versions record has to survive the same way, or a claim approved on Monday
    and settled on Thursday loses everything the first half of the run recorded.
    """
    reset_token_tracking()
    before_pause = guard_node(
        "intake_agent", lambda s: record_prompt("intake prompt") or {}, enforce_budget=False
    )(_state())["run_versions"]

    resumed = {}

    def second_half():
        reset_token_tracking()   # what a resume in another process looks like
        state = _state(run_versions=before_pause)
        resumed.update(
            guard_node(
                "settlement_calculator",
                lambda s: record_prompt("settlement prompt") or {},
                enforce_budget=False,
            )(state)["run_versions"]
        )

    t = threading.Thread(target=second_half)
    t.start()
    t.join()

    assert set(resumed["prompts"]) == {"intake_agent", "settlement_calculator"}
    assert resumed["prompts"]["intake_agent"] == before_pause["prompts"]["intake_agent"]


def test_two_claims_in_two_threads_do_not_share_prompt_hashes():
    """The same bug the token accumulator had: a module-level list mixes runs."""
    out = {}

    def run(name, prompt):
        reset_token_tracking()
        out[name] = guard_node(
            name, lambda s: record_prompt(prompt) or {}, enforce_budget=False
        )(_state())["run_versions"]["prompts"]

    threads = [
        threading.Thread(target=run, args=("intake_agent", "claim one")),
        threading.Thread(target=run, args=("policy_checker", "claim two")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert list(out["intake_agent"]) == ["intake_agent"]
    assert list(out["policy_checker"]) == ["policy_checker"]


# ── The audit entry ─────────────────────────────────────────────────────────

def test_final_decision_entry_carries_the_versions_block(audit_dir):
    import src.security.audit_log as audit

    versions = {"model": "gemini-3.6-flash", "config_sha256": "abc123", "prompts": {"intake_agent": "deadbeef"}}
    audit.log_final_decision(
        claim_id="CLM-P-9",
        decision="approved",
        amount_usd=1200.0,
        total_tokens=34000,
        total_cost_usd=0.14,
        versions=versions,
    )

    entry = json.loads(next(audit_dir.glob("audit_*.ndjson")).read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["versions"] == versions


def test_a_decision_logged_without_versions_still_writes_an_entry(audit_dir):
    """Old callers, and the crash path, must not break on the new field."""
    import src.security.audit_log as audit

    audit.log_final_decision(
        claim_id="CLM-P-10", decision="denied", amount_usd=0.0, total_tokens=0, total_cost_usd=0.0
    )

    entry = json.loads(next(audit_dir.glob("audit_*.ndjson")).read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["versions"] == {}
