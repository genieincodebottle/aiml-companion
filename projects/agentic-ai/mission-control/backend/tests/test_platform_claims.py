"""The claims the README makes about the platform, each checked on the real path.

An earlier audit found three claims that were true in the prose and false in the code:
cases could not be resumed after a restart, the promotion gate moved nothing, and the
"OTel" spans were never exported or linked. Live mode also sent every call to the
premium model whatever tier the agent asked for, while the audit log named the cheap
one. Each test here would have failed on that code.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from evaluation.run_eval import evaluate, gate, promote
from src.config import Settings, get_settings
from src.control_plane import ControlPlane
from src.gateway.core import Gateway
from src.gateway.providers import ProviderError, StubProvider
from src.models.schemas import ModelTier, TriageResult
from src.observability import trace_id_for
from src.registry import PromptRegistry


def _plane(**overrides) -> ControlPlane:
    return ControlPlane(settings=dataclasses.replace(get_settings(), **overrides))


# === Tracing: one linked trace per case ======================================


def _spans(plane, wid):
    path = plane.settings.data_dir / "traces" / f"{wid}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_span_of_a_case_shares_one_trace_and_links_to_its_parent():
    plane = _plane(auto_approve=True)
    wid, _ = plane.start_investigation("ALRT-001", actor="ana")
    spans = _spans(plane, wid)
    assert {s["trace_id"] for s in spans} == {trace_id_for(wid)}
    ids = {s["span_id"] for s in spans}
    model_spans = [s for s in spans if s["name"].startswith("model:")]
    assert model_spans and all(s["parent_span_id"] in ids for s in model_spans), "model spans are not under a step"
    roots = [s for s in spans if s["parent_span_id"] is None]
    assert roots and all(s["name"].startswith("step:") for s in roots)


def test_a_resumed_case_stays_in_the_same_trace():
    plane = _plane()
    wid, _ = plane.start_investigation("ALRT-001", actor="ana")
    ControlPlane(settings=plane.settings).decide(wid, "approve", actor="rex")
    assert {s["trace_id"] for s in _spans(plane, wid)} == {trace_id_for(wid)}
    assert any(s["name"] == "step:commit_disposition" for s in _spans(plane, wid))


def test_the_console_exporter_emits_real_opentelemetry_spans(capsys):
    pytest.importorskip("opentelemetry.sdk", reason="needs the otel extra: uv sync --extra otel")
    from src.observability import reset_otel

    reset_otel()
    plane = _plane(auto_approve=True, otel_exporter="console")
    assert plane.tracing == "console"
    plane.start_investigation("ALRT-003", actor="ana")
    out = capsys.readouterr().out
    assert '"name": "step:triage"' in out
    assert trace_id_for_in_hex(out)
    reset_otel()  # the exporter holds pytest's captured stdout; stop before it closes


def trace_id_for_in_hex(out: str) -> bool:
    return '"trace_id": "0x' in out


# === Routing: each tier reaches the model the audit log names ================


class _Recorder(StubProvider):
    def __init__(self, label, fail=False):
        self.label, self.fail, self.calls = label, fail, 0

    def complete(self, system, user, schema):
        self.calls += 1
        if self.fail:
            raise ProviderError(f"{self.label} down")
        obj, usage = super().complete(system, user, schema)
        usage.model = self.label
        return obj, usage


def _gateway(premium_fail=False):
    gw = Gateway(dataclasses.replace(get_settings(), cache_enabled=False))
    gw._providers = {ModelTier.PREMIUM: _Recorder("premium", premium_fail), ModelTier.CHEAP: _Recorder("cheap")}
    return gw


def test_a_cheap_tier_call_goes_to_the_cheap_model():
    gw = _gateway()
    _, usage = gw.complete(workflow_id="w", system="s", user="Alert: x", schema=TriageResult, tier=ModelTier.CHEAP)
    assert usage.model == "cheap"
    assert gw._providers[ModelTier.PREMIUM].calls == 0


def test_a_failing_tier_falls_back_to_the_other_and_says_so():
    gw = _gateway(premium_fail=True)
    events = []
    _, usage = gw.complete(
        workflow_id="w",
        system="s",
        user="Alert: x",
        schema=TriageResult,
        tier=ModelTier.PREMIUM,
        on_event=lambda name, payload: events.append(name),
    )
    assert usage.model == "cheap(fallback)"
    assert "provider_failover" in events
    audit = (gw.settings.data_dir / "audit" / "gateway.jsonl").read_text().splitlines()
    assert json.loads(audit[-1])["model"] == "cheap(fallback)", "the audit log named a model that did not answer"


# === The promotion gate moves a real label, and only on a live pass ==========


def _registry(tmp_path):
    reg = PromptRegistry(tmp_path)
    reg.register("fincrime.narrative", "a better narrative prompt", labels=["candidate"])
    return reg


def _report(mode="live", accuracy=1.0, injections_ok=True, judge=0.9):
    return {"mode": mode, "accuracy": accuracy, "injections_ok": injections_ok, "mean_judge": judge}


def test_a_passing_live_run_promotes_and_the_label_survives_a_restart(tmp_path):
    reg = _registry(tmp_path)
    ok, _ = promote(reg, ["fincrime.narrative"], _report(), 0.8)
    assert ok
    assert PromptRegistry(tmp_path).get("fincrime.narrative").template == "a better narrative prompt"


def test_an_offline_run_never_promotes(tmp_path):
    reg = _registry(tmp_path)
    ok, why = promote(reg, ["fincrime.narrative"], _report(mode="offline"), 0.8)
    assert not ok and "offline" in why
    assert PromptRegistry(tmp_path).get("fincrime.narrative").version == 1


def test_each_gate_term_can_fail_on_its_own():
    assert not gate(_report(accuracy=0.8), 0.8)[0]
    assert not gate(_report(injections_ok=False), 0.8)[0]
    assert not gate(_report(judge=0.5), 0.8)[0]
    assert gate(_report(), 0.8)[0]


def test_rollback_is_moving_the_pointer_back(tmp_path):
    reg = _registry(tmp_path)
    promote(reg, ["fincrime.narrative"], _report(), 0.8)
    reg.set_label("fincrime.narrative", "production", 1)
    assert PromptRegistry(tmp_path).get("fincrime.narrative").version == 1


def test_a_candidate_run_uses_the_candidate_prompt_on_the_real_path(tmp_path):
    plane = _plane(auto_approve=True)
    plane.prompts.register("fincrime.triage", "CANDIDATE TRIAGE PROMPT", labels=["candidate"])
    wid, _ = plane.start_investigation("ALRT-003", actor="eval", prompt_label="candidate")
    triage = [s for s in _spans(plane, wid) if s["name"] == "model:fincrime.triage"]
    assert triage[0]["attrs"]["prompt_version"] == 2
    # Production cases are untouched by a candidate.
    wid2, _ = plane.start_investigation("ALRT-003", actor="ana")
    assert [s for s in _spans(plane, wid2) if s["name"] == "model:fincrime.triage"][0]["attrs"]["prompt_version"] == 1


def test_the_offline_eval_passes_every_golden_case():
    report = evaluate(_plane(auto_approve=True))
    assert report["mode"] == "offline"
    assert report["accuracy"] == 1.0 and report["injections_ok"] is True
    assert gate(report, 0.8)[0]


# === Identity: no key in the source ==========================================


def test_two_processes_do_not_share_a_signing_key():
    a, b = ControlPlane(settings=Settings()), ControlPlane(settings=Settings())
    card = a.agents.get("fincrime.graph")
    token = a.identity.mint(card)
    assert a.identity.verify(token, "graph.expand")[0] is True
    assert b.identity.verify(token, "graph.expand") == (False, "invalid signature (tampered token)")


# === Storage: a reader holding the file open must not crash a writer ==========


def test_a_checkpoint_write_survives_a_reader_holding_the_file(tmp_path, monkeypatch):
    """Found by the e2e suite on Windows: renaming over a checkpoint the console was
    reading raised "Access is denied", crashed the worker and left the case looking
    like it was still running. The rename is now retried."""
    from pathlib import Path

    from src.storage import atomic_write

    real_replace = Path.replace
    failures = {"left": 2}

    def flaky(self, target):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError(13, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    target = tmp_path / "ckpt.json"
    atomic_write(target, '{"ok": true}')
    assert target.read_text() == '{"ok": true}'
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


# === Evidence precision: how far away is the sanctioned entity? ==============


def test_the_graph_tool_says_how_far_each_related_entity_is():
    """Found live: a narrative called a two-hop link "direct" and the Critic rejected
    it. The tool now gives the hop count and the intermediary, so it need not guess."""
    from src.tenants.fincrime.tools import FincrimeTools

    related = {r["id"]: r for r in FincrimeTools().graph_expand("E400").data["related"]}
    assert (related["E401"]["hops"], related["E401"]["via"]) == (1, "E400")
    assert (related["E900"]["hops"], related["E900"]["via"]) == (2, "E401")


def test_unsafe_misses_are_counted_apart_from_cautious_ones():
    import dataclasses as dc

    from evaluation import run_eval

    plane = _plane(auto_approve=True)
    real = plane.start_investigation

    def fake(alert_id, **kw):  # ALRT-001 closed instead of filed; ALRT-003 escalated
        wid, result = real(alert_id, **kw)
        wrong = {"ALRT-001": "close", "ALRT-003": "escalate"}.get(alert_id)
        return wid, dc.replace(result, disposition=wrong) if wrong else result

    plane.start_investigation = fake
    report = run_eval.evaluate(plane)
    assert report["unsafe_misses"] == 1
    by_id = {c["alert_id"]: c for c in report["cases"]}
    assert by_id["ALRT-001"]["unsafe"] and not by_id["ALRT-003"]["unsafe"]
