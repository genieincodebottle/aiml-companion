"""Mission Control CLI: drive the platform from a terminal, no browser, no key needed.

Commands:
    python cli.py alerts                    the alert catalog + ground truth
    python cli.py agents                    the registered agent cards
    python cli.py run ALRT-001              one case: analyst opens it, a reviewer decides
    python cli.py run ALRT-001 --fault      inject a commit fault and watch the saga compensate
    python cli.py run ALRT-001 --reject     the reviewer rejects instead
    python cli.py stream ALRT-001           the same, printing every event as it happens
    python cli.py cases                     recent cases from disk (they survive restarts)
    python cli.py decide WF-ID approve|reject --as NAME   decide a parked case
    python cli.py prompts                   prompt versions and labels
    python cli.py prompts rollback NAME VERSION          move `production` back
    python cli.py eval                      the evaluation + promotion gate
"""

from __future__ import annotations

import sys

from src.control_plane import ControlPlane
from src.models.events import EventLog
from src.runtime import MakerCheckerViolation
from src.tenants.fincrime.data import expected_disposition, list_alerts

ANALYST, REVIEWER = "ana.analyst", "rex.reviewer"


def _flag(args: list[str], name: str) -> str | None:
    if name in args:
        i = args.index(name)
        return args[i + 1] if i + 1 < len(args) else None
    return None


def cmd_alerts(_: ControlPlane) -> int:
    print(f"{'ALERT':<10} {'TYPE':<15} {'AMOUNT':>10}  EXPECTED   DESCRIPTION")
    for a in list_alerts():
        print(
            f"{a.id:<10} {a.alert_type:<15} {a.amount_usd:>10,.0f}  "
            f"{expected_disposition(a.id):<9}  {a.description[:46]}"
        )
    return 0


def cmd_agents(plane: ControlPlane) -> int:
    print(f"{'AGENT':<22} {'ROLE':<19} {'TIER':<8} SIDE-FX  TOOLS")
    for c in plane.list_agent_cards():
        tools = ",".join(c.allowed_tools) or "-"
        print(f"{c.name:<22} {c.role:<19} {c.model_tier.value:<8} {str(c.side_effecting):<7}  {tools}")
    return 0


def _dump(plane: ControlPlane, wid: str, seen: int) -> int:
    events = EventLog(wid, plane.settings.data_dir).replay()
    for evt in events[seen:]:
        payload = {k: v for k, v in evt.payload.items() if k != "completed"}
        print(f"  {evt.seq:>3}  {evt.type.value:<22} {payload}")
    return len(events)


def cmd_run(plane: ControlPlane, alert_id: str, *, fault: bool, reject: bool, stream: bool) -> int:
    print(f"mode={plane.settings.execution_mode}  policy={'OFF' if plane.settings.disable_policy else 'on'}")
    wid, result = plane.start_investigation(alert_id, actor=ANALYST, inject_commit_fault=fault)
    seen = _dump(plane, wid, 0) if stream else 0
    if result.status == "awaiting_approval":
        pa = result.pending_action or {}
        print(f"\n[GATE] {wid} is waiting for a reviewer: {pa.get('tool')} {pa.get('args')}")
        try:
            plane.decide(wid, "approve", "self-approval attempt", actor=ANALYST)
        except MakerCheckerViolation as exc:
            print(f"   maker-checker: {exc}")
        decision = "reject" if reject else "approve"
        print(f"   {REVIEWER} decides: {decision}\n")
        result = plane.decide(wid, decision, f"{decision} via CLI", actor=REVIEWER)
        if stream:
            _dump(plane, wid, seen)
    view = plane.view(wid)
    print(f"workflow    : {wid}")
    print(f"status      : {view.status.value}")
    print(f"disposition : {view.disposition}")
    print(f"opened by   : {view.started_by}   decided by: {view.decided_by} ({view.decision})")
    print(f"steps       : {view.step_count}   model spend: ${view.cost_usd:.6f}")
    if view.dead_letter_reason:
        print(f"dead-letter : {view.dead_letter_reason}")
    if result.state.get("compensated"):
        print("saga        : commit failed -> compensated -> escalated to a person")
    return 0


def cmd_cases(plane: ControlPlane) -> int:
    print(f"{'WORKFLOW':<26} {'STATUS':<18} {'DISPOSITION':<12} {'OPENED BY':<14} DECIDED BY")
    for v in plane.list_views(20):
        print(
            f"{v.workflow_id:<26} {v.status.value:<18} {v.disposition or '-':<12} "
            f"{v.started_by or '-':<14} {v.decided_by or '-'}"
        )
    return 0


def cmd_decide(plane: ControlPlane, wid: str, decision: str, actor: str | None) -> int:
    if not actor:
        print("usage: cli.py decide WF-ID approve|reject --as NAME")
        return 1
    result = plane.decide(wid, decision, f"{decision} via CLI", actor=actor)
    print(f"{wid}: {result.status}  disposition={result.disposition}")
    return 0


def cmd_prompts(plane: ControlPlane, rest: list[str]) -> int:
    if rest[:1] == ["rollback"] and len(rest) == 3:
        plane.prompts.set_label(rest[1], "production", int(rest[2]))
        print(f"{rest[1]}: production -> v{rest[2]}")
    for p in plane.prompts.summary():
        labels = ", ".join(f"{k}=v{v}" for k, v in sorted(p["labels"].items()))
        print(f"{p['name']:<22} versions={p['versions']}  {labels}")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "eval":
        from evaluation.run_eval import main as eval_main

        return eval_main(rest)
    plane = ControlPlane()
    positional = [r for i, r in enumerate(rest) if not r.startswith("--") and (i == 0 or rest[i - 1] != "--as")]
    if cmd == "alerts":
        return cmd_alerts(plane)
    if cmd == "agents":
        return cmd_agents(plane)
    if cmd in {"run", "stream"} and positional:
        return cmd_run(plane, positional[0], fault="--fault" in rest, reject="--reject" in rest, stream=cmd == "stream")
    if cmd == "cases":
        return cmd_cases(plane)
    if cmd == "decide" and len(positional) == 2:
        return cmd_decide(plane, positional[0], positional[1], _flag(rest, "--as"))
    if cmd == "prompts":
        return cmd_prompts(plane, positional)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
