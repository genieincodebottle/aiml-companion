"""The control plane: the composition root that wires the planes together.

It builds the pieces every tenant workflow shares (gateway, policy engine, identity
broker, registries, runtime) and exposes the few operations the API and CLI drive:
open an investigation, decide a paused one, and read status. Keeping this apart from
FastAPI keeps the platform usable head-less (CLI, tests, evals) exactly as over HTTP.
"""

from __future__ import annotations

import logging
import uuid

from .config import Settings, get_settings
from .gateway import Gateway
from .identity import IdentityBroker
from .models.schemas import Alert, WorkflowStatus, WorkflowView
from .observability import configure_otel
from .policy import PolicyEngine
from .registry import AgentRegistry, PromptRegistry
from .runtime import Runtime, WorkflowResult
from .tenants.fincrime import FincrimeSquad
from .tenants.fincrime.data import get_alert, list_alerts

logger = logging.getLogger(__name__)

#: Tenant name -> workload factory. The runtime uses it to rebuild a parked case after
#: a restart, so adding a tenant means adding one line here.
WORKLOADS = {FincrimeSquad.tenant: FincrimeSquad}


class ControlPlane:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.tracing = configure_otel(self.settings.otel_exporter)
        self.gateway = Gateway(self.settings)
        self.policy = PolicyEngine(disabled=self.settings.disable_policy)
        self.identity = IdentityBroker(self.settings.signing_key)
        self.agents = AgentRegistry()
        self.prompts = PromptRegistry(self.settings.data_dir)
        self.runtime = Runtime(
            self.settings, self.gateway, self.policy, self.identity, self.agents, self.prompts, WORKLOADS
        )

    # -- catalog ------------------------------------------------------------ #
    def list_alerts(self) -> list[Alert]:
        return list_alerts()

    def list_agent_cards(self):
        return self.agents.list_cards()

    # -- operations --------------------------------------------------------- #
    def open_investigation(
        self,
        alert_id: str,
        *,
        actor: str,
        inject_commit_fault: bool = False,
        prompt_label: str = "production",
    ) -> str:
        """Register a case and return its id. Run it with `run_investigation`."""
        alert = get_alert(alert_id)
        if alert is None:
            raise KeyError(f"unknown alert {alert_id}")
        workflow_id = f"wf-{alert_id}-{uuid.uuid4().hex[:8]}"
        logger.info("opening %s for alert %s (by %s)", workflow_id, alert_id, actor)
        self.runtime.open(
            workflow_id,
            FincrimeSquad(),
            actor=actor,
            prompt_label=prompt_label,
            alert=alert,
            inject_commit_fault=inject_commit_fault,
        )
        return workflow_id

    def run_investigation(self, workflow_id: str) -> WorkflowResult:
        return self.runtime.advance(workflow_id)

    def start_investigation(
        self,
        alert_id: str,
        *,
        actor: str = "analyst",
        inject_commit_fault: bool = False,
        prompt_label: str = "production",
    ) -> tuple[str, WorkflowResult]:
        """Open and run to the first pause (CLI, tests, eval)."""
        wid = self.open_investigation(
            alert_id, actor=actor, inject_commit_fault=inject_commit_fault, prompt_label=prompt_label
        )
        return wid, self.run_investigation(wid)

    def decide(self, workflow_id: str, decision: str, note: str | None = None, *, actor: str) -> WorkflowResult:
        return self.runtime.resume_after_approval(workflow_id, decision, note, actor=actor)

    # -- views -------------------------------------------------------------- #
    def view(self, workflow_id: str) -> WorkflowView:
        run = self.runtime.get_run(workflow_id)
        if run is not None:
            return _view(
                workflow_id,
                run.workload.tenant,
                run.status,
                len(run.completed),
                self.runtime.cost_of(workflow_id),
                run.ctx.state,
                run.pending_action,
            )
        ckpt = self.runtime.load_checkpoint(workflow_id)
        if ckpt is None:
            raise KeyError(f"unknown workflow {workflow_id}")
        return _from_checkpoint(ckpt)

    def list_views(self, limit: int = 50) -> list[WorkflowView]:
        """Recent cases, newest first, from the checkpoint files (so they survive restarts)."""
        out = []
        for ckpt in self.runtime.list_checkpoints(limit):
            live = self.runtime.get_run(ckpt["workflow_id"])
            out.append(self.view(ckpt["workflow_id"]) if live else _from_checkpoint(ckpt))
        return out


def _from_checkpoint(ckpt: dict) -> WorkflowView:
    return _view(
        ckpt["workflow_id"],
        ckpt.get("tenant", "fincrime"),
        WorkflowStatus(ckpt["status"]),
        len(ckpt.get("completed", [])),
        ckpt.get("cost_usd", 0.0),
        ckpt.get("state", {}),
        ckpt.get("pending_action"),
    )


def _view(workflow_id, tenant, status, steps, cost, state, pending) -> WorkflowView:
    audit = state.get("audit", {})
    return WorkflowView(
        workflow_id=workflow_id,
        tenant=tenant,
        alert_id=state.get("alert", {}).get("id", ""),
        status=status,
        step_count=steps,
        cost_usd=cost,
        disposition=state.get("disposition"),
        pending_action=pending if status == WorkflowStatus.AWAITING_APPROVAL else None,
        started_by=audit.get("started_by"),
        decided_by=audit.get("decided_by"),
        decision=audit.get("decision"),
        decision_note=audit.get("note"),
        decided_at=audit.get("decided_at"),
        dead_letter_reason=state.get("dead_letter_reason"),
    )


_plane: ControlPlane | None = None


def get_control_plane() -> ControlPlane:
    global _plane
    if _plane is None:
        _plane = ControlPlane()
    return _plane


def reset_control_plane() -> None:
    """Test hook: drop the memoized plane so env overrides take effect.

    Settings alone are not enough. The plane wires up a gateway and a provider
    when it is constructed, so a plane built under a live environment stays live
    no matter what `reset_settings_cache()` does afterwards.
    """
    global _plane
    _plane = None
