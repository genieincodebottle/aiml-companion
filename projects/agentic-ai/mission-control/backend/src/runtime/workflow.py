"""Workflow context + the tenant-workload interface.

`WorkflowContext` is what a tenant's step function is handed. It is the ONLY way a
step touches the platform: model calls go through `call_model` (gateway: routing,
cache, budget, redaction, audit), tool calls through `call_tool` (identity token +
policy check + injection scan). A step never imports the gateway or a provider
directly - that is what keeps every guarantee a platform property.

A `Workload` is a tenant's declaration of an investigation as an ordered list of
`Step`s. Steps are deliberately linear and named so the engine can checkpoint after
each and resume by skipping completed ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel

from ..gateway import Gateway
from ..guardrails import neutralize, scan_for_injection
from ..identity import IdentityBroker
from ..models.events import EventLog, EventType
from ..models.schemas import PolicyEffect
from ..observability import Tracer
from ..policy import PolicyEngine
from ..registry import AgentRegistry, PromptRegistry

T = TypeVar("T", bound=BaseModel)


class PolicyDenied(RuntimeError):
    """A tool call the policy engine refused (fail-closed)."""


def _neutralize_value(value: Any) -> Any:
    """Neutralize injection phrasings in strings while preserving the value's type.

    Stringifying non-string values (lists, dicts, numbers) would corrupt tool data -
    e.g. an empty list becoming the truthy string "[]" - so only strings are rewritten.
    """
    if isinstance(value, str):
        return neutralize(value)
    if isinstance(value, list):
        return [_neutralize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _neutralize_value(v) for k, v in value.items()}
    return value


@dataclass
class Step:
    """One named unit of work in a workflow. `side_effecting` steps are HITL-gated."""

    name: str
    fn: Callable[["WorkflowContext"], None]
    side_effecting: bool = False


class WorkflowContext:
    """Handed to every step. Wraps the platform planes behind a small surface."""

    def __init__(
        self,
        *,
        workflow_id: str,
        gateway: Gateway,
        policy: PolicyEngine,
        identity: IdentityBroker,
        agents: AgentRegistry,
        prompts: PromptRegistry,
        events: EventLog,
        tracer: Tracer,
        state: dict[str, Any],
    ) -> None:
        self.workflow_id = workflow_id
        self.gateway = gateway
        self.policy = policy
        self.identity = identity
        self.agents = agents
        self.prompts = prompts
        self.events = events
        self.tracer = tracer
        self.state = state

    # -- model calls -------------------------------------------------------- #
    def call_model(self, agent_name: str, user: str, schema: Type[T], prompt_name: str) -> T:
        card = self.agents.get(agent_name)
        # The eval harness runs a case under the `candidate` label to score a new prompt
        # before it is promoted; everything else reads `production`.
        prompt = self.prompts.get(prompt_name, self.state.get("prompt_label", "production"))
        system = prompt.template
        with self.tracer.span(
            f"model:{agent_name}", agent=agent_name, schema=schema.__name__, prompt_version=prompt.version
        ):
            obj, _usage = self.gateway.complete(
                workflow_id=self.workflow_id,
                system=system,
                user=user,
                schema=schema,
                tier=card.model_tier,
                on_event=lambda name, payload: self.events.append(EventType(name), agent=agent_name, **payload),
            )
        return obj

    # -- tool calls --------------------------------------------------------- #
    def call_tool(self, agent_name: str, tool: str, tool_fn: Callable[..., Any], **args: Any) -> Any:
        """Authorize (identity + policy), run a read-only tool, scan its output."""
        card = self.agents.get(agent_name)

        # 1. Workload identity: mint a scoped token, verify it covers this tool.
        token = self.identity.mint(card)
        ok, reason = self.identity.verify(token, tool)
        if not ok:
            self.events.append(EventType.POLICY_DENIED, agent=agent_name, tool=tool, reason=reason)
            raise PolicyDenied(f"identity check failed for {agent_name}->{tool}: {reason}")

        # 2. Policy: read-only tools must resolve to ALLOW.
        decision = self.policy.evaluate(card, tool, args)
        if decision.effect != PolicyEffect.ALLOW:
            self.events.append(
                EventType.POLICY_DENIED, agent=agent_name, tool=tool, rule=decision.rule, reason=decision.reason
            )
            raise PolicyDenied(f"policy {decision.effect.value} for {agent_name}->{tool}: {decision.reason}")
        self.events.append(EventType.POLICY_ALLOWED, agent=agent_name, tool=tool, rule=decision.rule)

        # 3. Run the tool.
        with self.tracer.span(f"tool:{tool}", agent=agent_name, tool=tool):
            result = tool_fn(**args)

        # 4. Injection scan on any text the tool surfaces (indirect-injection defense).
        text = str(getattr(result, "data", result))
        found, hits = scan_for_injection(text)
        if found:
            self.events.append(EventType.INJECTION_BLOCKED, agent=agent_name, tool=tool, patterns=hits)
            if hasattr(result, "data") and isinstance(result.data, dict):
                result.data = {k: _neutralize_value(v) for k, v in result.data.items()}
        return result

    def emit(self, event_type: EventType, **payload: Any) -> None:
        self.events.append(event_type, **payload)


@dataclass
class WorkflowResult:
    status: str
    disposition: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    pending_action: dict[str, Any] | None = None


class Workload:
    """Base class a tenant implements. The engine drives `steps()` in order."""

    tenant: str = "base"
    #: The agent whose card the policy engine checks for the side-effecting commit.
    committer: str = ""

    def seed_state(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def rehydrate(self, state: dict[str, Any]) -> None:
        """Recreate in-process objects ('_'-prefixed keys) after a restart."""

    def steps(self) -> list[Step]:
        raise NotImplementedError
