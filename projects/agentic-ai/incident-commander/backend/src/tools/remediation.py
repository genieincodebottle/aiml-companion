"""The write-side tool: execute a remediation against the sandbox.

This is the ONLY function in the system that mutates the stack. It is never
called directly by an agent; the orchestrator calls it only after the guardrail
manager has cleared the action (auto-approved low risk, or human-approved). That
chokepoint is what makes the agent safe to give real tools in production.
"""

from __future__ import annotations

from typing import Any

from ..models.state import ActionStatus, RemediationAction
from ..sandbox.stack import SandboxStack


def execute_remediation(stack: SandboxStack, action: RemediationAction) -> dict[str, Any]:
    """Apply an approved action to the sandbox and stamp the result on it."""
    if action.status not in (ActionStatus.APPROVED,):
        return {"executed": False, "reason": f"action not approved (status={action.status.value})"}

    result = stack.apply_action(action.action_type)
    action.status = ActionStatus.EXECUTED if result.get("applied") else ActionStatus.FAILED
    action.result = result
    return {"executed": True, **result}
