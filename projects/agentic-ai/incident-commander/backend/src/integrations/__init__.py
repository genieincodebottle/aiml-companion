"""The seams for driving this agent loop against real infrastructure.

Nothing in here is required to run the project: the sandbox satisfies both
protocols already. See ADR-0010 and docs/integration.md.
"""

from .protocols import ActionExecutor, ActionResult, ReversibleExecutor, TelemetrySource
from .registry import DEFAULT_ACTIONS, ActionSpec, allowed_for_role, get_action

__all__ = [
    "ActionExecutor",
    "ActionResult",
    "ActionSpec",
    "DEFAULT_ACTIONS",
    "ReversibleExecutor",
    "TelemetrySource",
    "allowed_for_role",
    "get_action",
]
