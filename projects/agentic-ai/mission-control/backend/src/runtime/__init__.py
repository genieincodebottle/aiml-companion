"""The durable agent runtime - where workflows actually execute.

Durable execution is the reliability spine: a workflow is an event-sourced state
machine that checkpoints after every step, so a crash (or a human pause that lasts
days) is resumed by replaying to the last checkpoint - without re-billing completed
model calls. Side-effecting commits run inside a saga so a partial failure is
compensated, and a workflow that blows a budget or dies is dead-lettered, never lost.
"""

from .engine import AlreadyDecided, MakerCheckerViolation, NotAwaitingApproval, Runtime
from .workflow import WorkflowContext, WorkflowResult

__all__ = [
    "AlreadyDecided",
    "MakerCheckerViolation",
    "NotAwaitingApproval",
    "Runtime",
    "WorkflowContext",
    "WorkflowResult",
]
