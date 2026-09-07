"""The service layer: orchestration and policy.

    app/            UI. Renders. Decides nothing.
    api/routes_*    Routing. Validates request shape, calls ONE service method,
                    maps the result to a response model.
    src/services/   Orchestration and policy. What order things happen in,
                    which guardrail runs when, what an operation means.
    src/            Capabilities. Retrieval, graph, LLM, guardrails - each does
                    one thing and knows nothing about HTTP.

The rule that keeps the boundary real: nothing under `src/` may import
`fastapi`. Services raise domain exceptions (GuardrailViolation, BudgetExceeded,
JobBusy, ValueError) and the routing layer decides what status code each one
means. That is what makes every operation callable from a notebook, a CLI or a
test with no web server running - and it is why the CLI in `run.py` and the API
cannot drift apart, because they call the same methods.
"""

from .graph_service import GraphService
from .jobs import JobBusy, JobService
from .qa import AnswerBundle, ComparisonBundle, QAService
from .security_service import SecurityService

__all__ = [
    "GraphService", "JobService", "JobBusy", "QAService",
    "AnswerBundle", "ComparisonBundle", "SecurityService",
]
