"""The entrypoint your evaluation harness calls.

YOU IMPLEMENT THIS. It is the seam between the pieces you built and the gate that
scores them, and it is deliberately the smallest interface in the repo:

    answer(ticket: str) -> str

`eval/run_eval.py` picks this up automatically. Until it exists, the runner
substitutes a null system that answers nothing, every quality score reads 0.00,
and the coverage gates fire first. That ordering is intentional: there is no point
tuning a score computed over 12 cases.

What lives behind this function is your architecture decision, and it is the one
worth writing an ADR about. A reasonable shape for this engagement:

    ticket text
      -> retrieve relevant policy (src/retrieval/hybrid.py)
      -> look up the shipment in the legacy system (via your MCP tools)
      -> decide the failure mode and the action
      -> return a grounded answer that cites the policy it relied on

Things the gate will notice:

- **Grounding.** The faithfulness floor in eval/thresholds.yaml exists because an
  answer that cites nothing cannot be checked by the customer. Return the policy
  id you used.
- **Refusal.** Some tickets do not contain enough information to act on. A system
  that always answers scores worse than one that says so, because a confident
  wrong routing costs a real delivery.
- **The contradiction.** Two of their policy documents disagree about redelivery
  attempts. Whatever your system does when it retrieves both, do it deliberately
  and be able to explain it in the demo. This is the single best thing you will
  find during this engagement and it belongs in your exec summary.
- **Cost and latency.** Wrap your model calls in the tracer from
  src/observability/trace.py and the gate reads the numbers for free.

Keep this function thin. Orchestration here, logic in the modules, so the pieces
stay testable on their own.
"""
from __future__ import annotations


def answer(ticket: str) -> str:
    """Answer one support ticket.

    Args:
        ticket: free text from customer/tickets.jsonl.

    Returns:
        The response your system would give, as a string. Include the policy id
        you grounded on. Return an explicit refusal when the ticket does not
        support a decision.
    """
    raise NotImplementedError(
        "Implement answer(). Run `python eval/run_eval.py` to see what it is scored on."
    )
