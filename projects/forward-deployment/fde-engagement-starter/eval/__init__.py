"""Evaluation harness for the Northwind Freight engagement.

PARTIAL BY DESIGN. The runner works on clone and fails the gate, because the
seed golden set is 12 cases across 3 failure modes and the gate needs 30 across
5. Closing that gap from customer/tickets.jsonl is your work, not ours.
"""

__all__ = ["judge", "run_eval"]
