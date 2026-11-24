"""The financial-crime investigation squad - the platform's flagship tenant.

A supervisor plans an AML investigation; specialist workers (triage, graph, screening,
narrative, critic) each run as a governed agent: their model calls go through the
gateway, their tool calls through the policy engine + identity broker, and their
disposition waits at a human gate. The squad proves the platform with a workload
demanding enough to be believed - streaming alerts, a knowledge graph, hybrid
evidence, evidence-cited generation, and regulated human-in-the-loop sign-off.
"""

from .squad import FincrimeSquad

__all__ = ["FincrimeSquad"]
