"""A self-contained simulated production stack the agent can safely break and fix.

This is what makes the project demoable end-to-end with no real infrastructure:
a fault injector corrupts the stack, the agent reads telemetry, and remediation
actions either fix it (correct action) or not (wrong action -> the loop continues).
"""

from .scenarios import SCENARIOS, Scenario, get_scenario, list_scenarios
from .stack import SandboxStack

__all__ = ["SCENARIOS", "Scenario", "get_scenario", "list_scenarios", "SandboxStack"]
