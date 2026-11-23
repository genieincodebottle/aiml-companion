"""Tools the agents use: read-only observability and (gated) remediation.

The split in this package IS the safety model. Everything in `observability` is
side-effect free and freely callable by an agent. `remediation` contains exactly
one mutating function, and it refuses to run on an action the guardrail layer and
(where required) a human have not approved.
"""

from .observability import (
    DEFAULT_BUNDLE,
    TOOL_REGISTRY,
    UNTRUSTED_TOOLS,
    ToolSpec,
    describe_tools,
    get_config_diff,
    get_dependency_health,
    get_deploy_history,
    get_logs,
    get_metrics,
    get_pool_stats,
    get_slo_burn,
    get_traces,
    observe_all,
    run_tool,
    run_tools,
)
from .remediation import execute_remediation

__all__ = [
    # registry
    "TOOL_REGISTRY",
    "DEFAULT_BUNDLE",
    "UNTRUSTED_TOOLS",
    "ToolSpec",
    "describe_tools",
    "run_tool",
    "run_tools",
    "observe_all",
    # individual read-only tools
    "get_metrics",
    "get_logs",
    "get_traces",
    "get_deploy_history",
    "get_pool_stats",
    "get_config_diff",
    "get_dependency_health",
    "get_slo_burn",
    # the single write-side tool
    "execute_remediation",
]
