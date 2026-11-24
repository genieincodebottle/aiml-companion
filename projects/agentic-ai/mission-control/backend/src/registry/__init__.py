"""The prompt registry and agent registry - versioned platform configuration."""

from .agents import AgentRegistry
from .prompts import PromptRegistry

__all__ = ["AgentRegistry", "PromptRegistry"]
