"""
LangChain tool-call tools backed by agent memory.

These are proper @tool-decorated functions that LLMs can call via
tool_choice / function_calling. They give agents access to long-term
and episodic memory without hardcoding retrieval into the prompt.

Learner takeaway:
  Tool calling = the LLM decides WHEN to search memory, not the code.
  The agent sees the tool descriptions and chooses to call them based
  on its reasoning. This is fundamentally different from always injecting
  all memory into the prompt (which wastes tokens and dilutes context).

Usage:
  from src.tools.memory_tools import MEMORY_TOOLS
  llm_with_tools = get_llm().bind_tools(MEMORY_TOOLS)
"""
from __future__ import annotations

import json
from langchain_core.tools import tool

from src.memory.manager import memory


@tool
def search_similar_claims(description: str, max_results: int = 3) -> str:
    """Search past claims for ones similar to the given description.

    Use this to find precedents: what happened with similar claims before?
    Returns claim IDs, outcomes, settlement amounts, and fraud scores.
    Helps calibrate decisions against historical data.

    Args:
        description: The incident description to search for similar claims.
        max_results: Number of similar claims to return (default 3).
    """
    results = memory.recall_similar_claims(description, k=max_results)
    if not results:
        return "No similar past claims found in memory. This may be a first-of-its-kind claim."
    lines = [f"Found {len(results)} similar past claims:"]
    for r in results:
        meta = r.get("metadata", {})
        lines.append(
            f"  - {r['claim_id']}: decision={meta.get('decision', '?')}, "
            f"settlement={meta.get('settlement_amount', '?')}, "
            f"fraud_score={meta.get('fraud_score', '?')}, "
            f"type={meta.get('incident_type', '?')}"
        )
        if r.get("description"):
            lines.append(f"    desc: {r['description'][:120]}...")
    return "\n".join(lines)


@tool
def search_fraud_episodes(claim_description: str, max_results: int = 3) -> str:
    """Search episodic memory for past fraud-related events similar to this claim.

    Use this to check: has a similar-sounding claim been confirmed as fraud before?
    Was a similar claim overridden by a human reviewer? Returns narratives
    of what happened in those past cases.

    Args:
        claim_description: The current claim's description to match against past episodes.
        max_results: Number of episodes to return (default 3).
    """
    episodes = memory.recall_episodes(
        claim_description, k=max_results, event_type=None
    )
    if not episodes:
        return "No relevant past episodes found in memory."
    lines = [f"Found {len(episodes)} relevant past episodes:"]
    for ep in episodes:
        meta = ep.get("metadata", {})
        lines.append(
            f"  - [{meta.get('event_type', '?')}] claim {meta.get('claim_id', '?')}: "
            f"{ep.get('narrative', '')[:150]}"
        )
    return "\n".join(lines)


@tool
def search_fraud_patterns(claim_description: str, max_results: int = 5) -> str:
    """Search the fraud knowledge base for known fraud patterns matching this claim.

    Use this to check if the claim's narrative matches any known fraud patterns
    (staged accidents, phantom claims, inflated bills, etc.).

    Args:
        claim_description: The claim description to match against known fraud patterns.
        max_results: Number of patterns to return (default 5).
    """
    patterns = memory.recall_fraud_patterns(claim_description, k=max_results)
    if not patterns:
        return "No matching fraud patterns found in the knowledge base."
    lines = [f"Found {len(patterns)} potentially matching fraud patterns:"]
    for p in patterns:
        meta = p.get("metadata", {})
        lines.append(
            f"  - [{meta.get('risk_level', '?')}] {p.get('description', '')[:150]}"
        )
    return "\n".join(lines)


@tool
def lookup_claim_policy(policy_number: str) -> str:
    """Look up a policy in the database to check coverage, limits, and status.

    Args:
        policy_number: The policy number to look up (e.g., POL-AUTO-TEST-US).
    """
    from src.tools.policy_lookup import lookup_policy
    policy = lookup_policy(policy_number)
    if not policy:
        return f"Policy {policy_number} not found in the database."

    coverage = policy.get("coverage", {})
    return (
        f"Policy: {policy_number}\n"
        f"  Type: {policy.get('type', '?')}\n"
        f"  Status: {policy.get('status', '?')}\n"
        f"  Start: {policy.get('start_date', '?')} End: {policy.get('end_date', '?')}\n"
        f"  Deductible: {policy.get('deductible', 0)}\n"
        f"  Coverage: {json.dumps(coverage, indent=4)}\n"
        f"  Exclusions: {', '.join(policy.get('exclusions', []))}\n"
        f"  Prior claims: {policy.get('claims_count', 0)}"
    )


# All memory-backed tools, ready to bind to any LLM
MEMORY_TOOLS = [
    search_similar_claims,
    search_fraud_episodes,
    search_fraud_patterns,
    lookup_claim_policy,
]
