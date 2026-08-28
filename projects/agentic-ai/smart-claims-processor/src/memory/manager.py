"""
Memory Manager - read/write interface for all three memory types.

This is what agents call. They don't touch ChromaDB directly.

Usage in agents:
    from src.memory.manager import memory

    # Retrieve similar past claims (long-term)
    similar = memory.recall_similar_claims("rear-ended at a red light", k=3)

    # Retrieve relevant episodes (episodic)
    episodes = memory.recall_episodes("high-value auto theft, no police report", k=3)

    # Store a completed claim for future retrieval
    memory.store_claim_outcome(claim_id, description, metadata)

    # Store a notable episode
    memory.store_episode(claim_id, narrative, event_type, metadata)
"""
from __future__ import annotations

import logging
from typing import Any

from src.memory import store

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified interface for short-term, long-term, and episodic memory."""

    # ── Long-term memory (all past claims) ──────────────────────────────

    def store_claim_outcome(
        self,
        claim_id: str,
        description: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store a completed claim in long-term memory for future retrieval.

        Called after every pipeline completion (in graph.py).
        The description is embedded; metadata is stored alongside for filtering.
        """
        col = store.long_term()
        # Flatten metadata values to strings (ChromaDB requirement)
        safe_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                     for k, v in metadata.items() if v is not None}
        try:
            col.upsert(
                ids=[claim_id],
                documents=[description],
                metadatas=[safe_meta],
            )
            logger.debug("Stored claim %s in long-term memory", claim_id)
        except Exception as e:
            logger.warning("Failed to store claim %s in memory: %s", claim_id, e)

    def recall_similar_claims(
        self,
        query: str,
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Retrieve the k most similar past claims.

        Returns list of dicts with: claim_id, description, distance, metadata.
        Agents use this for context: "similar claims were approved at $X" or
        "similar claims had fraud score Y".
        """
        col = store.long_term()
        if col.count() == 0:
            return []
        try:
            results = col.query(
                query_texts=[query],
                n_results=min(k, col.count()),
                where=where,
            )
        except Exception as e:
            logger.warning("Similar claims query failed: %s", e)
            return []

        claims = []
        for i in range(len(results["ids"][0])):
            claims.append({
                "claim_id": results["ids"][0][i],
                "description": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return claims

    # ── Episodic memory (notable events) ────────────────────────────────

    def store_episode(
        self,
        claim_id: str,
        narrative: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a notable episode - human override, fraud confirmation, appeal, etc.

        event_type: "human_override" | "fraud_confirmed" | "appeal_granted" |
                    "appeal_denied" | "auto_rejected" | "quality_gate_failed"
        """
        col = store.episodic()
        episode_id = f"{claim_id}_{event_type}"
        meta = {"claim_id": claim_id, "event_type": event_type}
        if metadata:
            meta.update({k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                         for k, v in metadata.items() if v is not None})
        try:
            col.upsert(ids=[episode_id], documents=[narrative], metadatas=[meta])
            logger.debug("Stored episode %s for claim %s", event_type, claim_id)
        except Exception as e:
            logger.warning("Failed to store episode: %s", e)

    def recall_episodes(
        self,
        query: str,
        k: int = 3,
        event_type: str | None = None,
    ) -> list[dict]:
        """Retrieve relevant past episodes. Optionally filter by event_type."""
        col = store.episodic()
        if col.count() == 0:
            return []
        where = {"event_type": event_type} if event_type else None
        try:
            results = col.query(
                query_texts=[query],
                n_results=min(k, col.count()),
                where=where,
            )
        except Exception as e:
            logger.warning("Episode recall failed: %s", e)
            return []

        episodes = []
        for i in range(len(results["ids"][0])):
            episodes.append({
                "episode_id": results["ids"][0][i],
                "narrative": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return episodes

    # ── Fraud knowledge base ────────────────────────────────────────────

    def seed_fraud_knowledge(self, patterns: list[dict]) -> None:
        """Bulk-load fraud patterns into the knowledge base.

        Each pattern: {"id": str, "description": str, "risk_level": str, ...}
        Called once on startup or via a seed script.
        """
        col = store.fraud_knowledge()
        if col.count() > 0:
            return  # already seeded
        ids, docs, metas = [], [], []
        for p in patterns:
            ids.append(p["id"])
            docs.append(p["description"])
            metas.append({k: str(v) for k, v in p.items() if k not in ("id", "description")})
        try:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info("Seeded %d fraud patterns into knowledge base", len(ids))
        except Exception as e:
            logger.warning("Fraud knowledge seeding failed: %s", e)

    def recall_fraud_patterns(self, claim_description: str, k: int = 5) -> list[dict]:
        """Find fraud patterns similar to the claim description."""
        col = store.fraud_knowledge()
        if col.count() == 0:
            return []
        try:
            results = col.query(query_texts=[claim_description], n_results=min(k, col.count()))
        except Exception as e:
            logger.warning("Fraud pattern recall failed: %s", e)
            return []

        patterns = []
        for i in range(len(results["ids"][0])):
            patterns.append({
                "pattern_id": results["ids"][0][i],
                "description": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return patterns

    # ── Short-term memory ───────────────────────────────────────────────
    # Note: Short-term memory is LangGraph's ClaimsState itself.
    # Each agent reads prior agents' outputs from the state dict.
    # No separate implementation needed - the state IS working memory.
    # This method formats it for prompt injection.

    def format_pipeline_context(self, state: dict) -> str:
        """Format what previous agents found as a short-term memory summary.

        Injected into each agent's prompt so it knows what happened upstream.
        """
        from src.utils import currency_symbol
        cs = currency_symbol()
        parts = []
        if state.get("intake_output"):
            io = state["intake_output"]
            parts.append(f"Intake: valid={getattr(io, 'is_valid', '?')}, confidence={getattr(io, 'confidence', '?')}")
        if state.get("fraud_output"):
            fo = state["fraud_output"]
            parts.append(f"Fraud: score={getattr(fo, 'fraud_score', '?')}, level={getattr(fo, 'fraud_risk_level', '?')}")
        if state.get("damage_output"):
            do = state["damage_output"]
            parts.append(f"Damage: assessed={cs}{getattr(do, 'assessed_damage_usd', '?'):,.2f}")
        if state.get("policy_output"):
            po = state["policy_output"]
            parts.append(f"Policy: status={getattr(po, 'coverage_status', '?')}, limit={cs}{getattr(po, 'covered_amount_usd', '?'):,.2f}")
        if state.get("settlement_output"):
            so = state["settlement_output"]
            parts.append(f"Settlement: {cs}{getattr(so, 'settlement_amount_usd', '?'):,.2f}, decision={getattr(so, 'decision', '?')}")
        return "\n".join(parts) if parts else "No prior agent outputs yet."


# Singleton - all agents share one instance
memory = MemoryManager()
