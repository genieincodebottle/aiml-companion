"""
Run provenance - what produced a decision, recorded next to the decision.

The audit log's hash chain proves a line was not edited after the fact. It says
nothing about what produced that line, so a decision cannot be reconstructed
from the log alone. Three years later, on an appeal, the questions are:

  Which model decided this?      MODEL comes from the environment, so two
                                 claims in the same week can be decided by two
                                 different models with nothing recording it.
  What did the prompt say?       Prompts are Python strings in the agent
                                 modules, and the file today is not the file
                                 that ran.
  Which thresholds applied?      A YAML edit moves low_confidence from 0.65 to
                                 0.60 and both audit entries look identical.
  What did it read?              Memory grows daily, so the same claim re-run
                                 next month retrieves different neighbours.

This module answers those four with hashes rather than copies: no prompt text,
no claim text and no PII ever enters the record.

Collection mirrors token accounting exactly (see src/llm.py). A ContextVar
holds what the current node did, `guard_node` folds it into graph state after
the node returns, and the checkpointer persists it, so the record survives a
HITL pause and a resume in another process.

    from src.provenance import run_versions_update
    update["run_versions"] = run_versions_update(state, "intake_agent")

Not covered: the fraud crew talks to the provider through CrewAI's own client,
which never reaches a LangChain callback, so that node records no prompt hash.
Its config and model are still pinned by the run-level fields.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

SHA_LEN = 16   # first 16 hex chars: collision-safe enough to pin a version, short enough to read on screen


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SHA_LEN]


# ── Per-node accumulators ───────────────────────────────────────────────────
# One ContextVar per run, same reasoning as the token accumulator: each claim
# runs in its own thread, and a module-level list would mix two claims together.

_prompts_var: contextvars.ContextVar[list] = contextvars.ContextVar("claim_prompt_hashes")
_retrieved_var: contextvars.ContextVar[list] = contextvars.ContextVar("claim_retrieved_ids")


def _get(var: contextvars.ContextVar) -> list:
    try:
        return var.get()
    except LookupError:
        fresh: list = []
        var.set(fresh)
        return fresh


def start_node() -> None:
    """Clear the accumulators so what follows belongs to one node."""
    _prompts_var.set([])
    _retrieved_var.set([])


def record_prompt(text: str) -> None:
    """Hash one prompt sent to a model. Called from the LangChain callback."""
    if not text:
        return
    _get(_prompts_var).append(_sha(text))


def record_retrieval(ids) -> None:
    """Record the ids a retrieval returned, in the order the model saw them."""
    seen = _get(_retrieved_var)
    for i in ids or []:
        if i is not None and str(i) not in seen:
            seen.append(str(i))


def collect_node() -> tuple[str | None, list]:
    """Return (prompt hash for this node, retrieved ids for this node)."""
    prompts = _get(_prompts_var)
    prompt_sha = _sha("|".join(prompts)) if prompts else None
    return prompt_sha, list(_get(_retrieved_var))


# ── Run level identity ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def config_fingerprint() -> str:
    """One hash over every config file that can change a decision.

    Hashing the files rather than the resolved dict keeps comments and key
    order in scope, so an edit of any kind changes the fingerprint.
    """
    from src.config import CONFIG_PATH, _COUNTRIES_DIR

    parts = []
    for path in [CONFIG_PATH] + sorted(_COUNTRIES_DIR.glob("*.yaml")):
        try:
            parts.append(f"{path.name}:{_sha(path.read_text(encoding='utf-8'))}")
        except OSError as e:
            logger.warning("Config fingerprint skipped %s: %s", path, e)
    return _sha("|".join(parts))


def run_identity() -> dict:
    """The model, provider and config in force right now."""
    from src.config import get_llm_config

    cfg = get_llm_config()
    return {
        "provider": cfg.get("provider"),
        "model": cfg.get("model"),
        "judge_model": cfg.get("judge_model") or cfg.get("model"),
        "config_sha256": config_fingerprint(),
    }


def run_versions_update(state: dict, node_name: str) -> dict:
    """Fold this node's provenance into the run's record and return the whole thing.

    Merging rather than replacing is what makes the record survive a pause: the
    node that runs after a resume sees the fields written before it.
    """
    record = dict(state.get("run_versions") or {})
    record.update(run_identity())
    prompt_sha, retrieved = collect_node()
    if prompt_sha:
        prompts = dict(record.get("prompts") or {})
        prompts[node_name] = prompt_sha
        record["prompts"] = prompts
    if retrieved:
        reads = dict(record.get("retrieved") or {})
        reads[node_name] = retrieved
        record["retrieved"] = reads
    return record
