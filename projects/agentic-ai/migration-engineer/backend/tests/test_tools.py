"""Tools: repo ops, the migration-rules retriever, and cross-repo memory."""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.tools import repo
from src.tools.memory import MemoryStore
from src.tools.rules import search_migration_rules
from src.worktree import make_worktree


def _worktree(tmp_id: str = "t") -> Path:
    return make_worktree("billing-service", f"tools-{tmp_id}", get_settings().data_dir)


def test_list_read_grep():
    wt = _worktree("lrg")
    files = repo.list_files(wt)["files"]
    assert "billing.py" in files
    content = repo.read_file(wt, "billing.py")["content"]
    assert "utcnow" in content
    hits = repo.grep(wt, r"datetime\.utcnow\(\)")
    assert hits["count"] >= 1
    assert any(h["path"] == "billing.py" for h in hits["matches"])


def test_write_file_roundtrip():
    wt = _worktree("wf")
    repo.write_file(wt, "billing.py", "X = 1\n")
    assert repo.read_file(wt, "billing.py")["content"] == "X = 1\n"


def test_rules_retrieval_ranks_datetime_first():
    res = search_migration_rules(query="datetime utcnow timezone deprecation")
    assert res["results"][0]["id"] == "modernize-datetime"
    # A retriever, not a lookup: an unrelated query should rank a different rule top.
    http = search_migration_rules(query="requests http client async")
    assert http["results"][0]["id"] == "requests-to-httpx"


def test_memory_records_and_recalls_across_repos():
    store = MemoryStore(get_settings().data_dir, "mem-job")
    store.record(note="adding timezone import was required", repo="auth-service", tags="datetime timezone")
    out = store.recall(query="timezone import")
    assert out["count"] == 1
    assert "timezone" in out["results"][0]["note"]
