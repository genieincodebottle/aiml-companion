"""Repository tools: read, list, grep, and write files inside a worktree.

These are plain, deterministic Python functions. The `ToolInvoker` (registry.py)
wraps them with guardrail hooks, budget accounting, and event emission; the live
Claude Agent SDK worker exposes the very same functions as MCP tools. Every handler
takes the worktree as its first argument and treats all paths as relative to it.
"""

from __future__ import annotations

from pathlib import Path


def _resolve(worktree: Path, rel: str) -> Path:
    """Resolve a repo-relative path, refusing anything that escapes the worktree.

    Containment applies to READS as well as writes: without it, `read_file` /
    `grep` could exfiltrate arbitrary host files (e.g. ../../.env, SSH keys) into
    model context and the event stream.
    """
    target = (worktree / rel).resolve()
    if not target.is_relative_to(worktree.resolve()):
        raise ValueError(f"path '{rel}' escapes the repository worktree")
    return target


def _iter_contained(worktree: Path, glob: str):
    """Glob inside the worktree, yielding only files that stay inside it."""
    root = worktree.resolve()
    for p in worktree.glob(glob):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rp = p.resolve()
        if rp.is_relative_to(root):
            yield rp.relative_to(root), rp


def list_files(worktree: Path, glob: str = "**/*.py") -> dict:
    files = sorted(str(rel).replace("\\", "/") for rel, _ in _iter_contained(worktree, glob))
    return {"files": files, "count": len(files)}


def read_file(worktree: Path, path: str) -> dict:
    target = _resolve(worktree, path)
    if not target.is_file():
        return {"path": path, "content": "", "error": "file not found"}
    return {"path": path, "content": target.read_text(encoding="utf-8", errors="replace")}


def grep(worktree: Path, pattern: str, glob: str = "**/*.py") -> dict:
    import re

    rx = re.compile(pattern)
    matches: list[dict] = []
    for rel, p in _iter_contained(worktree, glob):
        rel_str = str(rel).replace("\\", "/")
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if rx.search(line):
                matches.append({"path": rel_str, "line": i, "text": line.strip()})
    return {"pattern": pattern, "matches": matches, "count": len(matches)}


def write_file(worktree: Path, path: str, content: str) -> dict:
    """Write a file inside the worktree. Path policy is enforced by the hook BEFORE
    this ever runs, so by the time we are here the target is known to be safe."""
    target = _resolve(worktree, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": path, "bytes": len(content.encode("utf-8"))}
