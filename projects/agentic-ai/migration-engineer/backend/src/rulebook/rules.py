"""The migration rulebook and the fleet job catalog.

A `MigrationRule` carries two very different things on purpose:

  * `guidance` - prose the *real* Claude Agent SDK worker reads. In `live-sdk` mode
    the worker gets ONLY this guidance and must reason out the concrete edits itself,
    exactly like a human engineer handed a migration ticket.

  * `edit_steps` - a scripted, ordered transformation the *deterministic* stub worker
    applies (one step, re-test, next step). This is what makes the offline demo
    reproducible and free. It is NOT given to the live worker.

This split is the honest core of the "two execution modes" design: the stub simulates
what Claude would do; both drive the identical tool + hook + budget contract.

The rulebook also holds a few rules the fixtures do NOT use. They exist so that
`search_migration_rules(query)` (the RAG tool) has to actually retrieve the right rule
from a realistic catalog rather than trivially return the only entry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..vcs.base import RepoTarget


@dataclass(frozen=True)
class EditStep:
    """One concrete edit the stub worker applies (and then re-tests)."""

    description: str
    kind: str  # "regex_replace" | "ensure_import"
    # regex_replace
    find: str = ""
    replace: str = ""
    # ensure_import
    module: str = ""
    symbol: str = ""


@dataclass(frozen=True)
class MigrationRule:
    id: str
    name: str
    summary: str
    guidance: str
    detect: str  # regex; a file matching this is a migration target
    tags: tuple[str, ...] = ()
    edit_steps: tuple[EditStep, ...] = ()  # stub-only; empty for retrieval-only rules

    @property
    def retrieval_text(self) -> str:
        return f"{self.name}. {self.summary} {' '.join(self.tags)}"


# --- the catalog -----------------------------------------------------------

RULES: dict[str, MigrationRule] = {
    "modernize-datetime": MigrationRule(
        id="modernize-datetime",
        name="Modernize deprecated datetime.utcnow()",
        summary=(
            "Replace the deprecated, naive datetime.utcnow() with the timezone-aware "
            "datetime.now(timezone.utc), and make sure `timezone` is imported."
        ),
        guidance=(
            "Python 3.12 deprecated datetime.datetime.utcnow(): it returns a NAIVE datetime "
            "(no tzinfo), which silently corrupts any timezone math. Migrate every call:\n"
            "  1. Replace `datetime.utcnow()` with `datetime.now(timezone.utc)`.\n"
            "  2. Ensure `timezone` is imported from the datetime module in each file you edit "
            "(e.g. `from datetime import datetime, timezone`). Missing this import is the most "
            "common way a naive migration breaks at runtime with NameError.\n"
            "Do not change behaviour beyond making the timestamps timezone-aware. Do not touch "
            "the tests. Run the tests after each edit and keep iterating until they pass."
        ),
        detect=r"datetime\.utcnow\(\)",
        tags=("python", "datetime", "deprecation", "timezone", "py312"),
        edit_steps=(
            EditStep(
                description="Replace datetime.utcnow() with datetime.now(timezone.utc)",
                kind="regex_replace",
                find=r"datetime\.utcnow\(\)",
                replace="datetime.now(timezone.utc)",
            ),
            EditStep(
                description="Ensure `timezone` is imported from datetime",
                kind="ensure_import",
                module="datetime",
                symbol="timezone",
            ),
        ),
    ),
    # --- retrieval-only rules (not exercised by the fixtures) --------------
    "requests-to-httpx": MigrationRule(
        id="requests-to-httpx",
        name="Migrate requests to httpx",
        summary="Swap the synchronous requests library for httpx to enable async and HTTP/2.",
        guidance="Replace `import requests` with `import httpx` and adapt the small API differences.",
        detect=r"import requests",
        tags=("python", "http", "requests", "httpx", "async"),
    ),
    "pkg-resources-to-importlib": MigrationRule(
        id="pkg-resources-to-importlib",
        name="Replace pkg_resources with importlib.metadata",
        summary="pkg_resources is deprecated and slow to import; use importlib.metadata instead.",
        guidance="Replace pkg_resources.get_distribution(...).version with importlib.metadata.version(...).",
        detect=r"import pkg_resources",
        tags=("python", "packaging", "pkg_resources", "importlib", "deprecation"),
    ),
    "unittest-assert-aliases": MigrationRule(
        id="unittest-assert-aliases",
        name="Fix deprecated unittest assert aliases",
        summary="assertEquals/assertRaisesRegexp were removed; use assertEqual/assertRaisesRegex.",
        guidance="Rename the deprecated camelCase assert aliases to their supported spellings.",
        detect=r"assertEquals\(",
        tags=("python", "testing", "unittest", "deprecation"),
    ),
}


def get_rule(rule_id: str) -> MigrationRule | None:
    return RULES.get(rule_id)


# --- fleet jobs ------------------------------------------------------------


@dataclass(frozen=True)
class MigrationJob:
    id: str
    title: str
    rule_id: str
    targets: tuple[RepoTarget, ...]     # the repositories to migrate (real git)
    description: str

    @property
    def repo_ids(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.targets)


_FLEET = ("billing-service", "auth-service", "analytics-service", "notifications-service")

# Built-in demo job: four fixture repos, migrated through REAL git (a local bare remote).
_STATIC_JOBS: dict[str, MigrationJob] = {
    "datetime-fleet": MigrationJob(
        id="datetime-fleet",
        title="Modernize datetime.utcnow() across the platform",
        rule_id="modernize-datetime",
        targets=tuple(RepoTarget(kind="fixture", ref=r, name=r) for r in _FLEET),
        description=(
            "Roll the datetime.utcnow() deprecation fix out to four services (real git worktrees "
            "cloned from a local remote). Some already import `timezone` (single-pass); others do "
            "not, forcing the worker to iterate: edit, run tests, see the NameError, add the import."
        ),
    ),
}


def _env_github_job() -> MigrationJob | None:
    """A real GitHub job assembled from env (needs GITHUB_TOKEN to actually run):

        ME_TARGET_REPOS="owner/repo-a,owner/repo-b"
        ME_TARGET_RULE=modernize-datetime          # a rule id from the rulebook
        ME_TEST_COMMAND="python -m pytest -q"       # optional; else discover test_*.py
    """
    raw = os.getenv("ME_TARGET_REPOS", "").strip()
    if not raw:
        return None
    rule_id = os.getenv("ME_TARGET_RULE", "modernize-datetime")
    base = os.getenv("ME_BASE_BRANCH", "main")
    tc = os.getenv("ME_TEST_COMMAND", "").strip()
    test_command = tuple(tc.split()) if tc else None
    targets = tuple(
        RepoTarget(
            kind="github", ref=r.strip(), name=r.strip().split("/")[-1],
            base_branch=base, test_command=test_command,
        )
        for r in raw.split(",")
        if r.strip()
    )
    return MigrationJob(
        id="github-fleet",
        title=f"Migrate {len(targets)} GitHub repo(s) - {rule_id}",
        rule_id=rule_id,
        targets=targets,
        description="Real GitHub repositories from ME_TARGET_REPOS (requires GITHUB_TOKEN).",
    )


def all_jobs() -> dict[str, MigrationJob]:
    jobs = dict(_STATIC_JOBS)
    gh = _env_github_job()
    if gh is not None:
        jobs[gh.id] = gh
    return jobs


def list_jobs() -> list[dict]:
    out = []
    for j in all_jobs().values():
        rule = RULES[j.rule_id]
        out.append(
            {
                "id": j.id,
                "title": j.title,
                "rule_id": j.rule_id,
                "rule_name": rule.name,
                "repo_ids": [t.name for t in j.targets],
                "repo_count": len(j.targets),
                "kind": j.targets[0].kind if j.targets else "fixture",
                "description": j.description,
            }
        )
    return out


def get_job(job_id: str) -> MigrationJob | None:
    return all_jobs().get(job_id)
