"""Prose gates: no em dashes, no AI-slop tells.

Every other rule this project documents has a test behind it, and the writing
rules should not be the exception. An audit catches what exists today; a gate
catches what somebody adds next month.

Scope is the text a reader actually meets: markdown, the notebook's prose cells,
config comments, and the docstrings and comments in `src/`. Generated artefacts
under `artifacts/` are excluded because they are outputs, not writing.

The banned lists come from `AI_SLOP_NARRATION_AUDIT.md`. Where that document
calls for judgement rather than a rule - repeated structure, forced parallelism,
definitions the reader did not need - no test is attempted, because a regex that
tried would fail the good cases and teach people to silence it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"__pycache__", ".pytest_cache", "artifacts", ".venv", ".git"}
SUFFIXES = {".md", ".py", ".yaml", ".ipynb"}


def _files() -> list[Path]:
    # This file is excluded from its own scan: it is where the banned
    # vocabularies are written down, so every list below is a hit against
    # itself. Excluding the gate from the gate is correct; excluding anything
    # else needs a reason written next to it.
    return sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix in SUFFIXES
        and not SKIP_DIRS & set(p.parts)
        and p.name != Path(__file__).name
    )


def _text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix != ".ipynb":
        return raw
    # Prose and code the reader sees, not the execution outputs.
    return "\n".join("".join(c["source"]) for c in json.loads(raw)["cells"])


FILES = _files()

# The Tier-1 list from the audit doc, minus words this project uses in their
# ordinary technical sense. Each exemption is a judgement, so each is justified:
#
#   harness   - "evaluation harness" is the standard term for a test rig. The
#               slop use is the verb ("harness the power of"), which is caught
#               by the phrase list below instead.
#   robust    - not used, but kept out of the exemptions so it stays banned.
#
# Keep this list honest. Exempting a word because a sentence you like trips the
# gate is how a gate becomes decoration.
TIER1 = """delve comprehensive crucial pivotal nuanced multifaceted foster leverage
utilize bespoke paramount meticulous intricate holistic illuminate elevate encompass
hitherto streamline groundbreaking beacon realm commendable cognizant indelible
testament underscore spearhead cornerstone underpinning unwavering unparalleled
endeavor myriad plethora bustling vibrant reimagine cultivate galvanize elucidate
delineate juxtapose ascertain formidable daunting indispensable instrumental
invaluable robust""".split()

SLOP_PHRASES = [
    "it's worth noting", "it is worth noting", "needless to say",
    "here's the kicker", "at the end of the day", "in today's",
    "ever-evolving", "buckle up", "rest assured", "whether you're",
    "harness the power", "unlock the power", "unlock the potential",
    "take it to the next level", "game-changer", "cutting-edge",
    "state-of-the-art", "best-in-class", "seamlessly", "supercharge",
    "let's dive in", "let's delve", "deep dive into",
]

FILLER_ADVERBS = [
    "Moreover,", "Furthermore,", "Nevertheless,", "Notwithstanding,",
    "Additionally,", "Consequently,", "Subsequently,",
]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_or_en_dashes(path):
    """Project rule: simple dash or comma, never an em dash.

    Also catches the en dash, horizontal bar and Unicode minus, which arrive
    together when text is pasted from a word processor or a model that likes
    typography. Every formula in this project is written in ASCII.
    """
    offenders = [
        f"line {i}: {line.strip()[:70]}"
        for i, line in enumerate(_text(path).splitlines(), 1)
        if any(ch in line for ch in "—–―−")
    ]
    assert not offenders, f"{path.name} contains em/en dashes: {offenders[:5]}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_tier_one_ai_words(path):
    """Words that signal generated text on sight (audit doc [W4])."""
    text = _text(path).lower()
    hits = sorted({w for w in TIER1 if re.search(rf"\b{w}\b", text)})
    assert not hits, f"{path.name} uses Tier-1 AI words: {hits}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_slop_phrases(path):
    """Stock phrases with no information in them (audit doc [W3], [T3], [T11])."""
    text = _text(path).lower()
    hits = sorted({p for p in SLOP_PHRASES if p in text})
    assert not hits, f"{path.name} uses stock phrases: {hits}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_filler_transition_adverbs(path):
    """Formal transitions used as paragraph starters (audit doc [W5]).

    One is fine in isolation; the audit flags them because they cluster. The
    gate is stricter than the guidance on purpose - there is always a plainer
    way to open the sentence.
    """
    lines = _text(path).splitlines()
    hits = [
        f"line {i}: {a}"
        for i, line in enumerate(lines, 1)
        for a in FILLER_ADVERBS
        if line.lstrip().startswith(a)
    ]
    assert not hits, f"{path.name} opens with filler adverbs: {hits[:5]}"


def test_the_gate_covers_the_files_a_reader_actually_reads():
    """Guards the guard.

    If the discovery glob silently stopped matching, every test above would
    pass by inspecting nothing - the most comfortable kind of green.
    """
    names = {p.name for p in FILES}
    for expected in ("README.md", "QUICKSTART.md", "results.md", "prompts.py"):
        assert expected in names, f"{expected} is not being scanned"
    assert len(FILES) > 50, f"only {len(FILES)} files discovered; the glob is wrong"
