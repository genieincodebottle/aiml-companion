"""Diagram integrity, enforced in the test suite.

A diagram that is clipped or overlapping is a defect the author never sees,
because they look at the picture they meant to draw. These make it a build
failure instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from check_diagrams import check  # noqa: E402

IMAGES = ROOT / "docs" / "images"
EXPECTED = [
    "01-traditional-rag.svg", "02-graphrag.svg", "03-hybrid-graphrag.svg",
    "04-ingestion.svg", "05-query-flow.svg", "06-schema.svg",
    "07-multihop.svg", "08-production.svg",
]


@pytest.mark.parametrize("name", EXPECTED)
def test_diagram_exists(name):
    assert (IMAGES / name).exists(), (
        f"{name} is missing. Run: python scripts/generate_diagrams.py"
    )


@pytest.mark.parametrize("name", EXPECTED)
def test_diagram_has_no_clipping_or_overlap(name):
    problems = check(IMAGES / name)
    assert not problems, "\n".join(problems)


def test_readme_references_every_diagram():
    """A generated diagram nobody links to is dead weight, and a README
    reference to a file that does not exist is a broken image."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [name for name in EXPECTED if name not in readme]
    assert not missing, f"diagrams not referenced by the README: {missing}"
