"""The README diagrams, checked mechanically.

Nobody renders an SVG in code review. These assert the things that go wrong
silently: text running off the canvas, a transparent background that makes the
whole diagram vanish in GitHub's dark theme, and images referenced by the README
that were never generated.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
SVGS = sorted(IMAGES.glob("*.svg"))
NS = "{http://www.w3.org/2000/svg}"

# Sans-serif at the sizes used here averages a bit over half the font size per
# character. Deliberately generous - this exists to catch a label that overruns
# by 40%, not to typeset.
CHAR_WIDTH_RATIO = 0.56
MONO_RATIO = 0.62


def _root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def test_diagrams_exist():
    assert SVGS, "no diagrams generated; run python scripts/build_diagrams.py"


@pytest.mark.parametrize("path", SVGS, ids=lambda p: p.name)
class TestEachDiagram:
    def test_is_valid_xml_with_a_viewbox(self, path):
        root = _root(path)
        assert root.get("viewBox"), f"{path.name} has no viewBox"

    def test_has_an_opaque_background(self, path):
        """The rule to keep if you edit these.

        GitHub renders README images against a light or a dark page depending on
        the reader's theme. A transparent SVG with dark text is invisible to
        half the audience, and nothing in review would show it.
        """
        root = _root(path)
        width = float(root.get("width"))
        height = float(root.get("height"))
        covers = [
            r
            for r in root.iter(f"{NS}rect")
            if float(r.get("width", 0)) >= width
            and float(r.get("height", 0)) >= height
            and (r.get("fill") or "none") != "none"
        ]
        assert covers, f"{path.name} has no opaque full-canvas background rect"

    def test_is_labelled_for_screen_readers(self, path):
        root = _root(path)
        assert root.get("aria-label"), f"{path.name} has no aria-label"
        assert root.find(f"{NS}title") is not None, f"{path.name} has no <title>"

    def test_no_two_labels_overlap(self, path):
        """The test that was missing, and the bug it now catches.

        The lifecycle diagram drew its SLOW LOOP caption at y=34 - exactly the
        title's baseline - so the two printed on top of each other. Every other
        check passed: valid XML, opaque background, aria-label present, and all
        text comfortably inside the canvas. It was only wrong relative to
        something else, which is the one thing bounds checking cannot see.

        Boxes are estimated, so the tolerance is generous. This is here to catch
        a label landing on a heading, not to police kerning.
        """
        root = _root(path)
        boxes = []
        for node in root.iter(f"{NS}text"):
            content = (node.text or "").strip()
            if not content:
                continue
            size = float(node.get("font-size", 12))
            mono = "mono" in (node.get("font-family") or "")
            width = len(content) * size * (MONO_RATIO if mono else CHAR_WIDTH_RATIO)
            x = float(node.get("x", 0))
            y = float(node.get("y", 0))
            anchor = node.get("text-anchor", "start")
            left = x - width / 2 if anchor == "middle" else x
            # Cap height only: descenders rarely collide and counting them makes
            # every stacked line in a box look like an overlap.
            boxes.append((left, y - size * 0.75, left + width, y + size * 0.1, content))

        clashes = []
        for i in range(len(boxes)):
            ax1, ay1, ax2, ay2, atext = boxes[i]
            for j in range(i + 1, len(boxes)):
                bx1, by1, bx2, by2, btext = boxes[j]
                overlap_x = min(ax2, bx2) - max(ax1, bx1)
                overlap_y = min(ay2, by2) - max(ay1, by1)
                # Require a real intersection in BOTH axes before complaining.
                if overlap_x > 12 and overlap_y > 3:
                    clashes.append(f"{atext[:34]!r} x {btext[:34]!r}")
        assert not clashes, f"{path.name} has overlapping labels: {clashes[:4]}"

    def test_no_text_overflows_the_canvas(self, path):
        """Catches the failure mode that a generator makes easy: an edited label
        that is now wider than the box it sits in, or runs off the page."""
        root = _root(path)
        width = float(root.get("width"))
        offenders: list[str] = []

        for node in root.iter(f"{NS}text"):
            content = (node.text or "").strip()
            if not content:
                continue
            size = float(node.get("font-size", 12))
            mono = "mono" in (node.get("font-family") or "")
            estimated = len(content) * size * (MONO_RATIO if mono else CHAR_WIDTH_RATIO)
            x = float(node.get("x", 0))
            anchor = node.get("text-anchor", "start")

            left = x - estimated / 2 if anchor == "middle" else x
            right = left + estimated
            if left < -2 or right > width + 2:
                offenders.append(
                    f"{content[:44]!r} (~{estimated:.0f}px at x={x}, anchor={anchor})"
                )

        assert not offenders, f"{path.name} text outside the canvas: {offenders}"


def test_every_referenced_image_exists():
    """A broken image in a README is invisible to every test except this one."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"!\[[^\]]*\]\((docs/images/[^)]+)\)", readme))
    missing = [r for r in referenced if not (ROOT / r).exists()]
    assert not missing, f"README references images that do not exist: {missing}"


def test_generated_diagrams_are_reachable_from_the_readme():
    """An orphaned diagram is dead weight that still has to be maintained."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    orphans = [p.name for p in SVGS if p.name not in readme]
    assert not orphans, f"diagrams generated but never shown: {orphans}"
