#!/usr/bin/env python
"""Verify every diagram is renderable: nothing clipped, nothing overlapping.

    python scripts/check_diagrams.py

Diagrams fail in two ways that an author never notices, because the author
looks at the picture they meant to draw rather than the one on the page:

  CLIPPING   an element sits outside the viewBox and is silently cut off
  OVERLAP    two boxes collide and the text of one covers the other

Both are checkable, so they are checked rather than eyeballed. This runs in the
test suite, so a change to the generator that pushes a label off the canvas
fails the build instead of shipping.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
NS = {"svg": "http://www.w3.org/2000/svg"}

# Approximate advance width per character for the sans stack used, as a
# fraction of font size. Deliberately generous: the point is to catch text that
# is obviously too wide for the canvas, not to reproduce a font engine.
CHAR_W = 0.60


def check(path: Path) -> list[str]:
    problems: list[str] = []
    root = ET.parse(path).getroot()
    vb = root.get("viewBox", "")
    try:
        _, _, vw, vh = (float(v) for v in vb.split())
    except ValueError:
        return [f"{path.name}: missing or malformed viewBox ({vb!r})"]

    # --- background must be painted -------------------------------------
    # An SVG with a transparent background and dark text becomes invisible on
    # GitHub in dark mode - a failure the author never sees, because they are
    # on the theme they designed for.
    first = root.find("svg:rect", NS)
    if first is None or first.get("fill", "").lower() in {"none", "transparent", ""}:
        problems.append(f"{path.name}: no opaque background rect")

    boxes: list[tuple[float, float, float, float]] = []
    for rect in root.iter(f"{{{NS['svg']}}}rect"):
        x, y = float(rect.get("x", 0)), float(rect.get("y", 0))
        w, h = float(rect.get("width", 0)), float(rect.get("height", 0))
        if w >= vw and h >= vh:
            continue                       # the background rect itself
        if x < -0.5 or y < -0.5 or x + w > vw + 0.5 or y + h > vh + 0.5:
            problems.append(
                f"{path.name}: rect at ({x:.0f},{y:.0f}) {w:.0f}x{h:.0f} "
                f"escapes the {vw:.0f}x{vh:.0f} canvas"
            )
        boxes.append((x, y, w, h))

    # --- text inside the canvas, both ends ------------------------------
    for text in root.iter(f"{{{NS['svg']}}}text"):
        x, y = float(text.get("x", 0)), float(text.get("y", 0))
        size = float(text.get("font-size", 12))
        content = "".join(text.itertext())
        width = len(content) * size * CHAR_W
        anchor = text.get("text-anchor", "start")
        left = x - width / 2 if anchor == "middle" else (x - width if anchor == "end" else x)
        right = left + width
        if y < 0 or y > vh:
            problems.append(f"{path.name}: text baseline y={y:.0f} outside 0..{vh:.0f}"
                            f"  {content[:40]!r}")
        if left < -2 or right > vw + 2:
            problems.append(
                f"{path.name}: text overflows horizontally "
                f"({left:.0f}..{right:.0f} vs 0..{vw:.0f})  {content[:48]!r}"
            )

    # --- sibling boxes must not collide ---------------------------------
    # Nested boxes (a panel containing a card) are legitimate; only PARTIAL
    # overlaps indicate a layout mistake.
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            ox = min(ax + aw, bx + bw) - max(ax, bx)
            oy = min(ay + ah, by + bh) - max(ay, by)
            if ox <= 1 or oy <= 1:
                continue                    # no overlap
            contained = (
                (ax >= bx - 1 and ay >= by - 1 and ax + aw <= bx + bw + 1 and ay + ah <= by + bh + 1)
                or (bx >= ax - 1 and by >= ay - 1 and bx + bw <= ax + aw + 1 and by + bh <= ay + ah + 1)
            )
            if not contained:
                problems.append(
                    f"{path.name}: boxes partially overlap - "
                    f"({ax:.0f},{ay:.0f},{aw:.0f}x{ah:.0f}) and "
                    f"({bx:.0f},{by:.0f},{bw:.0f}x{bh:.0f})"
                )
    return problems


def main() -> int:
    files = sorted(SVG_DIR.glob("*.svg"))
    if not files:
        print("No diagrams found. Run scripts/generate_diagrams.py first.")
        return 1
    all_problems: list[str] = []
    for path in files:
        problems = check(path)
        status = "ok" if not problems else f"{len(problems)} problem(s)"
        print(f"  {path.name:<28} {status}")
        all_problems.extend(problems)
    if all_problems:
        print("\nProblems:")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1
    print(f"\n{len(files)} diagrams checked, no clipping or overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
