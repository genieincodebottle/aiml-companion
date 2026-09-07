#!/usr/bin/env python
"""Verify every diagram is renderable: nothing clipped, nothing overlapping.

    python scripts/check_diagrams.py

Diagrams fail in two ways that an author never notices, because the author
looks at the picture they meant to draw rather than the one on the page:

  CLIPPING   an element sits outside the viewBox and is silently cut off
  OVERLAP    two boxes collide and the text of one covers the other
  CROSSING   an edge passes through a node it does not connect, which is the
             node-link equivalent of two boxes colliding

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

    # --- NODE-LINK CHECKS ------------------------------------------------
    # Added when figures 6 and 7 became real graphs. Until then this file only
    # looked at <rect>, so a circle could sit half off the canvas and the check
    # still printed "ok". A gate that cannot see the shape it is guarding is
    # worse than no gate, because it is trusted.
    circles: list[tuple[float, float, float]] = []
    for circ in root.iter(f"{{{NS['svg']}}}circle"):
        cx, cy = float(circ.get("cx", 0)), float(circ.get("cy", 0))
        r = float(circ.get("r", 0))
        if r <= 0:
            continue
        circles.append((cx, cy, r))
        if cx - r < -0.5 or cy - r < -0.5 or cx + r > vw + 0.5 or cy + r > vh + 0.5:
            problems.append(
                f"{path.name}: node at ({cx:.0f},{cy:.0f}) r={r:.0f} "
                f"escapes the {vw:.0f}x{vh:.0f} canvas"
            )

    # two nodes must not overlap
    for i, a in enumerate(circles):
        for b in circles[i + 1:]:
            gap = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            if gap < a[2] + b[2] - 1:
                problems.append(
                    f"{path.name}: nodes at ({a[0]:.0f},{a[1]:.0f}) and "
                    f"({b[0]:.0f},{b[1]:.0f}) overlap"
                )

    # an edge must not pass through a node it does not terminate on.
    # An endpoint is treated as belonging to a node when it sits on that node's
    # rim, which is exactly what Canvas.link() produces.
    if circles:
        for line in root.iter(f"{{{NS['svg']}}}line"):
            x1, y1 = float(line.get("x1", 0)), float(line.get("y1", 0))
            x2, y2 = float(line.get("x2", 0)), float(line.get("y2", 0))

            def owner(px: float, py: float):
                for c in circles:
                    d = ((px - c[0]) ** 2 + (py - c[1]) ** 2) ** 0.5
                    if abs(d - c[2]) < 12 or d < c[2]:
                        return c
                return None

            ends = {owner(x1, y1), owner(x2, y2)} - {None}
            if not ends:
                continue                    # a plain connector, not a graph edge
            for t in range(1, 40):
                px = x1 + (x2 - x1) * t / 40
                py = y1 + (y2 - y1) * t / 40
                for c in circles:
                    if c in ends:
                        continue
                    if ((px - c[0]) ** 2 + (py - c[1]) ** 2) ** 0.5 < c[2] - 1:
                        problems.append(
                            f"{path.name}: edge ({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f}) "
                            f"passes through the node at ({c[0]:.0f},{c[1]:.0f})"
                        )
                        break
                else:
                    continue
                break

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
    total_nodes = 0
    for path in files:
        problems = check(path)
        nodes = len(ET.parse(path).getroot().findall(f".//{{{NS['svg']}}}circle"))
        total_nodes += nodes
        status = "ok" if not problems else f"{len(problems)} problem(s)"
        extra = f"  ({nodes} nodes)" if nodes else ""
        print(f"  {path.name:<28} {status}{extra}")
        all_problems.extend(problems)

    # A scanner that silently measures nothing reports clean. Two of these
    # figures are node-link graphs, so a run that sees no circles at all means
    # the checks above stopped matching, not that the diagrams got better.
    if total_nodes < 10:
        all_problems.append(
            f"checker measured only {total_nodes} nodes across {len(files)} files; "
            "the node-link checks are not matching anything"
        )
    if all_problems:
        print("\nProblems:")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1
    print(f"\n{len(files)} diagrams checked, no clipping or overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
