#!/usr/bin/env python
"""Generate the README diagrams into docs/images/.

    python scripts/build_diagrams.py

Written as a generator rather than hand-authored SVG for three reasons: the five
diagrams share one palette and one type scale and stay consistent; a wording
change is a one-line edit rather than hunting through XML; and the output is
reproducible, so a diff on a committed SVG means somebody actually changed the
picture.

Every diagram paints an OPAQUE background. GitHub renders README images against
either a light or a dark page depending on the reader's theme, and a transparent
SVG with dark text vanishes for half the audience. This is the one rule to keep
if you edit these.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
BG = "#fbfbfd"
INK = "#12141a"
MUTED = "#5b6270"
CARD = "#ffffff"
GREEN = "#0f9d58"
PURPLE = "#7c3aed"
RED = "#d93025"
BLUE = "#2563eb"
AMBER = "#b26a00"
RULE = "#c9ccd4"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def txt(x, y, s, size=12.5, fill=INK, anchor="middle", weight="400", mono=False):
    family = "'SF Mono', Menlo, Consolas, monospace" if mono else FONT
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'
    )


def box(x, y, w, h, stroke=INK, fill=CARD, rx=8, width=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{d}/>'
    )


def line(x1, y1, x2, y2, stroke=INK, width=1.4, arrow=None, dash=None):
    a = f' marker-end="url(#{arrow})"' if arrow else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{width}"{a}{d}/>'
    )


def path(d, stroke=INK, width=1.4, arrow=None, fill="none", dash=None):
    a = f' marker-end="url(#{arrow})"' if arrow else ""
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{a}{ds}/>'
    )


def marker(name, colour):
    """A fixed-size arrowhead.

    `markerUnits="userSpaceOnUse"` is the whole point. The default is
    `strokeWidth`, which multiplies the marker box by the line's stroke width -
    so a 1.8px line grew a 12.6px arrowhead while a 1.4px line got a 9.8px one,
    and the loop arrows came out as large mismatched triangles. With user-space
    units every arrowhead in every diagram is the same 9px, whatever the line.
    """
    return (
        f'<marker id="{name}" viewBox="0 0 9 9" refX="8" refY="4.5" '
        f'markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" '
        f'orient="auto-start-reverse">'
        f'<path d="M 0 0.6 L 9 4.5 L 0 8.4 z" fill="{colour}"/></marker>'
    )


def svg(name: str, title: str, w: int, h: int, body: list[str], markers: list[str]):
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{esc(title)}">'
        f"<title>{esc(title)}</title>"
        f'<defs>{"".join(markers)}</defs>'
        f'<rect width="{w}" height="{h}" fill="{BG}"/>'
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(head + "".join(body) + "</svg>", encoding="utf-8")
    return name


# ---------------------------------------------------------------------------
# 1. The four phases, and the two loops that close them
# ---------------------------------------------------------------------------
def lifecycle():
    """Vertical layout, top to bottom, with every band reserved before drawing.

    The first version put the SLOW LOOP arc and its label at `y0 - 56` and
    `y0 - 62`, which with `y0 = 96` landed on 40 and 34 - exactly the title and
    subtitle baselines. The label printed straight through the heading and the
    arc crossed both. Nothing errored, and the canvas-bounds test passed,
    because everything was comfortably inside the canvas. It was only wrong on
    top of something else.
    """
    W, H = 940, 430
    CX = W / 2

    TITLE_Y, SUB_Y = 30, 52          # heading band
    SLOW_LABEL_Y = 86                # slow-loop caption
    SLOW_APEX = 104                  # arc rises to about here
    BOX_Y, BOX_H = 146, 100          # the four phases
    FAST_DIP = 300                   # fast-loop arc dips to about here
    FAST_LABEL_Y = 330
    RULE_Y = 358
    FOOT1_Y, FOOT2_Y = 384, 406

    b = [
        txt(CX, TITLE_Y, "The lifecycle of an LLM judge", 17, INK, weight="600"),
        txt(CX, SUB_Y,
            "Not a benchmark score. A service that is built, tuned, deployed and kept aligned.",
            12.5, MUTED),
    ]

    phases = [
        ("I  BIRTH", "a labelled benchmark", "with human rationales", GREEN),
        ("II  TRAINING", "RART: the RUBRIC", "is the parameter", PURPLE),
        ("III  DEPLOYMENT", "gate + critic,", "DROP on failure", BLUE),
        ("IV  MONITORING", "a band pegged to", "human disagreement", AMBER),
    ]
    x0, bw, gap = 42, 196, 24
    centre = [x0 + i * (bw + gap) + bw / 2 for i in range(4)]

    for i, (name, l1, l2, colour) in enumerate(phases):
        x = x0 + i * (bw + gap)
        b.append(box(x, BOX_Y, bw, BOX_H, stroke=colour, width=1.8))
        b.append(txt(centre[i], BOX_Y + 30, name, 13.5, colour, weight="600"))
        b.append(txt(centre[i], BOX_Y + 56, l1, 12, INK))
        b.append(txt(centre[i], BOX_Y + 76, l2, 12, INK))
        if i < 3:
            b.append(line(x + bw + 5, BOX_Y + BOX_H / 2, x + bw + gap - 7,
                          BOX_Y + BOX_H / 2, stroke=INK, arrow="ar-ink"))

    # SLOW LOOP: IV -> I, above the boxes and below the heading band.
    b.append(txt(CX, SLOW_LABEL_Y,
                 "SLOW LOOP    each week's rated sample is appended to the benchmark",
                 12, GREEN, weight="600"))
    b.append(path(f"M {centre[3]} {BOX_Y - 7} C {centre[3]} {SLOW_APEX}, "
                  f"{centre[0]} {SLOW_APEX}, {centre[0]} {BOX_Y - 7}",
                  stroke=GREEN, arrow="ar-green", width=1.5, dash="5 4"))

    # FAST LOOP: IV -> II, below the boxes.
    b.append(path(f"M {centre[3]} {BOX_Y + BOX_H + 7} C {centre[3]} {FAST_DIP}, "
                  f"{centre[1]} {FAST_DIP}, {centre[1]} {BOX_Y + BOX_H + 7}",
                  stroke=RED, arrow="ar-red", width=1.5))
    b.append(txt(CX, FAST_LABEL_Y,
                 "FAST LOOP    drift detected -> re-tune, staged for a human",
                 12, RED, weight="600"))

    b.append(line(42, RULE_Y, W - 42, RULE_Y, stroke=RULE, width=1))
    b.append(txt(CX, FOOT1_Y,
                 "Phase IV is the one teams defer, and the only one that tells you the other three have stopped working.",
                 12.5, INK))
    b.append(txt(CX, FOOT2_Y,
                 "A judge aligned on the day it ships will not stay aligned: the catalogue moves, the generator moves, and so does the meaning of good.",
                 11.5, MUTED))
    return svg("01-lifecycle.svg", "The four phases of an LLM judge lifecycle", W, H, b,
               [marker("ar-ink", INK), marker("ar-red", RED), marker("ar-green", GREEN)])


# ---------------------------------------------------------------------------
# 2. One rubric, two readers
# ---------------------------------------------------------------------------
def two_readers():
    W, H = 940, 430
    b = [
        txt(W / 2, 34, "One rubric, two readers", 17, INK, weight="600"),
        txt(W / 2, 56, "The same file is prose for a model and rules for a deterministic engine.", 12.5, MUTED),
    ]
    rx, ry, rw, rh = 250, 82, 440, 104
    b.append(box(rx, ry, rw, rh, stroke=INK, width=1.8))
    b.append(txt(rx + rw / 2, ry + 24, "rubric  (domains/<name>/domain.yaml)", 12, MUTED))
    b.append(txt(rx + 16, ry + 50, "- Reject filler that would fit any item.", 12, INK, anchor="start", mono=True))
    b.append(txt(rx + 16, ry + 70, '  [banned: "you\'ll love it", "a must-watch"]', 12, PURPLE, anchor="start", mono=True))
    b.append(txt(rx + 16, ry + 90, "- Claims must be traceable. [grounded]", 12, INK, anchor="start", mono=True))

    b.append(line(rx + 110, ry + rh + 6, 210, 236, stroke=GREEN, arrow="ar-green"))
    b.append(line(rx + rw - 110, ry + rh + 6, 730, 236, stroke=PURPLE, arrow="ar-purple"))

    b.append(box(60, 244, 300, 108, stroke=GREEN, width=1.8))
    b.append(txt(210, 268, "LLM judge", 13.5, GREEN, weight="600"))
    b.append(txt(210, 292, "reads the sentences,", 12, INK))
    b.append(txt(210, 310, "ignores the brackets", 12, INK))
    b.append(txt(210, 334, "catches paraphrase - and costs money", 11.5, MUTED))

    b.append(box(580, 244, 300, 108, stroke=PURPLE, width=1.8))
    b.append(txt(730, 268, "rule engine  (src/rules.py)", 13.5, PURPLE, weight="600"))
    b.append(txt(730, 292, "reads the brackets,", 12, INK))
    b.append(txt(730, 310, "ignores the sentences", 12, INK))
    b.append(txt(730, 334, "free, exact, and cannot read", 11.5, MUTED))

    b.append(line(60, 376, W - 60, 376, stroke=RULE, width=1))
    b.append(txt(W / 2, 400,
                 "So the offline run is a real BASELINE, not a mock - and RART's optimiser edits text that both readers obey.",
                 12.5, INK))
    b.append(txt(W / 2, 420,
                 "Measured live: the Pro reflector added a prose clause AND a tag, improving both readers at once.",
                 11.5, MUTED))
    return svg("02-two-readers.svg", "One rubric read by both an LLM judge and a rule engine",
               W, H, b, [marker("ar-green", GREEN), marker("ar-purple", PURPLE)])


# ---------------------------------------------------------------------------
# 3. Gate and critic
# ---------------------------------------------------------------------------
def gate_critic():
    W, H = 940, 400
    b = [
        txt(W / 2, 34, "Phase III: one judge, two roles", 17, INK, weight="600"),
        txt(W / 2, 56, "It rejects (gate), and its reason becomes the writer's next instruction (critic).", 12.5, MUTED),
    ]
    y = 110
    b.append(box(48, y, 150, 70, stroke=INK))
    b.append(txt(123, y + 30, "record", 13, INK, weight="600"))
    b.append(txt(123, y + 50, "the subject", 11.5, MUTED))

    b.append(box(248, y, 150, 70, stroke=BLUE, width=1.8))
    b.append(txt(323, y + 30, "generator", 13, BLUE, weight="600"))
    b.append(txt(323, y + 50, "writes a draft", 11.5, MUTED))

    b.append(box(448, y, 168, 70, stroke=PURPLE, width=1.8))
    b.append(txt(532, y + 30, "judge", 13, PURPLE, weight="600"))
    b.append(txt(532, y + 50, "one per criterion", 11.5, MUTED))

    b.append(box(700, y - 34, 190, 62, stroke=GREEN, width=1.8))
    b.append(txt(795, y - 12, "SERVE", 13.5, GREEN, weight="600"))
    b.append(txt(795, y + 8, "passed every gate", 11.5, MUTED))

    b.append(box(700, y + 76, 190, 62, stroke=RED, width=1.8))
    b.append(txt(795, y + 98, "DROP", 13.5, RED, weight="600"))
    b.append(txt(795, y + 118, "budget exhausted", 11.5, MUTED))

    b.append(line(200, y + 35, 242, y + 35, arrow="ar-ink"))
    b.append(line(400, y + 35, 442, y + 35, arrow="ar-ink"))
    b.append(path(f"M 618 {y+22} C 660 {y+22}, 660 {y-8}, 696 {y-8}", stroke=GREEN, arrow="ar-green"))
    b.append(path(f"M 618 {y+50} C 660 {y+50}, 660 {y+104}, 696 {y+104}", stroke=RED, arrow="ar-red"))

    b.append(path(f"M 532 {y+74} C 532 {y+140}, 323 {y+140}, 323 {y+76}",
                  stroke=PURPLE, arrow="ar-purple", width=1.6))
    b.append(txt(428, y + 158, "the rejection REASON is the revision instruction   (K retries)",
                 12, PURPLE, weight="600"))

    b.append(line(48, 316, W - 48, 316, stroke=RULE, width=1))
    b.append(txt(W / 2, 342, "The asymmetry is one `if`, and it is the whole design:", 12.5, INK, weight="600"))
    b.append(txt(W / 2, 364, "a bad artefact served reaches a user and cannot be recalled   -   a good one dropped costs one opportunity", 12, INK))
    b.append(txt(W / 2, 386, "This is also why a right-verdict-wrong-reason rejection is a real defect: it steers the rewrite at the wrong problem.", 11.5, MUTED))
    return svg("03-gate-critic.svg", "The judge as gate and critic, with a bounded retry budget",
               W, H, b, [marker("ar-ink", INK), marker("ar-green", GREEN),
                         marker("ar-red", RED), marker("ar-purple", PURPLE)])


# ---------------------------------------------------------------------------
# 4. The floating band
# ---------------------------------------------------------------------------
def floating_band():
    W, H = 940, 430
    b = [
        txt(W / 2, 34, "Phase IV: the threshold floats", 17, INK, weight="600"),
        txt(W / 2, 56, "judge  >=  mean(raters)  -  2 x sd(raters)", 13, INK, mono=True),
        txt(W / 2, 78, "Same judge score in both panels. One passes, one fails - and that is correct.", 12.5, MUTED),
    ]

    def panel(x0, title, sub, sd_px, judge_y, verdict, colour):
        out = [box(x0, 104, 400, 220, stroke=RULE, width=1.2, fill=CARD)]
        out.append(txt(x0 + 200, 130, title, 13.5, INK, weight="600"))
        out.append(txt(x0 + 200, 150, sub, 11.5, MUTED))
        mean_y = 200
        # band
        out.append(f'<rect x="{x0+60}" y="{mean_y}" width="280" height="{sd_px}" '
                   f'fill="{colour}" opacity="0.10"/>')
        out.append(line(x0 + 60, mean_y, x0 + 340, mean_y, stroke=MUTED, width=1.4, dash="4 3"))
        out.append(txt(x0 + 352, mean_y + 4, "mean", 11, MUTED, anchor="start"))
        floor = mean_y + sd_px
        out.append(line(x0 + 60, floor, x0 + 340, floor, stroke=colour, width=1.8))
        out.append(txt(x0 + 352, floor + 4, "floor", 11, colour, anchor="start"))
        out.append(f'<circle cx="{x0+200}" cy="{judge_y}" r="6" fill="{INK}"/>')
        out.append(txt(x0 + 200, judge_y - 14, "judge", 11.5, INK, weight="600"))
        out.append(txt(x0 + 200, 306, verdict, 13, colour, weight="600"))
        return out

    b += panel(42, "Unanimous raters  (sd = 0)", "nobody found this week ambiguous",
               22, 238, "OUT OF BAND  -  alert", RED)
    b += panel(498, "Raters disagreed  (sd large)", "a genuinely hard week",
               76, 238, "IN BAND  -  no alert", GREEN)

    b.append(line(42, 348, W - 42, 348, stroke=RULE, width=1))
    b.append(txt(W / 2, 374,
                 "A hard week widens sd, which widens the band: the judge is not punished for finding hard what people also found hard.",
                 12.5, INK))
    b.append(txt(W / 2, 396,
                 "A FIXED threshold fails both ways - it fires every hard week until somebody mutes it, then sleeps through a slow slide on easy ones.",
                 12, MUTED))
    b.append(txt(W / 2, 418,
                 "The spread between raters is not noise to average away. It is the input.",
                 12, AMBER, weight="600"))
    return svg("04-floating-band.svg", "A drift band whose width follows rater disagreement",
               W, H, b, [])


# ---------------------------------------------------------------------------
# 5. Where drift hides
# ---------------------------------------------------------------------------
def drift_hides():
    W, H = 940, 400
    b = [
        txt(W / 2, 34, "Where drift hides", 17, INK, weight="600"),
        txt(W / 2, 56, "Week 6 of the shipped scenario: the aggregate is healthy and the judge is broken.", 12.5, MUTED),
    ]
    rows = [
        ("Established catalogue", "10 failures, judge caught 10", GREEN, "the judge is still good here"),
        ("Recently added titles", "4 failures, judge caught 1", RED, "new subject matter it was never tuned on"),
    ]
    y = 100
    for i, (name, stat, colour, note) in enumerate(rows):
        yy = y + i * 76
        b.append(box(60, yy, 500, 62, stroke=colour, width=1.8))
        b.append(txt(80, yy + 26, name, 13, colour, anchor="start", weight="600"))
        b.append(txt(80, yy + 46, stat, 12, INK, anchor="start"))
        b.append(txt(576, yy + 36, note, 11.5, MUTED, anchor="start"))

    b.append(box(60, 262, 400, 62, stroke=GREEN, width=1.8))
    b.append(txt(260, 286, "OVERALL   specificity 0.786  vs  floor 0.624", 12.5, INK, mono=True))
    b.append(txt(260, 308, "IN BAND  -  no alert", 12.5, GREEN, weight="600"))

    b.append(box(490, 262, 400, 62, stroke=RED, width=1.8))
    b.append(txt(690, 286, "NEW ITEMS  specificity 0.250  vs  floor 0.681", 12.5, INK, mono=True))
    b.append(txt(690, 308, "OUT OF BAND  -  alert", 12.5, RED, weight="600"))

    b.append(txt(W / 2, 356,
                 "Ten right verdicts outweigh three wrong ones, so a monitor watching only the aggregate files this as a normal week.",
                 12.5, INK))
    b.append(txt(W / 2, 380,
                 "check_new_items_separately: true   -   one line of config, and the only reason this is visible.",
                 12, PURPLE, weight="600"))
    return svg("05-drift-hides.svg", "Drift visible only when new items are checked separately",
               W, H, b, [])


if __name__ == "__main__":
    for fn in (lifecycle, two_readers, gate_critic, floating_band, drift_hides):
        print("wrote docs/images/" + fn())
