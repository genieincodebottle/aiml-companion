#!/usr/bin/env python
"""Generate every diagram in docs/images/ as SVG.

    python scripts/generate_diagrams.py

WHY A GENERATOR RATHER THAN HAND-DRAWN SVG
==========================================
Because the two ways diagrams go wrong are clipping and overlap, and both come
from guessing coordinates. Here every box width is computed from the text it
contains and every canvas is sized from its contents, so a diagram cannot be
cut off and two boxes cannot collide - the layout is derived, not eyeballed.

It also makes the set consistent: one palette, one type scale, one arrow style
across all eight figures, and changing any of them is a one-line edit rather
than eight files of find-and-replace.

READABILITY ON GITHUB
=====================
Every figure paints an explicit light background and uses dark text. GitHub
renders README images against either a light or a dark page depending on the
reader's theme, and an SVG with transparent background plus dark text becomes
invisible in dark mode - a failure the author never sees, because they are
usually on the theme they designed for.
"""

from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

# --- palette ---------------------------------------------------------------
BG = "#fbfbfd"
INK = "#12141a"
MUTED = "#5b6270"
LINE = "#c9cedb"

BLUE = "#2563eb"      # text / vector path
GREEN = "#0f9d58"     # graph path
PURPLE = "#7c3aed"    # hybrid / derived
ORANGE = "#d97706"    # structured source
RED = "#d92d20"       # failure / attack
GREY = "#6b7280"

# Character width factors for Helvetica-ish metrics at a given font size.
# Used to size boxes from their text so nothing is ever clipped.
_W = 0.56
_BOLD_W = 0.60


def tw(text: str, size: float, bold: bool = False) -> float:
    return len(text) * size * (_BOLD_W if bold else _W)


class Canvas:
    def __init__(self, width: float, height: float, title: str = "") -> None:
        self.w = width
        self.h = height
        self.title = title
        self.parts: list[str] = []

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    # -- primitives --------------------------------------------------------
    def box(self, x: float, y: float, w: float, h: float, *, fill: str = "#fff",
            stroke: str = LINE, rx: float = 8, width: float = 1.4,
            dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'rx="{rx}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}"{d}/>')

    def text(self, x: float, y: float, content: str, *, size: float = 13,
             fill: str = INK, bold: bool = False, anchor: str = "middle",
             mono: bool = False, italic: bool = False) -> None:
        family = ("ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
                  if mono else
                  "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif")
        weight = ' font-weight="600"' if bold else ""
        style = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
                 f'font-size="{size}" fill="{fill}" text-anchor="{anchor}"'
                 f'{weight}{style}>{html.escape(content)}</text>')

    def lines(self, x: float, y: float, rows: list[str], *, size: float = 12,
              fill: str = MUTED, anchor: str = "middle", leading: float = 1.45,
              mono: bool = False) -> float:
        for i, row in enumerate(rows):
            self.text(x, y + i * size * leading, row, size=size, fill=fill,
                      anchor=anchor, mono=mono)
        return y + max(len(rows) - 1, 0) * size * leading

    # -- GRAPH PRIMITIVES --------------------------------------------------
    # A schema drawn as rectangles teaches the reader that a graph database is
    # a set of tables with arrows. Circles joined by labelled edges is what the
    # data actually looks like, and it is the picture they will meet again in
    # the Neo4j browser.

    def node(self, x: float, y: float, r: float, label: str, sub: str = "", *,
             stroke: str = LINE, fill: str = "#ffffff", width: float = 2.0) -> None:
        """A graph node. Text is centred on the circle, one or two lines."""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{width}"/>')
        rows = [label] + ([sub] if sub else [])
        first = y - (len(rows) - 1) * 8 + 4
        for i, row in enumerate(rows):
            self.text(x, first + i * 16, row,
                      size=12.5 if i == 0 else 10, bold=(i == 0),
                      fill=INK if i == 0 else MUTED)

    def link(self, a: tuple[float, float, float], b: tuple[float, float, float],
             label: str = "", *, stroke: str = GREY, width: float = 1.6,
             dash: str = "", label_size: float = 9.5,
             label_side: float = 1.0) -> None:
        """An edge between two nodes, trimmed to both circle boundaries.

        Trimming is the whole point. An edge drawn centre to centre puts its
        arrowhead INSIDE the target node, and the label lands on top of the
        line rather than beside it.
        """
        (x1, y1, r1), (x2, y2, r2) = a, b
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / length, dy / length
        sx, sy = x1 + ux * r1, y1 + uy * r1
        ex, ey = x2 - ux * (r2 + 9), y2 - uy * (r2 + 9)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{d} '
                 f'marker-end="url(#arrow-{stroke.lstrip("#")})"/>')
        self._marker(stroke)
        if label:
            # Off the line, on the perpendicular, so the label never sits on
            # top of the edge it names.
            off = 13 * label_side
            self.text((x1 + x2) / 2 - uy * off, (y1 + y2) / 2 + ux * off + 4,
                      label, size=label_size, fill=stroke)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, *,
              stroke: str = GREY, width: float = 1.6, dash: str = "",
              label: str = "", label_size: float = 11,
              label_dy: float = -7) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{d} '
                 f'marker-end="url(#arrow-{stroke.lstrip("#")})"/>')
        self._marker(stroke)
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label,
                      size=label_size, fill=stroke)

    def _marker(self, stroke: str) -> None:
        key = f"arrow-{stroke.lstrip('#')}"
        if key in getattr(self, "_markers", set()):
            return
        if not hasattr(self, "_markers"):
            self._markers: set[str] = set()
        self._markers.add(key)

    def render(self) -> str:
        markers = "".join(
            f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="#{key.split("arrow-")[1]}"/>'
            f'</marker>'
            for key in sorted(getattr(self, "_markers", set()))
        )
        title = ""
        if self.title:
            title = f'<title>{html.escape(self.title)}</title>'
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w:.0f} '
            f'{self.h:.0f}" width="{self.w:.0f}" height="{self.h:.0f}" '
            f'role="img" aria-label="{html.escape(self.title)}">'
            f'{title}<defs>{markers}</defs>'
            f'<rect width="{self.w:.0f}" height="{self.h:.0f}" fill="{BG}"/>'
            + "".join(self.parts) + "</svg>"
        )

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(self.render(), encoding="utf-8")
        print(f"  wrote {name}  ({self.w:.0f}x{self.h:.0f})")


def heading(c: Canvas, text: str, sub: str = "") -> float:
    c.text(c.w / 2, 32, text, size=17, bold=True)
    if sub:
        c.text(c.w / 2, 54, sub, size=12.5, fill=MUTED)
        return 78
    return 58


def stage(c: Canvas, x: float, y: float, w: float, title: str,
          rows: list[str], *, colour: str = GREY, h: float | None = None) -> float:
    """A labelled stage box. Height is derived from the row count, so a box can
    never be too short for its own content."""
    height = h if h is not None else 42 + len(rows) * 17
    c.box(x, y, w, height, fill="#ffffff", stroke=colour)
    c.text(x + w / 2, y + 23, title, size=13, bold=True, fill=colour)
    c.lines(x + w / 2, y + 42, rows, size=11.5)
    return height


# ===========================================================================
# 1. Traditional RAG
# ===========================================================================
def diagram_traditional_rag() -> None:
    c = Canvas(940, 400, "Traditional RAG: retrieval by similarity only")
    top = heading(c, "1. Traditional RAG",
                  "One hop: the question finds text that looks like the answer")

    y = top + 14
    xs = [40, 250, 460, 670]
    w = 200
    stage(c, xs[0], y, w, "Question", ["\"Which products are\"", "\"exposed to Kaohsiung?\""],
          colour=INK, h=96)
    stage(c, xs[1], y, w, "Embed", ["one vector,", "768 dimensions"], colour=BLUE, h=96)
    stage(c, xs[2], y, w, "Vector search", ["top-k nearest chunks", "by cosine similarity"],
          colour=BLUE, h=96)
    stage(c, xs[3], y, w, "LLM", ["answer from", "those k chunks"], colour=INK, h=96)

    for i in range(3):
        c.arrow(xs[i] + w, y + 48, xs[i + 1], y + 48, stroke=BLUE)

    fy = y + 128
    c.box(40, fy, 860, 96, fill="#fff5f5", stroke=RED, dash="5 4")
    c.text(60, fy + 26, "Where it breaks", size=13, bold=True, fill=RED, anchor="start")
    c.lines(
        60, fy + 48,
        [
            "Retrieval returns text SIMILAR to the question. But no document says which products are exposed to Kaohsiung -",
            "the incident bulletin explicitly declines to say. The answer is a JOIN across five relationships, and a join is not",
            "a similarity. Raising k does not help: the sentence you need does not exist in any chunk, at any rank.",
        ],
        size=11.5, fill=INK, anchor="start",
    )
    c.save("01-traditional-rag.svg")


# ===========================================================================
# 2. GraphRAG
# ===========================================================================
def diagram_graphrag() -> None:
    c = Canvas(940, 430, "GraphRAG: retrieval by traversal")
    top = heading(c, "2. GraphRAG",
                  "Many hops: the question finds entities, then walks relationships")

    y = top + 14
    xs = [40, 232, 424, 616, 808]
    w = 172
    labels = [
        ("Question", ["\"exposed to", "Kaohsiung?\""], INK),
        ("Entity linking", ["full-text match", "-> :Location"], GREEN),
        ("Traversal", ["Cypher, 1-3 hops", "over the graph"], GREEN),
        ("Derived facts", ["paths that exist", "in no document"], PURPLE),
        ("LLM", ["answer from", "facts + evidence"], INK),
    ]
    for i, (title, rows, colour) in enumerate(labels):
        width = w if i < 4 else 92
        stage(c, xs[i], y, width, title, rows, colour=colour, h=96)
    for i in range(4):
        end = xs[i + 1]
        c.arrow(xs[i] + w, y + 48, end, y + 48, stroke=GREEN)

    gy = y + 126
    c.box(40, gy, 860, 118, fill="#f4fbf6", stroke=GREEN)
    c.text(60, gy + 25, "The traversal the answer needs", size=13, bold=True,
           fill=GREEN, anchor="start")

    chain = [
        ("Kaohsiung", "Location"), ("Sentinel Optics", "Supplier"),
        ("SEN-IR12", "Component"), ("TX-9", "Product"),
    ]
    bx = 62
    by = gy + 46
    for i, (name, kind) in enumerate(chain):
        bw = max(tw(name, 12, True), tw(kind, 10)) + 26
        c.box(bx, by, bw, 40, fill="#ffffff", stroke=GREEN, rx=6)
        c.text(bx + bw / 2, by + 18, name, size=12, bold=True)
        c.text(bx + bw / 2, by + 32, kind, size=10, fill=MUTED)
        if i < len(chain) - 1:
            c.arrow(bx + bw + 3, by + 20, bx + bw + 40, by + 20, stroke=GREEN,
                    width=1.3)
        bx += bw + 43

    c.lines(
        62, gy + 104,
        ["This chain is the answer. It appears in no chunk, so no amount of similarity search can retrieve it."],
        size=11.5, fill=INK, anchor="start",
    )
    c.save("02-graphrag.svg")


# ===========================================================================
# 3. Hybrid
# ===========================================================================
def diagram_hybrid() -> None:
    c = Canvas(980, 470, "Hybrid GraphRAG: vector anchors into graph traversal")
    top = heading(c, "3. Hybrid Graph + Vector RAG",
                  "Vector search finds where to start; the graph finds what similarity cannot")

    y = top + 10
    c.box(36, y, 210, 96, fill="#ffffff", stroke=INK)
    c.text(141, y + 24, "Question", size=13, bold=True)
    c.lines(141, y + 46, ["natural language,", "no schema knowledge"], size=11.5)

    # Two arms
    ay = y + 130
    c.box(36, ay, 300, 106, fill="#f5f8ff", stroke=BLUE)
    c.text(186, ay + 24, "Text arm", size=13, bold=True, fill=BLUE)
    c.lines(186, ay + 46, ["vector search (semantic)", "+ BM25 (exact terms)",
                           "fused with RRF"], size=11.5)

    c.box(372, ay, 300, 106, fill="#f4fbf6", stroke=GREEN)
    c.text(522, ay + 24, "Graph arm", size=13, bold=True, fill=GREEN)
    c.lines(522, ay + 46, ["entity linking -> traversal", "type-driven templates",
                           "derived structural facts"], size=11.5)

    c.arrow(141, y + 96, 141, ay - 4, stroke=BLUE)
    c.arrow(180, y + 96, 470, ay - 4, stroke=GREEN)

    # The bridge
    by = ay + 130
    c.box(36, by, 636, 62, fill="#faf5ff", stroke=PURPLE)
    c.text(354, by + 24, "THE BRIDGE:  (:Chunk)-[:MENTIONS]->(:Entity)", size=12.5,
           bold=True, fill=PURPLE, mono=True)
    c.text(354, by + 46,
           "vector hits become graph anchors; traversal results become quotable chunks",
           size=11.5, fill=MUTED)
    c.arrow(186, ay + 106, 186, by - 4, stroke=PURPLE, width=1.4)
    c.arrow(522, ay + 106, 522, by - 4, stroke=PURPLE, width=1.4)

    # Output
    c.box(706, ay, 238, 106, fill="#ffffff", stroke=INK)
    c.text(825, ay + 24, "Grounded answer", size=13, bold=True)
    c.lines(825, ay + 46, ["derived facts first,", "then supporting text,",
                           "every claim cited"], size=11.5)
    c.arrow(672, ay + 53, 706, ay + 53, stroke=PURPLE)

    c.lines(
        490, by + 92,
        ["Each arm fails differently. Name matching fails when the question names nothing "
         "(\"which products are exposed to the typhoon\").",
         "Vector anchoring fails when similarity lands on the wrong document. Together they "
         "are far more robust than either alone."],
        size=11.5, fill=INK,
    )
    c.save("03-hybrid-graphrag.svg")


# ===========================================================================
# 4. Ingestion
# ===========================================================================
def diagram_ingestion() -> None:
    c = Canvas(980, 560, "Ingestion: documents and ERP exports into Neo4j")
    top = heading(c, "4. Ingestion: two sources, treated differently",
                  "Not everything belongs in an LLM extractor")

    y = top + 8
    # Structured lane
    c.box(36, y, 430, 176, fill="#fffaf2", stroke=ORANGE)
    c.text(251, y + 25, "STRUCTURED  (ERP / PLM export)", size=12.5, bold=True,
           fill=ORANGE)
    c.lines(251, y + 48, ["products.csv, components.csv, bom.csv,",
                          "suppliers.csv, sites.csv, supplies.csv"], size=11.5, mono=True)
    c.box(70, y + 84, 362, 72, fill="#ffffff", stroke=ORANGE, rx=6)
    c.text(251, y + 106, "Loaded verbatim. No LLM.", size=12.5, bold=True)
    c.lines(251, y + 126,
            ["provenance = 'erp',  confidence = 1.0",
             "Already structured, governed and correct."], size=11)

    # Unstructured lane
    c.box(514, y, 430, 176, fill="#f4fbf6", stroke=GREEN)
    c.text(729, y + 25, "UNSTRUCTURED  (documents)", size=12.5, bold=True, fill=GREEN)
    c.lines(729, y + 48, ["audit reports, supplier questionnaires,",
                          "incident bulletins, regulatory notices"], size=11.5)
    c.box(548, y + 84, 362, 72, fill="#ffffff", stroke=GREEN, rx=6)
    c.text(729, y + 106, "Guardrail -> chunk -> LLM extract", size=12.5, bold=True)
    c.lines(729, y + 126,
            ["provenance = 'llm',  confidence + evidence quote",
             "Tier-2/3 dependencies no ERP has."], size=11)

    # Resolver
    ry = y + 208
    c.box(230, ry, 520, 92, fill="#ffffff", stroke=PURPLE)
    c.text(490, ry + 24, "Entity resolution  (deterministic, no model)", size=13,
           bold=True, fill=PURPLE)
    c.lines(490, ry + 46,
            ["1. normalised exact match   2. alias lookup   3. guarded fuzzy match",
             "ERP names are registered FIRST, so they win every identity contest"],
            size=11.5)
    c.arrow(251, y + 176, 330, ry - 4, stroke=ORANGE)
    c.arrow(729, y + 176, 650, ry - 4, stroke=GREEN)

    # Neo4j
    ny = ry + 124
    c.box(230, ny, 520, 112, fill="#f7f9ff", stroke=BLUE)
    c.text(490, ny + 26, "Neo4j", size=14, bold=True, fill=BLUE)
    c.lines(490, ny + 50,
            ["constraints + range indexes  |  VECTOR index on (:Chunk).embedding",
             "FULLTEXT on (:Entity).name + aliases  |  FULLTEXT on (:Chunk).text",
             "one store for embeddings AND relationships - no cross-database join key"],
            size=11.5)
    c.arrow(490, ry + 92, 490, ny - 4, stroke=PURPLE)

    c.text(490, ny + 136,
           "Deciding which facts belong on which side of that line is the central design judgement.",
           size=11.5, fill=INK)
    c.save("04-ingestion.svg")


# ===========================================================================
# 5. Query flow
# ===========================================================================
def diagram_query_flow() -> None:
    c = Canvas(980, 470, "Query flow: question to grounded answer")
    top = heading(c, "5. Query flow", "Question -> entities -> traversal -> evidence -> answer")

    steps = [
        ("Question", ["guardrails:", "injection, length,", "rate limit"], INK),
        ("Link + embed", ["full-text entity match", "+ query embedding"], BLUE),
        ("Retrieve", ["vector + BM25", "-> anchor chunks"], BLUE),
        ("Bridge", [":MENTIONS ->", "anchor entities"], PURPLE),
        ("Traverse", ["1-3 hops", "+ typed templates"], GREEN),
        ("Assemble", ["derived facts FIRST,", "then text, capped"], PURPLE),
        ("Generate", ["grounded prompt", "+ citations"], INK),
        ("Validate", ["citations, entities,", "numbers"], RED),
    ]
    y = top + 20
    x = 30
    w = 108
    for i, (title, rows, colour) in enumerate(steps):
        c.box(x, y, w, 104, fill="#ffffff", stroke=colour)
        c.text(x + w / 2, y + 24, title, size=12, bold=True, fill=colour)
        c.lines(x + w / 2, y + 46, rows, size=10)
        if i < len(steps) - 1:
            c.arrow(x + w, y + 52, x + w + 10, y + 52, stroke=GREY, width=1.4)
        x += w + 10

    ey = y + 138
    c.box(30, ey, 920, 108, fill="#faf5ff", stroke=PURPLE)
    c.text(50, ey + 26, "What the model actually sees", size=13, bold=True,
           fill=PURPLE, anchor="start")
    c.lines(
        50, ey + 50,
        ["[graph: exposure::location:kaohsiung]  DERIVED FROM THE KNOWLEDGE GRAPH",
         "  - product=NW-500 | component=PCB-A7 | exposed_supplier=Formosa Substrate Materials | tier_depth=1",
         "[SUB-TIER-FORMOSA] Sub-Tier Dossier ... \"Formosa Substrate Materials operates a single laminate line ...\""],
        size=11, fill=INK, anchor="start", mono=True,
    )
    c.text(490, ey + 138,
           "Derived facts are placed FIRST: they are the shortest and highest-value part of the context.",
           size=11.5, fill=INK)
    c.save("05-query-flow.svg")


# ===========================================================================
# 6. Schema
# ===========================================================================
def diagram_schema() -> None:
    """The schema, drawn as a graph rather than as a table diagram."""
    c = Canvas(980, 600, "Neo4j graph schema, drawn as a graph")
    top = heading(c, "6. Neo4j schema",
                  "Circles are nodes, arrows are relationships, and both carry properties")

    # Positions are chosen so no edge passes through a node it does not
    # terminate on. scripts/check_diagrams.py enforces that, because a
    # node-link diagram whose edges cut through unrelated circles reads as
    # scribble and the author never notices.
    y0 = top + 42
    N = {
        "doc":   (110, y0 + 30, 46),
        "chunk": (110, y0 + 196, 48),
        "sup":   (360, y0 + 84, 54),
        "loc":   (360, y0 + 258, 50),
        "find":  (610, y0 + 258, 50),
        "comp":  (610, y0 + 84, 54),
        "prod":  (846, y0 + 84, 52),
    }

    c.node(*N["doc"], ":Document", "doc_id, title", stroke=BLUE, fill="#f5f8ff")
    c.node(*N["chunk"], ":Chunk", "text, vector", stroke=BLUE, fill="#f5f8ff")
    c.node(*N["sup"], ":Supplier", "name, aliases", stroke=GREEN, fill="#f4fbf6")
    c.node(*N["loc"], ":Location", "name", stroke=GREEN, fill="#f4fbf6")
    c.node(*N["find"], ":Finding", "status", stroke=RED, fill="#fff5f5")
    c.node(*N["comp"], ":Component", "name", stroke=GREEN, fill="#f4fbf6")
    c.node(*N["prod"], ":Product", "name", stroke=GREEN, fill="#f4fbf6")

    c.link(N["doc"], N["chunk"], ":PART_OF", stroke=BLUE)
    c.link(N["chunk"], N["sup"], ":MENTIONS", stroke=PURPLE, width=2.0)
    c.link(N["chunk"], N["loc"], ":MENTIONS", stroke=PURPLE, width=2.0)
    c.link(N["sup"], N["comp"], ":SUPPLIES", stroke=GREEN)
    c.link(N["prod"], N["comp"], ":CONTAINS", stroke=GREEN)
    c.link(N["sup"], N["loc"], ":LOCATED_IN", stroke=GREEN)
    c.link(N["find"], N["sup"], ":RAISED_AGAINST", stroke=RED)

    # The self loop is the tier-2 structure: a supplier depending on another
    # supplier is what the whole project exists to find.
    sx, sy, sr = N["sup"]
    c.add(f'<path d="M {sx - 30:.0f} {sy - sr + 8:.0f} '
          f'C {sx - 84:.0f} {sy - 116:.0f} {sx + 84:.0f} {sy - 116:.0f} '
          f'{sx + 24:.0f} {sy - sr + 4:.0f}" fill="none" stroke="{GREEN}" '
          f'stroke-width="1.8" marker-end="url(#arrow-{GREEN.lstrip("#")})"/>')
    c._marker(GREEN)
    c.text(sx, sy - 104, ":DEPENDS_ON", size=10.5, fill=GREEN, bold=True)
    c.text(sx, sy - 88, "tier 2 and deeper", size=9.5, fill=MUTED)

    fy = y0 + 330
    c.box(34, fy, 448, 92, fill="#faf5ff", stroke=PURPLE)
    c.text(54, fy + 24, "The bridge, in one relationship", size=12.5, bold=True,
           fill=PURPLE, anchor="start")
    c.text(54, fy + 48, "(:Chunk)-[:MENTIONS]->(:Entity)", size=11.5, mono=True,
           anchor="start")
    c.text(54, fy + 72, "Forwards it turns a vector hit into anchors. Backwards it "
                        "turns a traversal into quotable text.",
           size=10.5, fill=MUTED, anchor="start")

    c.box(498, fy, 448, 92, fill="#ffffff", stroke=LINE)
    c.text(518, fy + 24, "Every entity carries TWO labels: (:Entity:Supplier)",
           size=12.5, bold=True, anchor="start")
    c.lines(
        518, fy + 48,
        [":Entity  one constraint, one full-text index, one generic traversal.",
         ":Supplier  is an index the planner scans. A property filter is not."],
        size=10.5, fill=INK, anchor="start",
    )
    c.save("06-schema.svg")


# ===========================================================================
# 7. Multi-hop worked example
# ===========================================================================
def diagram_multihop() -> None:
    """The flagship traversal, drawn the way the database walks it."""
    c = Canvas(980, 620, "Multi-hop traversal: the answer that exists in no document")
    top = heading(c, "7. Multi-hop, worked",
                  "\"A typhoon closes Kaohsiung. Which finished products are exposed?\"")

    y0 = top + 40
    N = {
        "kao":      (96, y0 + 96, 52),
        "formosa":  (302, y0 + 20, 56),
        "kaigan":   (302, y0 + 176, 56),
        "meridian": (512, y0 + 20, 56),
        "pcb":      (700, y0 + 20, 48),
        "dsp":      (700, y0 + 176, 48),
        "nw500":    (884, y0 + 20, 46),
        "nw220":    (884, y0 + 130, 46),
    }

    c.node(*N["kao"], "Kaohsiung", ":Location", stroke=ORANGE, fill="#fff8ed")
    c.node(*N["formosa"], "Formosa", "tier 2", stroke=GREEN, fill="#f4fbf6")
    c.node(*N["kaigan"], "Kaigan", "tier 0", stroke=GREEN, fill="#f4fbf6")
    c.node(*N["meridian"], "Meridian", "tier 1", stroke=GREEN, fill="#f4fbf6")
    c.node(*N["pcb"], "PCB-A7", ":Component", stroke=GREY, fill="#ffffff")
    c.node(*N["dsp"], "DSP-3300", ":Component", stroke=GREY, fill="#ffffff")
    c.node(*N["nw500"], "NW-500", ":Product", stroke=PURPLE, fill="#faf5ff")
    c.node(*N["nw220"], "NW-220", ":Product", stroke=PURPLE, fill="#faf5ff")

    c.link(N["kao"], N["formosa"], "LOCATED_IN", stroke=GREEN)
    c.link(N["kao"], N["kaigan"], "LOCATED_IN", stroke=GREEN, label_side=-1.0)
    c.link(N["formosa"], N["meridian"], "DEPENDS_ON", stroke=GREEN, width=2.0)
    c.link(N["meridian"], N["pcb"], "SUPPLIES", stroke=GREY)
    c.link(N["kaigan"], N["dsp"], "SUPPLIES", stroke=GREY)
    c.link(N["pcb"], N["nw500"], "CONTAINS", stroke=PURPLE)
    c.link(N["pcb"], N["nw220"], "CONTAINS", stroke=PURPLE)
    # DSP-3300 goes into NW-500 only. Check data/structured/bom.csv before
    # changing this: an earlier draft pointed it at NW-220, which is a
    # different product and makes NW-500 look single-sourced.
    c.link(N["dsp"], N["nw500"], "CONTAINS", stroke=PURPLE, label_side=-1.0)

    dy = y0 + 254
    c.box(34, dy, 912, 96, fill="#fff5f5", stroke=RED, dash="5 4")
    c.text(54, dy + 25, "Why no document contains this", size=13, bold=True,
           fill=RED, anchor="start")
    c.lines(
        54, dy + 48,
        ["Each edge lives in a DIFFERENT source: the incident bulletin names only the city, the sub-tier dossier names only",
         "Formosa's customer, the supplier profile names only the laminate, and the bill of materials is a CSV. The chain is",
         "never written down anywhere, which is exactly what the 2026 mapping report says in its own words."],
        size=11.5, fill=INK, anchor="start",
    )

    my = dy + 118
    c.box(34, my, 912, 86, fill="#f4fbf6", stroke=GREEN)
    c.text(54, my + 24, "Measured on the golden set (12 questions x 5 strategies)",
           size=12.5, bold=True, fill=GREEN, anchor="start")
    c.lines(
        54, my + 46,
        ["multi-hop questions - evidence recall:        hybrid 1.00   classic RAG 1.00   vector-only 0.75",
         "multi-hop questions - answer term coverage:   hybrid 1.00   classic RAG 0.47   vector-only 0.40"],
        size=11.5, fill=INK, anchor="start", mono=True,
    )
    c.save("07-multihop.svg")


# ===========================================================================
# 8. Production architecture
# ===========================================================================
def diagram_production() -> None:
    c = Canvas(980, 600, "Production architecture")
    top = heading(c, "8. Production architecture",
                  "Solid = built in this project.  Dashed = what production adds.")

    y = top + 10
    # Clients
    c.box(36, y, 200, 74, fill="#ffffff", stroke=INK)
    c.text(136, y + 26, "Clients", size=13, bold=True)
    c.lines(136, y + 46, ["Streamlit UI (thin)", "CLI, notebook, curl"], size=11)

    # Edge
    c.box(276, y, 200, 74, fill="#ffffff", stroke=RED, dash="5 4")
    c.text(376, y + 26, "Edge", size=13, bold=True, fill=RED)
    c.lines(376, y + 46, ["OIDC / mTLS, WAF", "distributed rate limit"], size=11)
    c.arrow(236, y + 37, 276, y + 37, stroke=GREY)

    # API
    ay = y + 106
    c.box(276, ay, 420, 118, fill="#f7f9ff", stroke=BLUE)
    c.text(486, ay + 26, "FastAPI  -  all business logic", size=13.5, bold=True, fill=BLUE)
    c.lines(486, ay + 48,
            ["guardrails (ingest / query / response)  |  retrieval strategies",
             "prompt construction  |  evaluation  |  audit log",
             "a control enforced in a frontend is one anyone can skip with curl"],
            size=11)
    c.arrow(376, y + 74, 376, ay - 4, stroke=RED, dash="5 4")

    # Stores
    sy = ay + 150
    c.box(36, sy, 280, 108, fill="#f4fbf6", stroke=GREEN)
    c.text(176, sy + 26, "Neo4j", size=13.5, bold=True, fill=GREEN)
    c.lines(176, sy + 48, ["graph + vector index", "+ full-text indexes",
                           "one store, no join key"], size=11)

    c.box(348, sy, 280, 108, fill="#ffffff", stroke=RED, dash="5 4")
    c.text(488, sy + 26, "Redis", size=13.5, bold=True, fill=RED)
    c.lines(488, sy + 48, ["shared rate limits", "answer + embedding cache",
                           "job queue"], size=11)

    c.box(660, sy, 284, 108, fill="#ffffff", stroke=RED, dash="5 4")
    c.text(802, sy + 26, "Observability", size=13.5, bold=True, fill=RED)
    c.lines(802, sy + 48, ["OpenTelemetry traces", "cost + latency per stage",
                           "retrieval quality alerts"], size=11)

    c.arrow(400, ay + 118, 220, sy - 4, stroke=GREEN)
    c.arrow(486, ay + 118, 486, sy - 4, stroke=RED, dash="5 4")
    c.arrow(600, ay + 118, 780, sy - 4, stroke=RED, dash="5 4")

    # Providers
    c.box(716, ay, 228, 118, fill="#ffffff", stroke=INK)
    c.text(830, ay + 26, "Model provider", size=13, bold=True)
    c.lines(830, ay + 48, ["extraction (thinking off)", "embeddings (cached)",
                           "generation", "per-request budget cap"], size=11)
    c.arrow(696, ay + 59, 716, ay + 59, stroke=GREY)

    # Ingest pipeline
    iy = sy + 140
    c.box(36, iy, 908, 96, fill="#fffaf2", stroke=ORANGE)
    c.text(56, iy + 26, "Ingestion, scheduled rather than on demand", size=13,
           bold=True, fill=ORANGE, anchor="start")
    c.lines(
        56, iy + 50,
        ["ERP/PLM change-data-capture -> backbone upsert   |   document queue -> guardrail -> extract -> resolve -> upsert",
         "incremental by document hash  |  human review queue for rejected fuzzy matches and flagged documents",
         "re-embedding is a migration: a new model means a new index, built alongside and swapped, never in place"],
        size=11.5, fill=INK, anchor="start",
    )
    c.save("08-production.svg")


def main() -> int:
    print("Generating diagrams into docs/images/")
    diagram_traditional_rag()
    diagram_graphrag()
    diagram_hybrid()
    diagram_ingestion()
    diagram_query_flow()
    diagram_schema()
    diagram_multihop()
    diagram_production()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
