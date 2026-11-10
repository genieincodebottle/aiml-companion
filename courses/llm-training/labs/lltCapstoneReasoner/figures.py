"""Small dependency-free SVG line charts for the capstone page, drawn from run files (no matplotlib needed).

  from figures import line_chart
  svg = line_chart(series, x_label="k (samples per problem)", y_label="pass@k", log_x=True)
series: list of dict(name, points=[(x, y), ...], color, dash=False, band=[(x, lo, hi), ...] optional)
"""
import math

FONT = "Verdana, sans-serif"


def _fmt(v):
    return ("%d" % v) if abs(v - round(v)) < 1e-9 else ("%.2f" % v).rstrip("0").rstrip(".")


def line_chart(series, x_label, y_label, title=None, log_x=False, y_min=None, y_max=None, x_ticks=None, y_ticks=None,
               width=640, height=360, legend_cols=2, y_pct=False):
    L, R, T, B = 64, 16, 24 if not title else 44, 92
    W, H = width - L - R, height - T - B
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]] + [b[i] for s in series for b in s.get("band", []) for i in (1, 2)]
    tx = (lambda v: math.log2(v)) if log_x else (lambda v: v)
    x0, x1 = tx(min(xs)), tx(max(xs))
    y0 = min(ys) if y_min is None else y_min
    y1 = max(ys) if y_max is None else y_max
    if y1 == y0:
        y1 = y0 + 1
    X = lambda v: L + (tx(v) - x0) / (x1 - x0 or 1) * W   # noqa: E731
    Y = lambda v: T + H - (v - y0) / (y1 - y0) * H         # noqa: E731
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" role="img" aria-label="%s" font-family="%s">'
           % (width, height, (title or y_label).replace('"', "'"), FONT),
           '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height)]
    if title:
        out.append('<text x="%d" y="24" font-size="15" fill="#1f2933" font-weight="bold">%s</text>' % (L, title))
    yt = y_ticks or [y0 + (y1 - y0) * i / 4 for i in range(5)]
    for v in yt:
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#e4e7eb" stroke-width="1"/>' % (L, L + W, Y(v), Y(v)))
        lab = ("%d%%" % round(v * 100)) if y_pct else _fmt(v)
        out.append('<text x="%d" y="%.1f" font-size="12" fill="#52606d" text-anchor="end">%s</text>' % (L - 8, Y(v) + 4, lab))
    xt = x_ticks or sorted(set(xs))
    for v in xt:
        out.append('<text x="%.1f" y="%d" font-size="12" fill="#52606d" text-anchor="middle">%s</text>' % (X(v), T + H + 18, _fmt(v)))
    out.append('<line x1="%d" x2="%d" y1="%d" y2="%d" stroke="#9aa5b1" stroke-width="1"/>' % (L, L + W, T + H, T + H))
    out.append('<line x1="%d" x2="%d" y1="%d" y2="%d" stroke="#9aa5b1" stroke-width="1"/>' % (L, L, T, T + H))
    out.append('<text x="%d" y="%d" font-size="13" fill="#323f4b" text-anchor="middle">%s</text>' % (L + W / 2, T + H + 38, x_label))
    out.append('<text x="16" y="%d" font-size="13" fill="#323f4b" text-anchor="middle" transform="rotate(-90 16 %d)">%s</text>'
               % (T + H / 2, T + H / 2, y_label))
    for s in series:
        if s.get("band"):
            top = " ".join("%.1f,%.1f" % (X(x), Y(hi)) for x, lo, hi in s["band"])
            bot = " ".join("%.1f,%.1f" % (X(x), Y(lo)) for x, lo, hi in reversed(s["band"]))
            out.append('<polygon points="%s %s" fill="%s" fill-opacity="0.15" stroke="none"/>' % (top, bot, s["color"]))
    for s in series:
        pts = " ".join("%.1f,%.1f" % (X(x), Y(y)) for x, y in s["points"])
        d = s.get("dash")
        dash = (' stroke-dasharray="%s"' % (d if isinstance(d, str) else "6 4")) if d else ""
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.2"%s/>' % (pts, s["color"], dash))
        if s.get("markers", True) and len(s["points"]) <= 12:
            for x, y in s["points"]:
                out.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (X(x), Y(y), s["color"]))
    col_w = W / legend_cols
    for i, s in enumerate(series):
        lx = L + (i % legend_cols) * col_w
        ly = T + H + 58 + (i // legend_cols) * 18
        d = s.get("dash")
        dash = (' stroke-dasharray="%s"' % (d if isinstance(d, str) else "6 4")) if d else ""
        out.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="%s" stroke-width="2.2"%s/>' % (lx, lx + 22, ly - 4, ly - 4, s["color"], dash))
        out.append('<text x="%.1f" y="%d" font-size="12" fill="#323f4b">%s</text>' % (lx + 28, ly, s["name"]))
    out.append("</svg>")
    return "".join(out)
