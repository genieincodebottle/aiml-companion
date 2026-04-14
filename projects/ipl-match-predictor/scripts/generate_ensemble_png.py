"""
generate_ensemble_png.py
Premium light-mode ensemble chart PNG at 2x resolution for Medium.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
WHITE      = "#ffffff"
BG         = "#f8fafc"
INDIGO_TOP = "#6366f1"
INDIGO_BAR = "#4f46e5"
AMBER_TOP  = "#f59e0b"
AMBER_BAR  = "#d97706"
GREEN_PILL = "#15803d"
GREEN_BG   = "#f0fdf4"
GREEN_BD   = "#86efac"
RED        = "#ef4444"
RED_BG     = "#fef2f2"
AMBER_NOTE = "#92400e"
AMBER_NOTE_BG = "#fffbeb"
AMBER_NOTE_BD = "#fcd34d"
GRID       = "#f1f5f9"
AXIS       = "#cbd5e1"
LABEL      = "#475569"
MUTED      = "#94a3b8"
TITLE      = "#0f172a"
SUBTITLE   = "#64748b"

fig = plt.figure(figsize=(10, 6.1), dpi=200)
fig.patch.set_facecolor(WHITE)

# Main axes for the chart
ax = fig.add_axes([0.13, 0.20, 0.82, 0.58])
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Data ──────────────────────────────────────────────────────────────────────
models = ["Gradient\nBoost", "Logistic\nRegression", "Random\nForest", "XGBoost"]
probs  = [60.1, 59.0, 49.7, 55.2]
colors = [INDIGO_BAR, INDIGO_BAR, AMBER_BAR, INDIGO_BAR]
text_colors = [INDIGO_TOP, INDIGO_TOP, AMBER_TOP, INDIGO_TOP]

x = np.arange(len(models))
bar_width = 0.46

# ── Grid lines ────────────────────────────────────────────────────────────────
for y_val in [45, 50, 55, 60, 65]:
    color = RED if y_val == 50 else GRID
    lw    = 1.4 if y_val == 50 else 0.8
    ls    = (0, (6, 4)) if y_val == 50 else (0, (4, 3))
    ax.axhline(y_val, color=color, linewidth=lw, linestyle=ls, zorder=1)

# ── Bars ──────────────────────────────────────────────────────────────────────
bars = ax.bar(x, probs, width=bar_width, color=colors,
              zorder=3, linewidth=0,
              bottom=0)

# Rounded top illusion via a thin rect overlay
for bar, col in zip(bars, colors):
    bx = bar.get_x()
    by = bar.get_y() + bar.get_height() - 0.3
    bw = bar.get_width()
    ax.add_patch(FancyBboxPatch((bx, by), bw, 0.6,
                                boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor=col, edgecolor="none", zorder=4))

# ── Value labels above bars ───────────────────────────────────────────────────
for xi, (val, tc) in enumerate(zip(probs, text_colors)):
    ax.text(xi, val + 0.5, f"{val}%",
            ha="center", va="bottom",
            fontsize=14, fontweight="800",
            color=tc, zorder=5)

# ── 50% coin flip annotation ──────────────────────────────────────────────────
ax.text(3.72, 50.25, "← 50% coin flip",
        ha="right", va="bottom",
        fontsize=9, color=RED, fontweight="600")

# ── RF callout box ────────────────────────────────────────────────────────────
ax.annotate(
    "  ⚠  RF at 49.7% — near coin flip\n  Signals genuine match uncertainty  ",
    xy=(2, 49.7), xytext=(2.5, 58.5),
    fontsize=8.5, color=AMBER_NOTE,
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.5", facecolor=AMBER_NOTE_BG,
              edgecolor=AMBER_NOTE_BD, linewidth=1.2),
    arrowprops=dict(arrowstyle="-|>", color=AMBER_NOTE_BD,
                    lw=1.2, connectionstyle="arc3,rad=-0.2"),
    zorder=6,
)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(-0.6, 3.8)
ax.set_ylim(43, 68)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, color=LABEL, linespacing=1.5)
ax.tick_params(axis="x", length=0, pad=6)
ax.set_yticks([45, 50, 55, 60, 65])
ax.set_yticklabels([f"{v}%" for v in [45, 50, 55, 60, 65]],
                   fontsize=9.5, color=MUTED)
ax.tick_params(axis="y", length=0)
ax.set_axisbelow(True)

# ── Top accent bar ────────────────────────────────────────────────────────────
fig.add_axes([0, 0.985, 1, 0.015]).set_visible(False)
accent = fig.add_axes([0, 0.982, 1, 0.018])
accent.set_facecolor(INDIGO_TOP)
accent.axis("off")

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.935,
         "Model Ensemble — RR vs RCB Pre-Match Prediction",
         ha="center", fontsize=15, fontweight="700",
         color=TITLE)
fig.text(0.5, 0.897,
         "Each of 4 models votes independently  ·  Final = ensemble average  ·  Made before toss",
         ha="center", fontsize=10, color=SUBTITLE)

# ── Divider ───────────────────────────────────────────────────────────────────
div = fig.add_axes([0.05, 0.875, 0.9, 0.002])
div.set_facecolor(AXIS)
div.axis("off")

# ── Ensemble result pill ──────────────────────────────────────────────────────
pill_ax = fig.add_axes([0.22, 0.025, 0.56, 0.10])
pill_ax.set_facecolor(GREEN_BG)
for spine in pill_ax.spines.values():
    spine.set_edgecolor(GREEN_BD)
    spine.set_linewidth(1.5)
pill_ax.set_xticks([])
pill_ax.set_yticks([])
pill_ax.text(0.5, 0.68, "ENSEMBLE AVERAGE",
             ha="center", va="center",
             fontsize=8.5, color="#166534",
             fontweight="600", transform=pill_ax.transAxes)
pill_ax.text(0.5, 0.28,
             "60% RR Win Probability  —  MODERATE confidence",
             ha="center", va="center",
             fontsize=11.5, fontweight="800",
             color=GREEN_PILL, transform=pill_ax.transAxes)

# ── Source note ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.008,
         "Source: Live prediction from IPL Prediction Arena · April 10, 2026 · Pre-match",
         ha="center", fontsize=8.5, color=MUTED)

out = OUT_DIR / "blog_diagram1_ensemble_premium.png"
fig.savefig(out, dpi=200, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved: {out}")
