"""
generate_blog_diagrams.py
Generates 4 publication-ready PNG diagrams for the RR vs RCB Medium blog post.
All data is from real model outputs (evaluation_report.md, key_findings.md, prediction screenshot).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
BG       = "#0f172a"
CARD     = "#1e293b"
INDIGO   = "#6366f1"
INDIGO_L = "#818cf8"
GREEN    = "#34d399"
AMBER    = "#f59e0b"
RED      = "#ef4444"
TEXT     = "#f1f5f9"
MUTED    = "#94a3b8"
BORDER   = "#334155"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "text.color":       TEXT,
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "axes.edgecolor":   BORDER,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────────────────────────────────────
# Diagram 1 – Ensemble Model Votes
# ─────────────────────────────────────────────────────────────────────────────
def diagram_ensemble():
    models = ["Gradient\nBoost", "Logistic\nRegression", "Random\nForest", "XGBoost"]
    probs  = [60.1, 59.0, 49.7, 55.2]
    colors = [INDIGO, INDIGO, AMBER, INDIGO]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.bar(models, probs, color=colors, width=0.5, zorder=3)

    # 50% reference line
    ax.axhline(50, color=RED, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(3.45, 50.6, "50% (coin flip)", color=RED, fontsize=9, va="bottom", ha="right")

    # Ensemble average line
    ax.axhline(60, color=GREEN, linewidth=1.5, linestyle="-", zorder=2, alpha=0.7)
    ax.text(3.45, 60.6, "Ensemble avg → 60%", color=GREEN, fontsize=9, va="bottom", ha="right")

    # Value labels on bars
    for bar, val in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                f"{val}%", ha="center", va="bottom",
                color=TEXT, fontsize=11, fontweight="bold")

    ax.set_ylim(40, 68)
    ax.set_ylabel("RR Win Probability (%)", color=MUTED, fontsize=10)
    ax.set_title("Model Ensemble — RR vs RCB Pre-Match Prediction\n"
                 "Each model votes independently. Final result = weighted average.",
                 color=TEXT, fontsize=12, fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelsize=10)
    ax.yaxis.set_tick_params(labelcolor=MUTED)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.6)

    # Annotation for RF disagreement
    ax.annotate("RF disagrees —\ngenuine uncertainty",
                xy=(2, 49.7), xytext=(2.4, 53),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2),
                color=AMBER, fontsize=8.5)

    fig.tight_layout()
    out = OUT_DIR / "blog_diagram1_ensemble.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagram 2 – Feature Importance (real data from evaluation_report.md)
# ─────────────────────────────────────────────────────────────────────────────
def diagram_feature_importance():
    features = [
        "Elo Difference",
        "RR Elo Rating",
        "RCB Elo Rating",
        "Elo Win Probability",
        "Elo × Momentum (RCB)",
        "Elo × Momentum (RR)",
        "Head-to-Head Record",
        "Momentum Difference",
        "RR Recent Form",
        "Home Advantage",
    ]
    importances = [0.1143, 0.1126, 0.1125, 0.1022, 0.1017,
                   0.0962, 0.0808, 0.0376, 0.0335, 0.0300]
    colors = [INDIGO, INDIGO, INDIGO, INDIGO, INDIGO_L,
              INDIGO_L, GREEN, GREEN, GREEN, AMBER]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y = np.arange(len(features))
    bars = ax.barh(y, importances, color=colors, height=0.6, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()

    for bar, val in zip(bars, importances):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=9, color=TEXT)

    ax.set_xlabel("Feature Importance Score", color=MUTED, fontsize=10)
    ax.set_title("What Actually Drives Predictions\n"
                 "Feature importances from trained Random Forest (1,095 matches, 2008–2024)",
                 color=TEXT, fontsize=12, fontweight="bold", pad=14)
    ax.xaxis.grid(True, color=BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 0.145)

    legend = [
        mpatches.Patch(color=INDIGO,   label="Elo-based features"),
        mpatches.Patch(color=GREEN,    label="Form / matchup"),
        mpatches.Patch(color=AMBER,    label="Venue / home"),
    ]
    ax.legend(handles=legend, loc="lower right",
              facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "blog_diagram2_feature_importance.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagram 3 – Toss Hypothesis Test (real p-value = 0.61)
# ─────────────────────────────────────────────────────────────────────────────
def diagram_toss_hypothesis():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # p-value bar visualization
    ax2 = fig.add_axes([0.1, 0.30, 0.8, 0.26])
    ax2.set_facecolor(CARD)

    # significance zone (0 to 0.05) in red
    ax2.barh(0, 0.05,  color=RED,   height=0.5, zorder=3)
    # non-significant zone (0.05 to 1.0)
    ax2.barh(0, 0.95, left=0.05, color="#1e3a5f", height=0.5, zorder=3)

    # p-value marker
    ax2.axvline(0.61, color=GREEN, linewidth=2.5, zorder=4)
    ax2.text(0.61, 0.42, "p = 0.61  (our result)", color=GREEN,
             fontsize=10, fontweight="bold", ha="center", va="bottom")
    ax2.text(0.025, -0.42, "Significant\n(p < 0.05)", color=RED,
             fontsize=8, ha="center", va="top")
    ax2.text(0.525, -0.42, "Not significant — toss has no real effect",
             color=MUTED, fontsize=8, ha="center", va="top")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.6, 0.8)
    ax2.set_xlabel("p-value", color=MUTED, fontsize=9)
    ax2.set_facecolor(CARD)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(BORDER)

    ax.text(0.5, 0.97, "Does Winning the Toss Actually Help?",
            transform=ax.transAxes, ha="center", va="top",
            color=TEXT, fontsize=13, fontweight="bold")
    ax.text(0.5, 0.88, "Binomial hypothesis test  |  H₀: toss winner wins 50% of matches  |  1,095 IPL matches (2008–2024)",
            transform=ax.transAxes, ha="center", va="top",
            color=MUTED, fontsize=9)
    ax.text(0.5, 0.13,
            "Verdict: p = 0.61  >>  threshold of 0.05  →  Toss has no statistically significant effect on match outcome.",
            transform=ax.transAxes, ha="center", va="bottom",
            color=GREEN, fontsize=9.5,
            bbox=dict(facecolor=CARD, edgecolor=BORDER, boxstyle="round,pad=0.5"))

    out = OUT_DIR / "blog_diagram3_toss_hypothesis.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagram 4 – Model Accuracy in Context
# ─────────────────────────────────────────────────────────────────────────────
def diagram_accuracy_context():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.patch.set_facecolor(BG)

    data = [
        ("Random\nGuess",   50.0, RED,    "Coin flip baseline.\nNo features used."),
        ("This Model\n(Ensemble)", 53.1, INDIGO, "Calibrated 4-model ensemble.\n1,095 matches training data."),
        ("T20\nVariance",   None, AMBER,  "1 over can flip any match.\nHigh noise domain."),
    ]

    for ax, (label, val, color, note) in zip(axes, data):
        ax.set_facecolor(CARD)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # border highlight for our model
        if "Model" in label:
            for spine in ["bottom", "top", "left", "right"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_edgecolor(INDIGO)
                ax.spines[spine].set_linewidth(2)

        ax.text(0.5, 0.82, label, ha="center", va="center",
                color=MUTED, fontsize=11, transform=ax.transAxes)

        if val is not None:
            ax.text(0.5, 0.52, f"{val}%", ha="center", va="center",
                    color=color, fontsize=30, fontweight="bold", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.52, "High", ha="center", va="center",
                    color=color, fontsize=30, fontweight="bold", transform=ax.transAxes)

        ax.text(0.5, 0.2, note, ha="center", va="center",
                color=MUTED, fontsize=8.5, transform=ax.transAxes, linespacing=1.5)

    fig.suptitle(
        "53.1% Accuracy — Why That's Honest, Not Disappointing\n"
        "+3.1% above random across 1,095 matches = real signal in a high-variance sport",
        color=TEXT, fontsize=12, fontweight="bold", y=1.02
    )

    fig.tight_layout(pad=1.5)
    out = OUT_DIR / "blog_diagram4_accuracy_context.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating blog diagrams...")
    diagram_ensemble()
    diagram_feature_importance()
    diagram_toss_hypothesis()
    diagram_accuracy_context()
    print("\nAll 4 diagrams saved to artifacts/figures/")
