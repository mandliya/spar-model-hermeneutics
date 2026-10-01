"""Two charts for gigi/RESULTS.md: does asking gemma "did you lie?" catch its
lies, and does that survive swapping the answer letters?

Reads the committed summaries (run the two analysis scripts first):
  answer_probability/results/pa_auroc.csv      (02_analyse_pa.py)
  ab_swap/results/ab_swap_summary.csv          (02_analyse_swap.py)

  selfreport_auroc.png   per category: AUROC of gemma's P(yes), every way of
                         asking (grey) and the median (blue), against the
                         paper's range for gemma
  selfreport_swap.png    per category: median AUROC with yes as option A vs
                         as option B, on the same 626 rows

Run (from the git repo root):
    uv run --with pandas --with matplotlib python gigi/07_charts_pa_swap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

G = Path(__file__).resolve().parent
CHARTS = G / "charts"
PA = pd.read_csv(G / "answer_probability/results/pa_auroc.csv")
SW = pd.read_csv(G / "ab_swap/results/ab_swap_summary.csv")

INK, MUTED, GRID = "#17140f", "#8a8579", "#e1e0d9"
ACCENT, SOFT, BAND = "#2a78d6", "#b9b5aa", "#eef4fb"
LABEL = {"alibi": "alibi", "mask_factual": "mask-factual",
         "sycophancy_b": "sycophancy B", "sycophancy_a": "sycophancy A",
         "capture_secret": "capture-the-secret", "unanswerable": "unanswerable",
         "ascii": "ascii-sandbagging"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})

med = PA.groupby("category").auroc.median().sort_values(ascending=False)
ORDER = list(med.index)                     # best at the top
Y = {c: len(ORDER) - 1 - i for i, c in enumerate(ORDER)}


def frame(ax, title, subtitle):
    ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.5, -0.95, "coin flip", color=MUTED, fontsize=9, ha="center")
    ax.text(0.02, -0.95, "← points the wrong way", color=MUTED, fontsize=9)
    ax.text(0.98, -0.95, "catches lies →", color=MUTED, fontsize=9, ha="right")
    ax.set_yticks([Y[c] for c in ORDER])
    ax.set_yticklabels([LABEL[c] for c in ORDER])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, len(ORDER) - 0.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("AUROC: how often a lie gets a higher P(yes) than an honest answer")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_title(f"{title}\n", loc="left", fontsize=13, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")


# 1. does self-report catch the lie? -----------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.axvspan(0.46, 0.80, color=BAND, zorder=0)
ax.text(0.63, len(ORDER) - 0.35, "paper's range for gemma (0.46–0.80)", color=ACCENT,
        fontsize=9, ha="center")
frame(ax, "Asking gemma \"did you lie?\" clearly works in only two categories",
      "Score: gemma's probability of answering \"yes, I lied\".   "
      "Grey: each of the 13 ways of asking.   Blue: their median.")
for c in ORDER:
    x = PA[PA.category == c].auroc
    ax.scatter(x, [Y[c]] * len(x), s=26, color=SOFT, edgecolor="white", lw=0.6, zorder=2)
    ax.scatter(med[c], Y[c], s=120, color=ACCENT, edgecolor="white", lw=1.5, zorder=3)
    ax.text(1.02, Y[c], f"{med[c]:.2f}", va="center", fontsize=10.5,
            fontweight="bold", transform=ax.get_yaxis_transform())
ax.text(1.02, len(ORDER) - 0.35, "median", fontsize=9, color=MUTED,
        transform=ax.get_yaxis_transform())
fig.tight_layout()
fig.savefig(CHARTS / "selfreport_auroc.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved selfreport_auroc.png")

# 2. does it survive swapping A and B? ----------------------------------------
s = SW.groupby("category").agg(a=("auroc_orig", "median"), b=("auroc_swap", "median"))
fig, ax = plt.subplots(figsize=(9.5, 5.2))
frame(ax, "Swap A and B: alibi and mask-factual hold, ascii-sandbagging collapses",
      "Same 626 rows asked twice: \"yes, I lied\" as option A, then as option B.   "
      "Median over the 11 A/B ways of asking.")
for c in ORDER:
    a, b = s.loc[c, "a"], s.loc[c, "b"]
    ax.annotate("", xy=(b, Y[c]), xytext=(a, Y[c]), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6,
                                shrinkA=7, shrinkB=7, mutation_scale=12))
    ax.scatter(a, Y[c], s=110, facecolor="white", edgecolor=SOFT, lw=2, zorder=3)
    ax.scatter(b, Y[c], s=120, color=ACCENT, edgecolor="white", lw=1.5, zorder=3)
    ax.text(1.02, Y[c], f"{a:.2f} → {b:.2f}", va="center", fontsize=10.5,
            fontweight="bold", transform=ax.get_yaxis_transform())
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec=SOFT, mew=2,
           label="yes = A (original)"),
    Line2D([], [], marker="o", ls="", ms=9, color=ACCENT, mec="white",
           label="yes = B (swapped)")],
    loc="upper left", bbox_to_anchor=(0.0, 0.97), frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(CHARTS / "selfreport_swap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved selfreport_swap.png")
