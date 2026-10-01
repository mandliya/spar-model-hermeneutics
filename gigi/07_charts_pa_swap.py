"""Self-report charts for gigi/RESULTS.md, for the four follow-up questions the
project keeps: neutral, neutral_cot, named, named_cot.

Reads the committed summaries (run the two analysis scripts first):
  answer_probability/results/pa_auroc.csv      (02_analyse_pa.py)
  ab_swap/results/ab_swap_summary.csv          (02_analyse_swap.py)

  selfreport_by_question.png  per category and question: balanced accuracy
                              from the letter gemma wrote (left) and AUROC
                              from its probability of "yes" (right)
  selfreport_swap.png         per category: median AUROC over the four
                              questions with yes as option A vs as option B,
                              on the same 626 rows

Run (from the git repo root):
    uv run --with pandas --with matplotlib python gigi/07_charts_pa_swap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

G = Path(__file__).resolve().parent
CHARTS = G / "charts"
KEEP = ["neutral", "neutral_cot", "named", "named_cot"]
PA = pd.read_csv(G / "answer_probability/results/pa_auroc.csv")
PA = PA[PA.condition.isin(KEEP)]
SW = pd.read_csv(G / "ab_swap/results/ab_swap_summary.csv")
SW = SW[SW.condition.isin(KEEP)]

INK, MUTED, GRID = "#17140f", "#8a8579", "#e1e0d9"
ACCENT, SOFT, BAND = "#2a78d6", "#b9b5aa", "#eef4fb"
ORANGE = "#eb6834"
LABEL = {"alibi": "alibi", "mask_factual": "mask-factual",
         "sycophancy_b": "sycophancy B", "sycophancy_a": "sycophancy A",
         "capture_secret": "capture-the-secret", "unanswerable": "unanswerable",
         "ascii": "ascii-sandbagging"}
# colour = which question (generic vs names the act); hollow = reason first
STYLE = {"neutral": dict(marker="o", color=ORANGE, filled=True),
         "neutral_cot": dict(marker="o", color=ORANGE, filled=False),
         "named": dict(marker="s", color=ACCENT, filled=True),
         "named_cot": dict(marker="s", color=ACCENT, filled=False)}
NICE = {"neutral": "neutral: \"did you lie?\"",
        "neutral_cot": "neutral_cot: same, reason first",
        "named": "named: names the act",
        "named_cot": "named_cot: same, reason first"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})

ORDER = list(PA.groupby("category").auroc.median().sort_values(ascending=False).index)
Y = {c: len(ORDER) - 1 - i for i, c in enumerate(ORDER)}


def rows_axis(ax, xmax, mid, xlabel, show_labels):
    ax.axvline(mid, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(mid, len(ORDER) - 0.4, "coin flip", color=MUTED, fontsize=9, ha="center")
    ax.set_yticks([Y[c] for c in ORDER])
    ax.set_yticklabels([LABEL[c] for c in ORDER])
    ax.tick_params(axis="y", length=0, labelleft=show_labels)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(ORDER) - 0.1)
    ax.set_xlabel(xlabel, fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def dot(ax, x, y, cond, size=70):
    st = STYLE[cond]
    ax.scatter(x, y, s=size, marker=st["marker"], zorder=3, linewidths=1.8,
               facecolor=st["color"] if st["filled"] else "white", edgecolor=st["color"])


# 1. self-report by question -------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
rows_axis(a1, 100, 50, "Balanced accuracy from the letter gemma wrote (%)", True)
rows_axis(a2, 1.0, 0.5, "AUROC from gemma's probability of \"yes\"", False)
a2.axvspan(0.46, 0.80, color=BAND, zorder=0)
a2.text(0.63, -0.45, "paper's range for gemma", color=ACCENT, fontsize=9, ha="center")
for _, r in PA.iterrows():
    dot(a1, r.balanced_acc_from_letters, Y[r.category], r.condition)
    dot(a2, r.auroc, Y[r.category], r.condition)
a1.set_title("Scored on the letter (A = yes, B = no)", fontsize=10.5, loc="left", color=INK)
a2.set_title("Scored on the probability of \"yes\"", fontsize=10.5, loc="left", color=INK)
fig.suptitle("Asking gemma \"did you lie?\" clearly works only for alibi",
             x=0.125, y=1.04, ha="left", fontsize=13, fontweight="bold")
fig.text(0.125, 0.975, "Each marker is one of the four follow-up questions. "
         "Right of the dashed line = catches lies; left = points the wrong way.",
         fontsize=9.5, color=MUTED, ha="left")
handles = [Line2D([], [], ls="", marker=STYLE[c]["marker"], ms=8, mew=1.8,
                  mec=STYLE[c]["color"],
                  mfc=STYLE[c]["color"] if STYLE[c]["filled"] else "white", label=NICE[c])
           for c in KEEP]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.07))
fig.savefig(CHARTS / "selfreport_by_question.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved selfreport_by_question.png")

# 2. does it survive swapping A and B? ----------------------------------------
s = SW.groupby("category").agg(a=("auroc_orig", "median"), b=("auroc_swap", "median"))
fig, ax = plt.subplots(figsize=(9.5, 5.0))
rows_axis(ax, 1.0, 0.5, "AUROC: how often a lie gets a higher P(yes) than an honest answer", True)
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
ax.set_title("Swap A and B: alibi holds, ascii-sandbagging collapses\n",
             loc="left", fontsize=13, fontweight="bold")
ax.text(0, 1.02, "Same 626 rows asked twice: \"yes, I lied\" as option A, then as option B.   "
        "Median over the four questions.", transform=ax.transAxes, fontsize=9.5,
        color=MUTED, va="bottom")
fig.tight_layout()
fig.savefig(CHARTS / "selfreport_swap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved selfreport_swap.png")
