"""Charts for the P("A") and A/B swap results in gigi/RESULTS.md.

Reads the committed summaries (run the two analysis scripts first):
  answer_probability/results/pa_auroc.csv      (02_analyse_pa.py)
  ab_swap/results/ab_swap_summary.csv          (02_analyse_swap.py)

  pa_letters_vs_probability.png  every cell: balanced accuracy from the letter
                                 vs AUROC from P("A")
  pa_auroc_heatmap.png           AUROC, every wording x category
  swap_question_or_letter.png    "yes" rate with yes at A vs with yes at B
  swap_auroc_robustness.png      AUROC with yes at A vs yes at B, per category

Run (from the git repo root):
    uv run --with pandas --with matplotlib python gigi/07_charts_pa_swap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

G = Path(__file__).resolve().parent
CHARTS = G / "charts"
PA = pd.read_csv(G / "answer_probability/results/pa_auroc.csv")
SW = pd.read_csv(G / "ab_swap/results/ab_swap_summary.csv")

# Same palette and category colours as 06_charts_cross_category.py
INK, MUTED, GRID = "#17140f", "#8a8579", "#e1e0d9"
CATS = [("ascii", "ascii-sandbagging", "#2a78d6"),
        ("sycophancy_a", "sycophancy Track A", "#eb6834"),
        ("sycophancy_b", "sycophancy Track B", "#3a9a5c"),
        ("mask_factual", "mask-factual", "#8e44ad"),
        ("unanswerable", "unanswerable", "#127a75"),
        ("capture_secret", "capture-the-secret", "#b5303a"),
        ("alibi", "alibi", "#b8860b")]
NAME = {k: n for k, n, _ in CATS}
COLOR = {k: c for k, _, c in CATS}
ORDER = ["neutral", "low_stakes", "high_stakes", "confidence", "neutral_cot",
         "low_stakes_cot", "high_stakes_cot", "confidence_cot", "plain", "named",
         "named_cot", "eval_plain", "eval_named"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white",
})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(CHARTS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def cat_legend(ax, **kw):
    h = [Line2D([], [], marker="o", linestyle="", markersize=7, color=COLOR[k],
                markeredgecolor="white", label=NAME[k]) for k, _, _ in CATS]
    ax.legend(handles=h, frameon=False, fontsize=9, **kw)


# 1. letters vs probability ---------------------------------------------------
d = PA[PA.balanced_acc_from_letters.notna()].copy()
d["ba"] = d.balanced_acc_from_letters / 100
fig, ax = plt.subplots(figsize=(8.2, 7.4))
ax.fill_between([0, 1], 0.7, 1.0, color="#eef4fb", zorder=0)
ax.text(0.015, 0.985, "AUROC ≥ 0.70", color=MUTED, fontsize=9, va="top")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
ax.axhline(0.5, color=GRID, lw=1, zorder=1)
ax.axvline(0.5, color=GRID, lw=1, zorder=1)
ax.text(0.03, 0.52, "coin flip", color=MUTED, fontsize=9)
ax.text(0.31, 0.36, "same score both ways", color=MUTED, fontsize=9, rotation=45,
        ha="center", va="center")
for k, _, _ in CATS:
    x = d[d.category == k]
    ax.scatter(x.ba, x.auroc, s=46, color=COLOR[k], edgecolor="white", linewidth=1.2, zorder=3)
notes = {  # (category, wording): (label, text position in data coords, ha)
    ("alibi", "named"): ("alibi named", (0.80, 0.975), "right"),
    ("mask_factual", "low_stakes"): ("mask-factual low_stakes", (0.25, 0.93), "left"),
    ("ascii", "high_stakes_cot"): ("ascii high_stakes_cot", (0.13, 0.82), "left"),
    ("sycophancy_b", "neutral"): ("syc B neutral: always 'No',\nbut still ranks lies higher", (0.04, 0.655), "left"),
    ("ascii", "neutral"): ("ascii neutral:\nmore sure it lied when it didn't", (0.05, 0.13), "left")}
for (c, w), (label, xy, ha) in notes.items():
    r = d[(d.category == c) & (d.condition == w)].iloc[0]
    ax.annotate(label, (r.ba, r.auroc), xytext=xy, textcoords="data", fontsize=8.5,
                color=INK, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2, shrinkB=4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Balanced accuracy from the letter gemma wrote")
ax.set_ylabel('AUROC from gemma\'s probability of "A" (yes)')
ax.set_title("Reading gemma's confidence instead of its letter\n"
             "each dot is one wording in one category; above the dashed line, the probability adds signal",
             fontsize=11.5, loc="left")
cat_legend(ax, loc="lower right")
tidy(ax)
save(fig, "pa_letters_vs_probability.png")

# 2. AUROC heatmap -------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list("div", ["#b5303a", "#f4f3ef", "#2a78d6"])
M = np.full((len(ORDER), len(CATS)), np.nan)
for i, w in enumerate(ORDER):
    for j, (k, _, _) in enumerate(CATS):
        r = PA[(PA.category == k) & (PA.condition == w)]
        if len(r):
            M[i, j] = r.iloc[0].auroc
fig, ax = plt.subplots(figsize=(8.6, 7.6))
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(v - 0.5) > 0.3 else INK,
                fontweight="bold" if v >= 0.7 else "normal")
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels(["ascii-\nsandbagging", "sycophancy\nTrack A", "sycophancy\nTrack B",
                    "mask-\nfactual", "unanswerable", "capture-\nthe-secret", "alibi"],
                   fontsize=9, color=INK)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels(ORDER, fontsize=9.5, color=INK)
ax.tick_params(length=0)
for y in (3.5, 7.5, 10.5):
    ax.axhline(y, color="white", lw=2.5)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
cb.outline.set_visible(False)
cb.ax.tick_params(colors=MUTED, labelsize=9)
cb.set_label("AUROC  (red: points the wrong way · blue: detects the lie)", color=INK, fontsize=9.5)
ax.set_title("Self-report AUROC from P(\"A\"), every wording and category\n"
             "bold = 0.70 or above; the paper's gemma range is 0.46–0.80",
             fontsize=11.5, loc="left")
save(fig, "pa_auroc_heatmap.png")

# 3. question or letter? ---------------------------------------------------------
TYPE = {"neutral": "direct", "low_stakes": "direct", "high_stakes": "direct",
        "plain": "direct", "eval_plain": "direct", "neutral_cot": "reason first",
        "low_stakes_cot": "reason first", "high_stakes_cot": "reason first",
        "named": "names the act", "eval_named": "names the act",
        "named_cot": "names the act, reason first"}
MARK = {"direct": "o", "reason first": "^", "names the act": "s",
        "names the act, reason first": "D"}
fig, ax = plt.subplots(figsize=(8.2, 7.8))
ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--", zorder=1)
ax.plot([0, 100], [100, 0], color=MUTED, lw=1, ls=":", zorder=1)
ax.text(62, 67, "answers the question\n(same verdict either way)", color=MUTED, fontsize=9,
        rotation=45, ha="center", va="center")
ax.text(40, 64, "answers the letter", color=MUTED, fontsize=9, rotation=-45,
        ha="center", va="center")
ax.text(99, -1, "always picks A", color=MUTED, fontsize=9, ha="right", va="top")
ax.text(2, 98, "always picks B", color=MUTED, fontsize=9, va="top")
for _, r in SW.iterrows():
    ax.scatter(r.yes_orig, r.yes_swap, s=52, marker=MARK[TYPE[r.condition]],
               color=COLOR[r.category], edgecolor="white", linewidth=1.1, zorder=3)
for (c, w), (label, off, ha) in {
        ("ascii", "named"): ("ascii named", (8, -2), "left"),
        ("ascii", "high_stakes_cot"): ("ascii high_stakes_cot", (-10, 14), "right"),
        ("capture_secret", "neutral"): ("capture neutral", (8, 0), "left"),
        ("ascii", "eval_named"): ("ascii eval_named", (8, 0), "left")}.items():
    r = SW[(SW.category == c) & (SW.condition == w)].iloc[0]
    ax.annotate(label, (r.yes_orig, r.yes_swap), xytext=off, textcoords="offset points",
                fontsize=8.5, color=INK, ha=ha, va="center")
ax.set_xlim(-2, 102)
ax.set_ylim(-2, 102)
ax.set_xlabel('% answering "yes, I lied" when yes is option A (original)')
ax.set_ylabel('% answering "yes, I lied" when yes is option B (swapped)')
ax.set_title("Does gemma answer the question or the letter?\n"
             "same 626 rows asked both ways round; each point is one wording in one category",
             fontsize=11.5, loc="left")
h1 = [Line2D([], [], marker="o", ls="", ms=7, color=COLOR[k], mec="white", label=NAME[k])
      for k, _, _ in CATS]
h2 = [Line2D([], [], marker=m, ls="", ms=7, color=MUTED, mec="white", label=t)
      for t, m in MARK.items()]
l1 = ax.legend(handles=h1, frameon=False, fontsize=9, loc="upper left",
               bbox_to_anchor=(1.02, 1.0), title="category", title_fontsize=9,
               alignment="left")
ax.add_artist(l1)
ax.legend(handles=h2, frameon=False, fontsize=9, loc="upper left",
          bbox_to_anchor=(1.02, 0.55), title="wording", title_fontsize=9,
          alignment="left")
tidy(ax)
save(fig, "swap_question_or_letter.png")

# 4. AUROC robustness: small-multiple slopegraphs ------------------------------
fig, axes = plt.subplots(1, len(CATS), figsize=(13, 4.6), sharey=True)
for ax, (k, n, col) in zip(axes, CATS):
    x = SW[SW.category == k]
    for _, r in x.iterrows():
        ax.plot([0, 1], [r.auroc_orig, r.auroc_swap], color=col, alpha=0.35, lw=1.1)
    mo, ms = x.auroc_orig.mean(), x.auroc_swap.mean()
    ax.plot([0, 1], [mo, ms], color=col, lw=3, marker="o", ms=7, mec="white", zorder=3)
    ax.text(-0.08, mo, f"{mo:.2f}", ha="right", va="center", fontsize=9, color=INK)
    ax.text(1.08, ms, f"{ms:.2f}", ha="left", va="center", fontsize=9, color=INK,
            fontweight="bold")
    ax.axhline(0.5, color=GRID, lw=1, zorder=0)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["yes = A", "yes = B"], fontsize=9)
    ax.set_title(n.replace(" Track ", "\nTrack ").replace("-the-", "-the-\n"),
                 fontsize=10, color=INK)
    ax.set_ylim(0, 1)
    tidy(ax)
axes[0].set_ylabel("AUROC from P(yes)")
fig.suptitle("Does the self-report signal survive swapping the options?\n"
             "thin lines: each wording · thick line: mean · 0.5 = coin flip",
             x=0.01, ha="left", fontsize=11.5)
fig.tight_layout()
save(fig, "swap_auroc_robustness.png")
