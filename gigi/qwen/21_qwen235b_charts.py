"""The Qwen3-235B chart and tables for gigi/RESULTS_HERMENEUTICS.md, from compare_models.py's CSVs.

  charts/qwen235b_selfreport.png  per category: the range of each Qwen model's four
      self-report questions (AUROC from P("yes")), Qwen3-32B, Qwen3-Next-80B and
      Qwen3-235B, against the paper's Qwen3-235B range
  RESULTS_HERMENEUTICS.md  the tables between the QWEN235-RATES, QWEN235-AUROC,
      QWEN235-SWAP, QWEN235-COUNTS and QWEN235-PAPER markers (the last: the paper's Figure 4
      self-report numbers next to ours, pooled over all categories as the paper scores)

Run from the repo root (after compare_models.py):
    uv run --with pandas --with matplotlib python gigi/qwen/21_qwen235b_charts.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

G = Path(__file__).resolve().parent.parent
RES = G / "qwen/results"
S = pd.read_csv(RES / "comparison_selfreport_auroc.csv")
LABEL = {"alibi": "alibi", "mask_factual": "mask-factual", "ascii": "ascii-sandbagging", "sycophancy_b": "sycophancy B", "sycophancy_a": "sycophancy A",
         "capture_secret": "capture-the-secret", "unanswerable": "unanswerable",
         "mask_known_facts": "MASK known facts", "mask_statistics": "MASK statistics",
         "mask_continuations": "MASK continuations", "mask_disinformation": "MASK disinformation"}
INK, MUTED, GRID = "#17140f", "#8a8579", "#e1e0d9"
GEMMA = ("gemma-3-27b", "gemma-3-27b", "#2a78d6")
QWENS = [("qwen3-32b", "Qwen3-32B", "#eb6834"), ("qwen3-next-80b", "Qwen3-Next-80B", "#1baf7a"),
         ("qwen3-235b", "Qwen3-235B", "#eda100")]   # one fixed colour per model, on every chart
M = "qwen3-235b"
TITLE = "Our Qwen3-235B reports its own lies no better than Qwen3-32B,\nand nowhere near the paper's Qwen3-235B"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK})
cats = [c for c in LABEL if c in set(S[S.model == M].category)]
rng = S.groupby(["model", "category"]).auroc.agg(["min", "max"])
n70 = {m: int((S[(S.model == m) & S.category.isin(cats)].auroc >= 0.70).sum()) for m, _, _ in [GEMMA] + QWENS}
ncell = {m: int((S.model == m).sum() if m == M else (S[(S.model == m) & S.category.isin(cats)]).shape[0]) for m, _, _ in [GEMMA] + QWENS}

# 1. self-report: the three Qwen models ---------------------------------------------------
order = sorted(cats, key=lambda c: -rng.loc[(M, c), "max"])
Y = {c: len(order) - 1 - i for i, c in enumerate(order)}
fig, ax = plt.subplots(figsize=(10, 7.4))
ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
ax.set_yticks([Y[c] for c in order]); ax.set_yticklabels([LABEL[c] for c in order])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.03); ax.set_ylim(-0.7, len(order) - 0.1)
ax.set_xlabel("AUROC from the model's probability of \"yes, I lied\"", fontsize=10)
ax.annotate("Each bar = the range over the four follow-up questions (neutral, neutral_cot, named, named_cot).\n"
            "Each model is asked about its own conversations.  0.5 = coin flip, 1 = perfect",
            xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.4)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.axvspan(0.98, 0.99, color=MUTED, alpha=0.35, zorder=0)
ax.text(0.972, Y[order[-1]] + 0.45, "paper: Qwen3-235B,\nasked about its own lies", color=MUTED, fontsize=8.5, ha="right", va="center")
for c in order:
    for (m, _name, col), dy in zip(QWENS, (0.25, 0.0, -0.25)):
        if (m, c) not in rng.index:
            continue
        lo, hi = rng.loc[(m, c), "min"], rng.loc[(m, c), "max"]
        ax.plot([lo, hi], [Y[c] + dy, Y[c] + dy], color=col, lw=6, solid_capstyle="round", alpha=0.9, zorder=3)
ax.set_title(TITLE, loc="left", fontsize=13, fontweight="bold", pad=26)
ax.text(0, 1.015, f"AUROC of 0.70 or more, out of {ncell[M]} category × question combinations: "
        + ", ".join(f"{name} {n70[m]}" for m, name, _ in QWENS) + ".", transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.legend(handles=[Line2D([], [], color=col, lw=6, label=name + " (ours)") for _, name, col in QWENS],
           loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
fig.savefig(G / "charts/qwen235b_selfreport.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved the chart | cells with AUROC >= 0.70:", n70, "of", ncell)

# 2. the RESULTS_HERMENEUTICS.md tables ---------------------------------------------------
R = pd.read_csv(RES / "comparison_lying_rates.csv").set_index(["model", "category"])
C = pd.read_csv(RES / "comparison_counts.csv")
W = pd.read_csv(RES / "comparison_swap.csv").set_index(["model", "category"])
fmt = lambda m, c: (f"{100 * R.loc[(m, c), 'rate']:.1f}% ({100 * R.loc[(m, c), 'ci_lo']:.0f}–"
                    f"{100 * R.loc[(m, c), 'ci_hi']:.0f}%, n = {int(R.loc[(m, c), 'n'])})") if (m, c) in R.index else "—"
by = sorted([c for c in LABEL if (M, c) in R.index], key=lambda c: -R.loc[(M, c), "rate"])
ALL = [GEMMA] + QWENS
rates = ["| Category | " + " | ".join(name for _, name, _ in ALL) + " |", "|---" * (len(ALL) + 1) + "|"]
rates += [f"| {LABEL[c]} | " + " | ".join(fmt(m, c) for m, _, _ in ALL) + " |" for c in by]

A = S[S.model == M].set_index(["category", "question"])
cell = lambda c, q: (lambda r: f"{'**' if r.auroc >= 0.70 else ''}{r.auroc:.2f}{'**' if r.auroc >= 0.70 else ''} ({r.ci_lo:.2f}–{r.ci_hi:.2f})")(A.loc[(c, q)])
span = lambda m, c: f"{rng.loc[(m, c), 'min']:.2f}–{rng.loc[(m, c), 'max']:.2f}" if (m, c) in rng.index else "—"
auroc = ["| Category | n (lied / honest) | `neutral` | `neutral_cot` | `named` | `named_cot` | Qwen3-Next-80B (range) | Qwen3-32B (range) | gemma-3-27b (range) |",
         "|---|---|---|---|---|---|---|---|---|"]
for c in order:
    r = A.loc[(c, "neutral")]
    auroc.append(f"| {LABEL[c]} | {int(r.n_lied)} / {int(r.n_honest)} | " + " | ".join(cell(c, q) for q in ("neutral", "neutral_cot", "named", "named_cot"))
                 + f" | {span('qwen3-next-80b', c)} | {span('qwen3-32b', c)} | {span('gemma-3-27b', c)} |")

sw = lambda m, c: f"{W.loc[(m, c), 'auroc_orig']:.2f} → {W.loc[(m, c), 'auroc_swap']:.2f}" if (m, c) in W.index else "not run"
swap = ["| Category | " + " | ".join(name for _, name, _ in ALL) + " |", "|---" * (len(ALL) + 1) + "|"]
swap += [f"| {LABEL[c]} | " + " | ".join(sw(m, c) for m, _, _ in ALL) + " |" for c in order if (M, c) in W.index]

counts = ["| Category | confess | double down | false confess | honest | balanced accuracy | Qwen3-Next-80B: balanced accuracy |",
          "|---|---|---|---|---|---|---|"]
for c in order:
    x = C[(C.model == M) & (C.category == c) & (C.question == "neutral")].iloc[0]
    y = C[(C.model == "qwen3-next-80b") & (C.category == c) & (C.question == "neutral")]
    counts.append(f"| {LABEL[c]} | {int(x.confess)} | {int(x.double_down)} | {int(x.false_confess)} | {int(x.honest)} | {x.balanced_accuracy:.0f} | "
                  + (f"{y.iloc[0].balanced_accuracy:.0f}" if len(y) else "—") + " |")

# the paper's self-report numbers next to ours, scored the paper's way (pooled over all categories).
# PAPER = read off the paper's Figure 4 ("Baseline Performance Comparison by Fold"; alignment.anthropic.com/
# 2026/lie-detectors/fig4.png): (fold A reasoning, fold A single token, fold B reasoning, fold B single token)
PAPER = {"gemma-3-27b": {"plain": (0.47, 0.51, 0.60, 0.60), "rubric": (0.65, 0.65, 0.57, 0.49)},
         "qwen3-next-80b": {"plain": (0.69, 0.71, 0.68, 0.69), "rubric": (0.45, 0.44, 0.75, 0.70)},
         "qwen3-235b": {"plain": (0.85, 0.81, 0.72, 0.80), "rubric": (0.99, 0.99, 0.70, 0.59)}}
PO = pd.read_csv(RES / "comparison_selfreport_auroc_pooled.csv").set_index(["model", "question"])
span2 = lambda a, b: f"{min(a, b):.2f}" if a == b else f"{min(a, b):.2f}–{max(a, b):.2f}"
pp = lambda v: f"A {span2(*v[:2])}, B {span2(*v[2:])}"
po = lambda m, q: f"{PO.loc[(m, q), 'auroc']:.2f} ({PO.loc[(m, q), 'ci_lo']:.2f}–{PO.loc[(m, q), 'ci_hi']:.2f})"
paper = ["| Model | Paper: Direct Self-Report (fold A, fold B) | Paper: with Rubric (fold A, fold B) | Ours: `neutral` | Ours: `neutral_cot` | Ours: `named` | Ours: `named_cot` |",
         "|---|---|---|---|---|---|---|"]
for m, name, _ in [GEMMA] + QWENS:
    pa = PAPER.get(m)
    paper.append(f"| {name} | {pp(pa['plain']) if pa else 'not tested'} | {pp(pa['rubric']) if pa else 'not tested'} | "
                 + " | ".join(po(m, q) for q in ("neutral", "neutral_cot", "named", "named_cot")) + " |")

res = G / "RESULTS_HERMENEUTICS.md"
s = res.read_text()
for tag, lines in [("QWEN235-RATES", rates), ("QWEN235-AUROC", auroc), ("QWEN235-SWAP", swap), ("QWEN235-COUNTS", counts),
                   ("QWEN235-PAPER", paper)]:
    s, k = re.subn(rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->",
                   lambda _: f"<!-- {tag}:START -->\n" + "\n".join(lines) + f"\n<!-- {tag}:END -->", s, flags=re.S)
    print(f"RESULTS_HERMENEUTICS.md {tag} table updated" if k else f"!! no {tag} markers in RESULTS_HERMENEUTICS.md - table not written")
res.write_text(s)
