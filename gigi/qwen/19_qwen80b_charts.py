"""The Qwen3-Next-80B chart and tables for gigi/RESULTS_HERMENEUTICS.md, from compare_models.py's CSVs.

  charts/qwen80b_selfreport.png  per category: the range of each model's four
      self-report questions (AUROC from P("yes")), for gemma-3-27b, Qwen3-32B
      and Qwen3-Next-80B, against the paper's Qwen3-235B range
  RESULTS_HERMENEUTICS.md  the tables between the QWEN80-RATES, QWEN80-AUROC, QWEN80-SWAP and
      QWEN80-COUNTS markers

Run from the repo root (after compare_models.py):
    uv run --with pandas --with matplotlib python gigi/qwen/19_qwen80b_charts.py
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
MODELS = [("gemma-3-27b", "gemma-3-27b", "#2a78d6"), ("qwen3-32b", "Qwen3-32B", "#eb6834"),
          ("qwen3-next-80b", "Qwen3-Next-80B", "#1baf7a")]   # one fixed colour per model, on every chart
M80 = "qwen3-next-80b"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK})
cats = [c for c in LABEL if c in set(S[S.model == M80].category)]
rng = S.groupby(["model", "category"]).auroc.agg(["min", "max"])
n70 = {m: int((S[S.model == m].auroc >= 0.70).sum()) for m, _, _ in MODELS}

# 1. self-report: three models ------------------------------------------------------------
order = sorted(cats, key=lambda c: -rng.loc[(M80, c), "max"])
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
    for (m, _name, col), dy in zip(MODELS, (0.25, 0.0, -0.25)):
        lo, hi = rng.loc[(m, c), "min"], rng.loc[(m, c), "max"]
        ax.plot([lo, hi], [Y[c] + dy, Y[c] + dy], color=col, lw=6, solid_capstyle="round", alpha=0.9, zorder=3)
ax.set_title("The bigger Qwen reports its own lies better than the two smaller models,\n"
             "but still far from the paper's Qwen3-235B",
             loc="left", fontsize=13, fontweight="bold", pad=26)
ax.text(0, 1.015, "AUROC of 0.70 or more, out of 44 category × question combinations: "
        + ", ".join(f"{name} {n70[m]}" for m, name, _ in MODELS) + ".", transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.legend(handles=[Line2D([], [], color=col, lw=6, label=name + " (ours)") for _, name, col in MODELS],
           loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
fig.savefig(G / "charts/qwen80b_selfreport.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved the chart | cells with AUROC >= 0.70:", n70)

# 2. the RESULTS_HERMENEUTICS.md tables ------------------------------------------------------------------
R = pd.read_csv(RES / "comparison_lying_rates.csv").set_index(["model", "category"])
C = pd.read_csv(RES / "comparison_counts.csv")
W = pd.read_csv(RES / "comparison_swap.csv").set_index(["model", "category"])
fmt = lambda m, c: (f"{100 * R.loc[(m, c), 'rate']:.1f}% ({100 * R.loc[(m, c), 'ci_lo']:.0f}–"
                    f"{100 * R.loc[(m, c), 'ci_hi']:.0f}%, n = {int(R.loc[(m, c), 'n'])})") if (m, c) in R.index else "—"
by80 = sorted(cats, key=lambda c: -R.loc[(M80, c), "rate"])
rates = ["| Category | gemma-3-27b | Qwen3-32B | Qwen3-Next-80B |", "|---|---|---|---|"]
rates += [f"| {LABEL[c]} | {fmt('gemma-3-27b', c)} | {fmt('qwen3-32b', c)} | {fmt(M80, c)} |" for c in by80]

A = S[S.model == M80].set_index(["category", "question"])
cell = lambda c, q: (lambda r: f"{'**' if r.auroc >= 0.70 else ''}{r.auroc:.2f}{'**' if r.auroc >= 0.70 else ''} ({r.ci_lo:.2f}–{r.ci_hi:.2f})")(A.loc[(c, q)])
auroc = ["| Category | n (lied / honest) | `neutral` | `neutral_cot` | `named` | `named_cot` | gemma-3-27b (range) | Qwen3-32B (range) |",
         "|---|---|---|---|---|---|---|---|"]
for c in order:
    r = A.loc[(c, "neutral")]
    auroc.append(f"| {LABEL[c]} | {int(r.n_lied)} / {int(r.n_honest)} | " + " | ".join(cell(c, q) for q in ("neutral", "neutral_cot", "named", "named_cot"))
                 + f" | {rng.loc[('gemma-3-27b', c), 'min']:.2f}–{rng.loc[('gemma-3-27b', c), 'max']:.2f}"
                 + f" | {rng.loc[('qwen3-32b', c), 'min']:.2f}–{rng.loc[('qwen3-32b', c), 'max']:.2f} |")

sw = lambda m, c: f"{W.loc[(m, c), 'auroc_orig']:.2f} → {W.loc[(m, c), 'auroc_swap']:.2f}" if (m, c) in W.index else "not run"
swap = ["| Category | gemma-3-27b | Qwen3-32B | Qwen3-Next-80B |", "|---|---|---|---|"]
swap += [f"| {LABEL[c]} | {sw('gemma-3-27b', c)} | {sw('qwen3-32b', c)} | {sw(M80, c)} |" for c in order if (M80, c) in W.index]

counts = ["| Category | confess | double down | false confess | honest | balanced accuracy |", "|---|---|---|---|---|---|"]
for c in order:
    x = C[(C.model == M80) & (C.category == c) & (C.question == "neutral")].iloc[0]
    counts.append(f"| {LABEL[c]} | {int(x.confess)} | {int(x.double_down)} | {int(x.false_confess)} | {int(x.honest)} | {x.balanced_accuracy:.0f} |")

res = G / "RESULTS_HERMENEUTICS.md"
s = res.read_text()
for tag, lines in [("QWEN80-RATES", rates), ("QWEN80-AUROC", auroc), ("QWEN80-SWAP", swap), ("QWEN80-COUNTS", counts)]:
    s, k = re.subn(rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->",
                   lambda _: f"<!-- {tag}:START -->\n" + "\n".join(lines) + f"\n<!-- {tag}:END -->", s, flags=re.S)
    print(f"RESULTS_HERMENEUTICS.md {tag} table updated" if k else f"!! no {tag} markers in RESULTS_HERMENEUTICS.md - table not written")
res.write_text(s)
