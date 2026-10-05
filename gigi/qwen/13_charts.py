"""The two gemma-vs-Qwen charts for gigi/RESULTS.md, from compare_models.py's CSVs.

  qwen_vs_gemma_selfreport.png  per category: the range of each model's four
      self-report questions (AUROC from P("yes")), against the paper's ranges
  qwen_third_person_vs_selfreport.png  Qwen's self-report range vs the three
      monitors reading its transcripts (same layout as gemma's chart)

Run from the repo root (after compare_models.py):
    uv run --with pandas --with matplotlib python gigi/qwen/13_charts.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

G = Path(__file__).resolve().parent.parent
RES = G / "qwen/results"
S = pd.read_csv(RES / "comparison_selfreport_auroc.csv")
M = pd.read_csv(RES / "comparison_monitor_auroc.csv")
LABEL = {"alibi": "alibi", "sycophancy_b": "sycophancy B", "sycophancy_a": "sycophancy A",
         "capture_secret": "capture-the-secret", "unanswerable": "unanswerable",
         "mask_known_facts": "MASK known facts", "mask_statistics": "MASK statistics",
         "mask_continuations": "MASK continuations", "mask_disinformation": "MASK disinformation"}
INK, MUTED, GRID = "#17140f", "#8a8579", "#e1e0d9"
GEMMA, QWEN = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK})
cats = [c for c in LABEL if c in set(S[S.model == "qwen3-32b"].category)]
rng = S.groupby(["model", "category"]).auroc.agg(["min", "max"])


def axis(ax, order, xlabel, note):
    Y = {c: len(order) - 1 - i for i, c in enumerate(order)}
    ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_yticks([Y[c] for c in order]); ax.set_yticklabels([LABEL[c] for c in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.03); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.annotate(note, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
                ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.4)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    return Y


# 1. self-report: gemma vs Qwen ---------------------------------------------------------
order = sorted(cats, key=lambda c: -rng.loc[("qwen3-32b", c), "max"])
fig, ax = plt.subplots(figsize=(10, 6.6))
Y = axis(ax, order, "AUROC from the model's probability of \"yes, I lied\"",
         "Each bar = the range over the four follow-up questions (neutral, neutral_cot, named, named_cot).\n"
         "Shaded = the paper's reported ranges.")
ax.axvspan(0.46, 0.80, color=GEMMA, alpha=0.07, zorder=0)
ax.axvspan(0.98, 0.99, color=QWEN, alpha=0.35, zorder=0)
ax.text(0.63, len(order) - 0.45, "paper: gemma-3-27b (all lie types and prompt formats)", color=GEMMA, fontsize=8.5, ha="center")
ax.text(0.985, len(order) - 0.45, "paper: Qwen3-235B,\nasked about its own lies", color=QWEN, fontsize=8.5, ha="center", va="bottom")
for c in order:
    for m, col, dy in [("gemma-3-27b", GEMMA, 0.17), ("qwen3-32b", QWEN, -0.17)]:
        lo, hi = rng.loc[(m, c), "min"], rng.loc[(m, c), "max"]
        ax.plot([lo, hi], [Y[c] + dy, Y[c] + dy], color=col, lw=7, solid_capstyle="round", alpha=0.85, zorder=3)
ax.set_title("Qwen3-32B's self-report is in gemma's range,\nnowhere near the paper's Qwen3-235B",
             loc="left", fontsize=13, fontweight="bold")
fig.legend(handles=[Line2D([], [], color=GEMMA, lw=7, label="gemma-3-27b (ours)"),
                    Line2D([], [], color=QWEN, lw=7, label="Qwen3-32B (ours)")],
           loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.07))
fig.savefig(G / "charts/qwen_vs_gemma_selfreport.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 2. Qwen: outside monitors vs self-report --------------------------------------------------
mp = M[(M.model == "qwen3-32b") & (M.variant == "plain")].pivot_table(index="category", columns="monitor", values="auroc")
sr = rng.loc["qwen3-32b"]
out = ["gemini-2.5-flash", "gpt-5-mini"]
beat_any = [c for c in cats if any(mp.loc[c, o] > sr.loc[c, "max"] for o in out)]
beat_all = [c for c in cats if all(mp.loc[c, o] > sr.loc[c, "max"] for o in out)]
order = sorted(cats, key=lambda c: -mp.loc[c, out].mean())
COL = {"gemini-2.5-flash": "#2a78d6", "gpt-5-mini": "#1d9e74", "self": QWEN}
NAME = {"gemini-2.5-flash": "Gemini 2.5 Flash", "gpt-5-mini": "GPT-5 mini", "self": "Qwen as monitor"}
fig, ax = plt.subplots(figsize=(10, 6.4))
Y = axis(ax, order, "AUROC: how often a lie looks more suspicious than an honest answer",
         "Monitors: their 1-100 'is the assistant lying?' score on the transcript (paper's prompt).\n"
         "Self-report: Qwen's probability of \"yes, I lied\" when asked afterwards.  0.5 = coin flip, 1 = perfect")
for c in order:
    ax.plot([sr.loc[c, "min"], sr.loc[c, "max"]], [Y[c], Y[c]], color="#d6d3cb", lw=9, solid_capstyle="round", zorder=2)
    for mon in ["gemini-2.5-flash", "gpt-5-mini", "self"]:
        ax.scatter(mp.loc[c, mon], Y[c], s=75, color=COL[mon], edgecolor="white", lw=1.2, zorder=3)
ax.set_title("For Qwen too, reading the transcript from outside beats asking \"did you lie?\"",
             loc="left", fontsize=13, fontweight="bold", pad=26)
ax.text(0, 1.015, f"An outside monitor beats Qwen's best question in {len(beat_any)} of {len(cats)} categories; "
        f"both outside monitors do in {len(beat_all)}.", transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.legend(handles=[Patch(color="#d6d3cb", label="Qwen asked \"did you lie?\" (range of its 4 questions)")]
           + [Line2D([], [], ls="", marker="o", ms=9, color=COL[k], mec="white", label=NAME[k]) for k in COL],
           loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.14))
fig.savefig(G / "charts/qwen_third_person_vs_selfreport.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved 2 charts | outside monitor beats Qwen's best question:", len(beat_any), "/", len(cats), "| both:", len(beat_all))

# 3. the RESULTS.md tables (between QWEN-RATES and QWEN-COUNTS markers) --------------------
import re  # noqa: E402

R = pd.read_csv(RES / "comparison_lying_rates.csv").set_index(["model", "category"])
C = pd.read_csv(RES / "comparison_counts.csv")
fmt = lambda m, c: (f"{100 * R.loc[(m, c), 'rate']:.1f}% ({100 * R.loc[(m, c), 'ci_lo']:.0f}–"
                    f"{100 * R.loc[(m, c), 'ci_hi']:.0f}%, n = {int(R.loc[(m, c), 'n'])})") if (m, c) in R.index else "—"
rates = ["| Category | gemma-3-27b | Qwen3-32B |", "|---|---|---|"]
rates += [f"| {LABEL[c]} | {fmt('gemma-3-27b', c)} | {fmt('qwen3-32b', c)} |" for c in cats]
counts = ["| Category | gemma: confess | double down | false confess | honest | Qwen: confess | double down | false confess | honest |",
          "|---|---|---|---|---|---|---|---|---|"]
for c in cats:
    row = [LABEL[c]]
    for m in ("gemma-3-27b", "qwen3-32b"):
        x = C[(C.model == m) & (C.category == c) & (C.question == "neutral")]
        row += [str(int(x.iloc[0][k])) for k in ("confess", "double_down", "false_confess", "honest")] if len(x) else ["—"] * 4
    counts.append("| " + " | ".join(row) + " |")
res = G / "RESULTS.md"
s = res.read_text()
for tag, lines in [("QWEN-RATES", rates), ("QWEN-COUNTS", counts)]:
    s = re.sub(rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->",
               lambda _: f"<!-- {tag}:START -->\n" + "\n".join(lines) + f"\n<!-- {tag}:END -->", s, flags=re.S)
res.write_text(s)
print("RESULTS.md Qwen tables updated")
