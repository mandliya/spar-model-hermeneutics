"""The substitute-gap chart and table for gigi/RESULTS.md, from substitute_gap.py's CSV.

  charts/substitute_gap_selfreport.png  per category: the range of the four
      self-report questions for Qwen3-Next-80B asked about its own conversation
      (target) and for gemma-3-27b asked about the same conversation
      (substitute), same labels (did Qwen lie)
  RESULTS.md  the table between the SUBSTITUTE-GAP markers

Run from the repo root (after substitute_gap.py):
    uv run --with pandas --with matplotlib python gigi/qwen/18_substitute_charts.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

G = Path(__file__).resolve().parent.parent
RES = G / "qwen/results"
R = pd.read_csv(RES / "substitute_gap_selfreport.csv")
OWN = pd.read_csv(RES / "comparison_selfreport_auroc.csv")
OWN = OWN[OWN.model == "gemma-3-27b"].groupby("category").auroc.agg(["min", "max"])
LABEL = {"alibi": "alibi", "mask_factual": "mask-factual", "ascii": "ascii-sandbagging", "sycophancy_b": "sycophancy B", "sycophancy_a": "sycophancy A",
         "capture_secret": "capture-the-secret", "unanswerable": "unanswerable",
         "mask_known_facts": "MASK known facts", "mask_statistics": "MASK statistics",
         "mask_continuations": "MASK continuations", "mask_disinformation": "MASK disinformation"}
INK, MUTED, GRID = "#17140f", "#8a8579", "#e1e0d9"
GEMMA, QWEN = "#2a78d6", "#1baf7a"   # gemma and Qwen3-Next-80B, the same colours as on every other chart
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK})

O, SW = R[R.order == "original"], R[R.order == "swapped"]
sgn = lambda v: f"{v:+.2f}".replace("-", "−")
agg = lambda d: d.groupby("category").agg(
    n_lied=("n_lied", "max"), n_honest=("n_honest", "max"),
    t_lo=("target_auroc", "min"), t_hi=("target_auroc", "max"), t_mean=("target_auroc", "mean"),
    s_lo=("substitute_auroc", "min"), s_hi=("substitute_auroc", "max"), s_mean=("substitute_auroc", "mean"),
    gap=("gap", "mean"), target_better=("gap_lo", lambda s: int((s > 0).sum())),
    substitute_better=("gap_hi", lambda s: int((s < 0).sum())))
A, B = agg(O), agg(SW)
order = list(A.sort_values("gap", ascending=False).index)
n_t, n_s = int((O.gap_lo > 0).sum()), int((O.gap_hi < 0).sum())

# 1. the chart ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.6))
Y = {c: len(order) - 1 - i for i, c in enumerate(order)}
ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
ax.set_yticks([Y[c] for c in order]); ax.set_yticklabels([LABEL[c] for c in order])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.03); ax.set_ylim(-0.7, len(order) - 0.1)
ax.set_xlabel("AUROC from the model's probability of \"yes, I lied\", against whether Qwen lied", fontsize=10)
ax.annotate("Each bar = the range over the four follow-up questions (neutral, neutral_cot, named, named_cot); dot = their mean.\n"
            "Same conversations and same labels for both models.  0.5 = coin flip, 1 = perfect",
            xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.4)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.text(1.075, len(order) - 0.42, "mean gap", fontsize=8.5, color=MUTED, ha="center", va="center")
for c in order:
    r = A.loc[c]
    for lo, hi, mean, col, dy in [(r.t_lo, r.t_hi, r.t_mean, QWEN, 0.17), (r.s_lo, r.s_hi, r.s_mean, GEMMA, -0.17)]:
        ax.plot([lo, hi], [Y[c] + dy, Y[c] + dy], color=col, lw=7, solid_capstyle="round", alpha=0.85, zorder=3)
        ax.scatter(mean, Y[c] + dy, s=38, color=col, edgecolor="white", lw=1.6, zorder=4)
    ax.text(1.075, Y[c], sgn(r.gap), fontsize=10, color=INK, ha="center", va="center", clip_on=False)
ax.set_title("Asking gemma about Qwen-80B's conversations detects Qwen's lies about as well\n"
             "as asking Qwen itself, except on ascii and sycophancy A",
             loc="left", fontsize=13, fontweight="bold", pad=26)
ax.text(0, 1.015, f"No clear difference in {len(O) - n_t - n_s} of {len(O)} category × question combinations; "
        f"Qwen better in {n_t}, gemma in {n_s}. Gap = Qwen − gemma.", transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.legend(handles=[Line2D([], [], color=QWEN, lw=7, label="Qwen3-Next-80B asked about its own conversation (target)"),
                    Line2D([], [], color=GEMMA, lw=7, label="gemma-3-27b asked about Qwen's conversation (substitute)")],
           loc="lower center", ncol=1, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.12))
fig.savefig(G / "charts/substitute_gap_selfreport.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved the chart | cells {len(O)}: Qwen better {n_t}, gemma better {n_s}, no clear difference {len(O) - n_t - n_s}")

# 2. the RESULTS.md table (between the SUBSTITUTE-GAP markers) ----------------------------
rng = lambda lo, hi: f"{lo:.2f}–{hi:.2f}"
rows = ["| Category | n (Qwen lied / honest) | Qwen-80B about itself | gemma reading Qwen | Mean gap | Clearly better, of 4 questions (Qwen / gemma) | Swapped letter order: mean gap | Reference: gemma about its own conversations |",
        "|---|---|---|---|---|---|---|---|"]
for c in order:
    r = A.loc[c]
    sw = sgn(B.loc[c, 'gap']) if c in B.index else "not run"
    rows.append(f"| {LABEL[c]} | {int(r.n_lied)} / {int(r.n_honest)} | {rng(r.t_lo, r.t_hi)} | {rng(r.s_lo, r.s_hi)} | "
                f"{sgn(r.gap)} | {int(r.target_better)} / {int(r.substitute_better)} | {sw} | {rng(OWN.loc[c, 'min'], OWN.loc[c, 'max'])} |")

# 3. does gemma give the same answers as Qwen? (between the SUBSTITUTE-ANSWERS markers) -----
def wilson(k, n, z=1.96):
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return c - h, c + h


pct = lambda v: f"{100 * v:.1f}%"
N = ["both_yes", "both_no", "only_target_yes", "only_substitute_yes"]
ans = ["| Question | n (answers) | Qwen says \"yes, I lied\" | gemma says yes | Same answer (95% interval) | gemma yes, Qwen no | Qwen yes, gemma no |",
       "|---|---|---|---|---|---|---|"]
for q in ["neutral", "neutral_cot", "named", "named_cot", "all four"]:
    t = (O if q == "all four" else O[O.question == q])[N].sum()
    n = int(t.sum()); same = int(t.both_yes + t.both_no); lo, hi = wilson(same, n)
    ans.append(f"| {'**all four**' if q == 'all four' else '`' + q + '`'} | {n:,} | {pct((t.both_yes + t.only_target_yes) / n)} | "
               f"{pct((t.both_yes + t.only_substitute_yes) / n)} | {pct(same / n)} ({100 * lo:.0f}–{100 * hi:.0f}%) | "
               f"{int(t.only_substitute_yes):,} | {int(t.only_target_yes):,} |")
cat = ["| Category | n | `neutral`: Qwen yes | gemma yes | same answer (95% interval) | `named`: Qwen yes | gemma yes | same answer (95% interval) |",
       "|---|---|---|---|---|---|---|---|"]
for c in order:
    row = [LABEL[c], str(int(A.loc[c, "n_lied"] + A.loc[c, "n_honest"]))]
    for q in ("neutral", "named"):
        x = O[(O.category == c) & (O.question == q)].iloc[0]
        row += [pct(x.target_yes), pct(x.substitute_yes), f"{pct(x.agreement)} ({100 * x.agreement_lo:.0f}–{100 * x.agreement_hi:.0f}%)"]
    cat.append("| " + " | ".join(row) + " |")

res = G / "RESULTS.md"
s = res.read_text()
for tag, lines in [("SUBSTITUTE-GAP", rows), ("SUBSTITUTE-ANSWERS", ans), ("SUBSTITUTE-ANSWERS-BY-CATEGORY", cat)]:
    s, k = re.subn(rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->",
                   lambda _: f"<!-- {tag}:START -->\n" + "\n".join(lines) + f"\n<!-- {tag}:END -->", s, flags=re.S)
    print(f"RESULTS.md {tag} table updated" if k else f"!! no {tag} markers in RESULTS.md - table not written")
res.write_text(s)
print("\n".join(ans + [""] + cat))
