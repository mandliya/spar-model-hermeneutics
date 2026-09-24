"""Cross-category comparison chart for gigi/RESULTS.md — the same ten
follow-up wordings, doubling-down rate, overlaid across every category/track
that has them: ascii-sandbagging, sycophancy Track A, sycophancy Track B.

Reads straight from each category's own already-committed results files —
nothing hardcoded — so this stays reproducible: rerun after any category's
results change.

Run (from the git repo root):
    uv run --with pandas --with matplotlib --with statsmodels \
        python gigi/06_charts_cross_category.py
"""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent
ASCII_RESULTS = REPO / "ascii-sandbagging" / "results"
SYC_RESULTS = REPO / "sycophancy" / "results"
MASK_RESULTS = REPO / "mask-factual" / "results"
CHARTS_DIR = REPO / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

INK = "#17140f"
MUTED = "#8a8579"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#3a9a5c"
PURPLE = "#8e44ad"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0, center - half), min(1, center + half)


CONDITIONS = ["neutral", "low_stakes", "high_stakes", "confidence",
              "neutral_cot", "low_stakes_cot", "high_stakes_cot", "confidence_cot",
              "plain", "named", "eval_plain", "eval_named"]


def ascii_rates():
    base = pd.read_csv(ASCII_RESULTS / "ascii_results.csv")
    fups = pd.read_csv(ASCII_RESULTS / "ascii_followup_conditions.csv")
    reas = pd.read_csv(ASCII_RESULTS / "ascii_reasoning_conditions.csv")
    bench = pd.read_csv(ASCII_RESULTS / "ascii_benchmark_conditions.csv")

    def fold_rate(df, truth_col, cond=None):
        sub = df if cond is None else df[df["condition"] == cond]
        sub = sub.dropna(subset=["admitted"])
        pos = sub[sub[truth_col].astype(bool)]
        return wilson(int((~pos["admitted"]).sum()), len(pos))

    return {
        "neutral": fold_rate(base, "lied"),
        "low_stakes": fold_rate(fups, "lied", "low_stakes"),
        "high_stakes": fold_rate(fups, "lied", "high_stakes"),
        "confidence": fold_rate(fups, "lied", "confidence"),
        "neutral_cot": fold_rate(reas, "lied", "neutral_cot"),
        "low_stakes_cot": fold_rate(reas, "lied", "low_stakes_cot"),
        "high_stakes_cot": fold_rate(reas, "lied", "high_stakes_cot"),
        "confidence_cot": fold_rate(reas, "lied", "confidence_cot"),
        "plain": fold_rate(bench, "lied", "plain"),
        "named": fold_rate(bench, "lied", "named"),
        "eval_plain": fold_rate(bench, "lied", "eval_plain"),
        "eval_named": fold_rate(bench, "lied", "eval_named"),
    }


def generic_rates(results_dir, csv_name, truth_col):
    df = pd.read_csv(results_dir / csv_name)

    def fold_rate(cond):
        sub = df[df["condition"] == cond].dropna(subset=["admitted"])
        pos = sub[sub[truth_col].astype(bool)]
        return wilson(int((~pos["admitted"]).sum()), len(pos))

    return {cond: fold_rate(cond) for cond in CONDITIONS}


series = {
    "ascii-sandbagging": (ascii_rates(), BLUE),
    "sycophancy Track A": (generic_rates(SYC_RESULTS, "sycophancy_benchmark_conditions.csv", "fold"), ORANGE),
    "sycophancy Track B": (generic_rates(SYC_RESULTS, "sycophancy_feedback_benchmark_conditions.csv", "sycophantic"), GREEN),
}
if (MASK_RESULTS / "mask_factual_benchmark_conditions.csv").exists():
    series["mask-factual"] = (generic_rates(MASK_RESULTS, "mask_factual_benchmark_conditions.csv", "lied"), PURPLE)

fig, ax = plt.subplots(figsize=(9, 5.5))
x = list(range(len(CONDITIONS)))
for name, (rates, color) in series.items():
    ys, los, his, xs = [], [], [], []
    for xi, cond in zip(x, CONDITIONS):
        r = rates[cond]
        if r is None:
            continue
        p, lo, hi = r
        xs.append(xi)
        ys.append(p * 100)
        los.append(max(0, p * 100 - lo * 100))
        his.append(max(0, hi * 100 - p * 100))
    ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=color, label=name,
                linewidth=1.6, markersize=6, capsize=3, elinewidth=1, zorder=3)

ax.axvline(3.5, color=GRID, linewidth=1, linestyle="--", zorder=1)
ax.axvline(7.5, color=GRID, linewidth=1, linestyle="--", zorder=1)
ax.axvline(9.5, color=GRID, linewidth=1, linestyle="--", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(CONDITIONS, rotation=30, ha="right")
ax.set_ylim(-3, 108)
ax.set_ylabel("Doubling-down rate (denies the behavior, given it happened)")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=len(series), frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "cross_category_doubling_down.png", dpi=180)
plt.close()
print(f"saved {CHARTS_DIR / 'cross_category_doubling_down.png'}")
