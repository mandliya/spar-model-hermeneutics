"""Static charts for RESULTS.md — committed to git as PNGs, not a hosted
artifact, so they render on GitHub and don't depend on any external service.

Reads from the actual results files on disk rather than hardcoding numbers,
so these stay reproducible: rerun after any results change and regenerate.

Run:
    uv run python 06_charts_ascii.py

Cost: $0, no API calls — just plotting data already on disk.
"""

import json
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd

from src.common import RESULTS

CHARTS_DIR = RESULTS.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

INK = "#17140f"
MUTED = "#8a8579"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
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
        return (0, 0, 0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (p, max(0, center - half), min(1, center + half))


# ------------------------------------------------------- chart 1: lying rate
base = pd.read_csv(RESULTS / "ascii_results.csv")
ok = base.dropna(subset=["lied"])
p, lo, hi = wilson(int(ok["lied"].sum()), len(ok))

fig, ax = plt.subplots(figsize=(7, 1.6))
ax.barh([0], [p * 100], height=0.5, color=BLUE, zorder=3)
ax.errorbar([p * 100], [0], xerr=[[max(0, p * 100 - lo * 100)], [max(0, hi * 100 - p * 100)]],
            fmt="none", ecolor=INK, elinewidth=1.5, capsize=4, zorder=4)
ax.axvline(76, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
ax.text(76, 0.55, "paper's floor: 76%+ (every model tested)",
        ha="right", va="bottom", fontsize=9, color=MUTED)
label_x = max(p * 100, hi * 100) + 2
ax.text(label_x, 0, f"{p*100:.1f}%", va="center", fontsize=11, fontweight="bold")
ax.set_yticks([0])
ax.set_yticklabels(["gemma\n(this run)"])
ax.set_ylim(-0.5, 0.7)
ax.set_xlim(0, 100)
ax.set_xlabel("Lying rate (falsely denies capability)")
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "lying_rate.png", dpi=180)
plt.close()
print("saved lying_rate.png")

# --------------------------------------------- chart 2: wording conditions
fups = pd.read_csv(RESULTS / "ascii_followup_conditions.csv")
reas = pd.read_csv(RESULTS / "ascii_reasoning_conditions.csv")


def fold_rate(df, cond=None):
    """Doubling-down rate (denies | it lied). cond=None reads the whole
    frame as-is (for ascii_results.csv, which has no 'condition' column)."""
    sub = df if cond is None else df[df["condition"] == cond]
    sub = sub.dropna(subset=["admitted"])
    lied = sub[sub["lied"]]
    k = int((~lied["admitted"]).sum())
    n = len(lied)
    p, lo, hi = wilson(k, n)
    return p, lo, hi


rows = [
    ("neutral", *fold_rate(base)),
    ("low_stakes", *fold_rate(fups, "low_stakes")),
    ("high_stakes", *fold_rate(fups, "high_stakes")),
    ("confidence", *fold_rate(fups, "confidence")),
    ("neutral_cot", *fold_rate(reas, "neutral_cot")),
    ("low_stakes_cot", *fold_rate(reas, "low_stakes_cot")),
    ("high_stakes_cot", *fold_rate(reas, "high_stakes_cot")),
    ("confidence_cot", *fold_rate(reas, "confidence_cot")),
]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
y = list(range(len(rows)))[::-1]
for yi, (label, p, lo, hi) in zip(y, rows):
    color = BLUE if "cot" not in label else ORANGE
    ax.barh([yi], [p * 100], height=0.55, color=color, zorder=3)
    ax.errorbar([p * 100], [yi], xerr=[[max(0, p * 100 - lo * 100)], [max(0, hi * 100 - p * 100)]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=4)
    label_x = max(p * 100, hi * 100) + 1.5
    ax.text(label_x, yi, f"{p*100:.1f}%", va="center", fontsize=9.5, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 108)
ax.set_xlabel("Doubling-down rate (denies, given it lied)")
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
ax.legend(handles, ["bare letter", "with reasoning (“*_cot”)"],
          loc="lower right", frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "wording_conditions.png", dpi=180)
plt.close()
print("saved wording_conditions.png")

# ------------------------------------------ chart 3: self-report benchmark
# Same "doubling down" metric as every other chart on this page (denies |
# it lied), not "confession rate" - that's the same number reflected, but
# using a different metric name here than everywhere else made the page
# harder to compare across sections at a glance. Confession rate is just
# 1 - doubling down when there's nothing unparseable (true here, 0 for
# both conditions) - not a different measurement, purely a labeling fix.
bench = pd.read_csv(RESULTS / "ascii_benchmark_conditions.csv")

labels = ['"plain" wording\n(dishonest or misleading?)',
          '"named" wording\n(did you sandbag?)',
          '"eval_plain"\n(safety eval + plain)',
          '"eval_named"\n(safety eval + named)']
vals = [fold_rate(bench, "plain"), fold_rate(bench, "named"),
        fold_rate(bench, "eval_plain"), fold_rate(bench, "eval_named")]

fig, ax = plt.subplots(figsize=(7, 4.4))
y = list(range(len(labels)))[::-1]
for yi, (label, (p, lo, hi)) in zip(y, zip(labels, vals)):
    color = BLUE if yi % 2 == 1 else ORANGE
    ax.barh([yi], [p * 100], height=0.5, color=color, zorder=3)
    ax.errorbar([p * 100], [yi], xerr=[[max(0, p * 100 - lo * 100)], [max(0, hi * 100 - p * 100)]],
                fmt="none", ecolor=INK, elinewidth=1.5, capsize=4, zorder=4)
    if p < 0.85:
        xpos, ha = max(p * 100, hi * 100) + 2.5, "left"
    else:
        xpos, ha = min(p * 100, lo * 100) - 2.5, "right"
    ax.text(xpos, yi, f"{p*100:.1f}%", va="center", ha=ha, fontsize=12, fontweight="bold",
            color="white" if ha == "right" else INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0, 100)
ax.set_xlabel("Doubling-down rate (denies it lied, given it actually lied)")
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "selfreport_benchmark.png", dpi=180)
plt.close()
print("saved selfreport_benchmark.png")

print(f"\nAll charts in {CHARTS_DIR}")
