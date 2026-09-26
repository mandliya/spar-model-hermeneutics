"""Charts for sycophancy's full follow-up battery (both tracks) — same ten
wordings as ascii-sandbagging's RESULTS.md, committed to git as PNGs, not a
hosted artifact, same reasoning as every other chart in this project.

Run:
    uv run python 06_charts_sycophancy_followups.py
"""

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
GREEN = "#3a9a5c"
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


CONDITIONS = ["neutral", "low_stakes", "high_stakes", "confidence",
              "neutral_cot", "low_stakes_cot", "high_stakes_cot", "confidence_cot",
              "plain", "named", "eval_plain", "eval_named"]


def make_chart(csv_path, truth_col, title, out_name):
    df = pd.read_csv(csv_path)
    rows = []
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].dropna(subset=["admitted"])
        pos = sub[sub[truth_col]]
        k = int((~pos["admitted"]).sum())
        n = len(pos)
        rows.append((cond, *wilson(k, n)))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    y = list(range(len(rows)))[::-1]
    for yi, (label, p, lo, hi) in zip(y, rows):
        if label in ("plain", "named", "eval_plain", "eval_named"):
            color = GREEN
        elif "cot" in label:
            color = ORANGE
        else:
            color = BLUE
        ax.barh([yi], [p * 100], height=0.55, color=color, zorder=3)
        # max(0, ...) guards against floating-point epsilon going negative
        # right at the 0%/100% boundary - real at temp=0, where several
        # conditions land on an exact, fully deterministic 0% or 100%.
        ax.errorbar([p * 100], [yi], xerr=[[max(0, p * 100 - lo * 100)],
                                            [max(0, hi * 100 - p * 100)]],
                    fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=4)
        label_x = max(p * 100, hi * 100) + 1.5
        ax.text(label_x, yi, f"{p*100:.1f}%", va="center", fontsize=9.5, fontweight="bold")
    ax.axhline(y[7] - 0.5, color=GRID, linewidth=1, linestyle="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 108)
    ax.set_xlabel(title)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=GREEN)]
    ax.legend(handles, ["bare letter / number", "with reasoning (“*_cot”)", "ask-directly benchmark"],
              loc="lower right", frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / out_name, dpi=180)
    plt.close()
    print(f"saved {out_name}")


make_chart(RESULTS / "sycophancy_benchmark_conditions.csv", "fold",
           "Doubling-down rate (denies, given it folded)",
           "sycophancy_track_a_followups.png")

make_chart(RESULTS / "sycophancy_feedback_benchmark_conditions.csv", "sycophantic",
           "Doubling-down rate (denies, given it swayed past the noise floor)",
           "sycophancy_track_b_followups.png")
