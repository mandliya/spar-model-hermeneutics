"""Static chart for gigi/sycophancy/RESULTS.md — committed to git as a PNG,
not a hosted artifact, same reasoning as the ascii-sandbagging charts.

Run:
    uv run python 06_charts_sycophancy.py
"""

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

df = pd.read_csv(RESULTS / "sycophancy_feedback_conditions.csv")
main = df[df["kind"] == "main"].copy()

baselines = (main[main["task"] == "sycophancy_feedback_user_did_not_write"]
             .set_index("argument_group_id")["sentiment_score"])
main["baseline"] = main["argument_group_id"].map(baselines)
main["shift"] = main["sentiment_score"] - main["baseline"]

rows = []
for task, label in [
    ("sycophancy_feedback_user_likes", "user_likes"),
    ("sycophancy_feedback_user_wrote", "user_wrote"),
    ("sycophancy_feedback_user_dislikes", "user_dislikes"),
]:
    sub = main[main["task"] == task]["shift"].dropna()
    m = sub.mean()
    se = sub.std(ddof=1) / (len(sub) ** 0.5)
    rows.append((label, m, se, len(sub)))

noise = df[df["kind"] == "noise_rerun"].copy()
noise_pairs = noise.merge(
    main[main["task"] == "sycophancy_feedback_user_did_not_write"][["argument_group_id", "sentiment_score"]],
    on="argument_group_id", suffixes=("_2", "_1"))
noise_floor = (noise_pairs["sentiment_score_2"] - noise_pairs["sentiment_score_1"]).abs().mean()

fig, ax = plt.subplots(figsize=(7, 3.2))
y = list(range(len(rows)))[::-1]
ax.axvspan(-noise_floor, noise_floor, color=GRID, alpha=0.6, zorder=1)
ax.axvline(0, color=MUTED, linewidth=1, zorder=2)
for yi, (label, m, se, n) in zip(y, rows):
    color = BLUE if m > 0 else ORANGE
    ax.barh([yi], [m], height=0.5, color=color, zorder=3)
    ax.errorbar([m], [yi], xerr=[se], fmt="none", ecolor=INK, elinewidth=1.5, capsize=4, zorder=4)
    xpos = m + (0.35 if m > 0 else -0.35)
    ha = "left" if m > 0 else "right"
    ax.text(xpos, yi, f"{m:+.2f}", va="center", ha=ha, fontsize=11, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(-4.2, 4.5)
ax.set_xlabel("Sentiment shift vs. same argument's neutral baseline (1-10 scale)")
ax.text(0, len(rows) - 0.35, f"noise floor: ±{noise_floor:.2f}",
        ha="center", va="bottom", fontsize=9, color=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "sycophancy_feedback_shift.png", dpi=180)
plt.close()
print("saved sycophancy_feedback_shift.png")
