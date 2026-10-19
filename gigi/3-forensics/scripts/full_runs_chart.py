"""Chart of results/full_runs.csv: one row per environment, each model's rate of the environment's bad
behaviour as a dot with its 95% Wilson interval and a "k/n" label. Rows without scored runs say why.
  python3 gigi/3-forensics/scripts/full_runs_chart.py [title]     -> charts/full_runs.png
"""
import csv, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
COL = {"qwen3-next-80b": "#1baf7a", "gemma-3-27b": "#2a78d6"}    # same model colours as charts.py
NAME = {"qwen3-next-80b": "Qwen3-Next-80B", "gemma-3-27b": "gemma-3-27b"}
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": MUTED})

rows = list(csv.DictReader(open(HERE / "results" / "full_runs.csv")))
envs = []
for r in rows:
    if r["environment"] not in [e for e, _ in envs]:
        envs.append((r["environment"], r["behaviour"]))
scored = lambda e: sum(int(r["n"]) for r in rows if r["environment"] == e)
envs.sort(key=lambda eb: (scored(eb[0]) == 0, -max([float(r["rate"] or 0) for r in rows if r["environment"] == eb[0]])))

fig, ax = plt.subplots(figsize=(10, 0.55 * len(envs) + 1.6))
for i, (env, behaviour) in enumerate(envs):
    y = len(envs) - 1 - i
    for j, model in enumerate(COL):
        r = next((r for r in rows if r["environment"] == env and r["model"] == model), None)
        yy = y + (0.17 if j == 0 else -0.17)
        if not r or not int(r["n"]):
            why = ("not run (DeepInfra drops gemma's tool calls here)" if model == "gemma-3-27b" and env in
                   ("currency_conversion", "funding_email") else
                   "not graded (grader skipped: nothing to judge)" if r and int(r["unscored"]) else
                   "all runs lost to provider errors" if r and int(r["lost_provider"]) else "no scored runs")
            ax.text(1, yy, f"{NAME[model]}: {why}", va="center", fontsize=8, color=MUTED)
            continue
        p, lo, hi = float(r["rate"]) * 100, float(r["ci_low"]) * 100, float(r["ci_high"]) * 100
        ax.plot([lo, hi], [yy, yy], color=COL[model], lw=2, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.scatter([p], [yy], s=64, color=COL[model], edgecolor="white", linewidth=2, zorder=3)
        ax.text(min(hi, 100) + 1.5 if p < 75 else lo - 1.5, yy, f"{r['k']}/{r['n']}", va="center",
                ha="left" if p < 75 else "right", fontsize=8, color=INK)
    ax.text(-2, y, f"{env}\n", ha="right", va="center", fontsize=9.5, color=INK)
    ax.text(-2, y - 0.08, behaviour, ha="right", va="top", fontsize=8, color=MUTED)
ax.set_xlim(0, 105); ax.set_ylim(-0.7, len(envs) - 0.3); ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=8, color=COL[m]) for m in COL],
          labels=[NAME[m] for m in COL], frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
title = sys.argv[1] if len(sys.argv) > 1 else "Share of runs showing each environment's bad behaviour"
ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=24)
fig.text(0.01, -0.01, "Dot = share of finished runs; line = 95% Wilson interval; label = runs with the behaviour / runs scored. "
         "Modal pass (530 runs, Qwen and gemma pinned to DeepInfra fp8). Judge: Claude Opus 5, Opus 4.8 where Opus 5 refused.", fontsize=8, color=MUTED)
out = HERE / "charts" / "full_runs.png"
fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
print(out)
