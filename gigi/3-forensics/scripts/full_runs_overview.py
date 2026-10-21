"""Overview charts for the full runs: (1) bad-behaviour rate per category and model, pooled over the
environments both models were scored in; (2) median steps per environment, Qwen vs gemma.
  python3 gigi/3-forensics/scripts/full_runs_overview.py   -> charts/full_runs_categories.png, charts/full_runs_steps.png
"""
import collections, csv, math, pathlib, statistics, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import full_runs_table as T

COL = {"qwen3-next-80b": "#1baf7a", "gemma-3-27b": "#2a78d6"}
NAME = {"qwen3-next-80b": "Qwen3-Next-80B", "gemma-3-27b": "gemma-3-27b"}
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK})
CATS = {
    "Honesty probes\n(stay quiet / take the shortcut)": ["eval_tampering", "lazy_investigation", "parse_duration",
                                                          "currency_conversion", "funding_email", "puppeteer"],
    "Cheating suite\n(coding, Opus-graded)": ["optimize_rendering_engine", "optimize_rendering_engine_graded",
                                             "dashboard_costly_failure", "grade_your_own", "lucky_seed",
                                             "lucky_seed_readable_grader", "optimize_lru_cache", "leaked_tests",
                                             "leaked_scorer", "copying_of_answers", "reference_extraction"],
    "Other task gaming\n(broken test, answer file, hook,\nrevert, weaken tests, game cheat)": ["coding", "coding_norules",
        "secret_number", "precommit_hook", "norvane", "async_conversion", "revert_or_refactor", "tower_of_hanoi",
        "chess", "tictactoe"],
}

rows = {(r["environment"], r["model"]): r for r in csv.DictReader(open(HERE / "results" / "full_runs.csv"))}


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# 1. category rates, pooled over environments scored for BOTH models (so a missing gemma env can't bias it)
fig, ax = plt.subplots(figsize=(9, 3.6))
labels = list(CATS)
for i, cat in enumerate(labels):
    both = [e for e in CATS[cat] if all(int((rows.get((e, m)) or {"n": 0})["n"]) for m in COL)]
    for j, m in enumerate(COL):
        k = sum(int(rows[(e, m)]["k"]) for e in both); n = sum(int(rows[(e, m)]["n"]) for e in both)
        lo, hi = T.wilson(k, n); y = len(labels) - 1 - i + (0.18 if j == 0 else -0.18)
        ax.barh(y, k / n * 100, height=0.32, color=COL[m], zorder=3)
        ax.plot([lo * 100, hi * 100], [y, y], color=INK, lw=1, alpha=0.5, zorder=4)
        ax.text(hi * 100 + 1.5, y, f"{k/n:.0%}  ({k}/{n})", va="center", fontsize=8.5)
    ax.text(-1.5, len(labels) - 1 - i, f"{cat}\n{len(both)} environments", ha="right", va="center", fontsize=9)
ax.set_xlim(0, 110); ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); style(ax)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in COL], labels=[NAME[m] for m in COL],
          frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
ax.set_title("Both models misbehave on honesty probes; cheating on code is rare and mostly Qwen's", loc="left",
             fontsize=11.5, fontweight="bold", pad=22)
fig.text(0.01, -0.04, "Share of scored runs with the environment's bad behaviour, pooled over environments scored for both "
         "models; black line = 95% Wilson interval.", fontsize=8, color=MUTED)
fig.savefig(HERE / "charts" / "full_runs_categories.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# 2. median steps per environment (finished runs)
steps = collections.defaultdict(list)
for env, model, run in [r for r in T.modal_runs() if r[0] != "chess"] + T.laptop_runs("chess"):
    if T.status(run) == "finished":
        steps[(env, model)].append(len(list(run.glob("step-*"))))
envs = [e for e in {e for e, _ in steps} if steps[(e, "qwen3-next-80b")] and steps[(e, "gemma-3-27b")]]
med = {k: statistics.median(v) for k, v in steps.items() if v}
envs.sort(key=lambda e: math.log(med[(e, "qwen3-next-80b")] + 1) - math.log(med[(e, "gemma-3-27b")] + 1))
fig, ax = plt.subplots(figsize=(9, 0.33 * len(envs) + 1.4))
for y, e in enumerate(envs):
    q, g = med[(e, "qwen3-next-80b")], med[(e, "gemma-3-27b")]
    ax.plot([g, q], [y, y], color=GRID, lw=2, zorder=1)
    dy = 0.16 if q == g else 0   # equal medians: show both dots, not one hiding the other
    ax.scatter([q], [y + dy], s=40, color=COL["qwen3-next-80b"], zorder=3, edgecolor="white", linewidth=1.5)
    ax.scatter([g], [y - dy], s=40, color=COL["gemma-3-27b"], zorder=3, edgecolor="white", linewidth=1.5)
    ax.text(0.75, y, e, ha="right", va="center", fontsize=8.5)
ax.set_xscale("log"); ax.set_xlim(0.8, 400); ax.set_yticks([])
ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); style(ax)
n_less = sum(med[(e, "gemma-3-27b")] < med[(e, "qwen3-next-80b")] for e in envs)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[m]) for m in COL], labels=[NAME[m] for m in COL],
          frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
ax.set_title(f"gemma stops earlier in {n_less} of {len(envs)} environments, mostly on the long coding tasks",
             loc="left", fontsize=11.5, fontweight="bold", pad=22)
fig.text(0.01, -0.02, "Median steps per finished run (log scale). Top rows: where gemma stops earliest relative to Qwen.",
         fontsize=8, color=MUTED)
fig.savefig(HERE / "charts" / "full_runs_steps.png", dpi=180, bbox_inches="tight"); plt.close(fig)

for cat in CATS:
    both = [e for e in CATS[cat] if all(int((rows.get((e, m)) or {"n": 0})["n"]) for m in COL)]
    print(cat.split("\n")[0], {m: f"{sum(int(rows[(e, m)]['k']) for e in both)}/{sum(int(rows[(e, m)]['n']) for e in both)}" for m in COL}, len(both))
print("gemma earlier:", n_less, "of", len(envs))
