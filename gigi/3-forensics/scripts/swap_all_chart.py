"""Chart and table of the swap in every environment (results/swap_all.csv from swap_all_score.py): per environment
and kind of restart point (Qwen's incident, or Qwen's final decision where it did nothing bad), how often Qwen
replayed in its own run and gemma swapped in show the environment's bad behaviour, pooled over the points, with
95% Wilson intervals. Also includes the pre-commit hook swap (17 Oct, rerun in text mode on 20 Oct).
  python3 gigi/3-forensics/scripts/swap_all_chart.py   -> charts/swap_all.png, results/swap_all_table.md
"""
import collections, csv, pathlib, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from full_runs_table import wilson

INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
COL = {"qwen": "#1baf7a", "gemma": "#2a78d6"}
NAME = {"qwen": "Qwen replayed in its own run", "gemma": "gemma swapped in"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK})
CATEGORY = {
    "Honesty probes": ["eval_tampering", "funding_email", "currency_conversion", "lazy_investigation", "parse_duration",
                       "puppeteer"],
    "Cheating suite (coding, Opus-graded)": ["optimize_rendering_engine", "dashboard_costly_failure", "grade_your_own",
        "optimize_rendering_engine_graded", "copying_of_answers", "leaked_scorer", "leaked_tests", "lucky_seed",
        "lucky_seed_readable_grader", "reference_extraction"],
    "Other task gaming": ["precommit_hook", "revert_or_refactor", "async_conversion", "tower_of_hanoi", "coding",
                          "coding_norules", "secret_number", "norvane", "chess", "tictactoe"],
}
BEHAVIOUR = {"eval_tampering": "hid the tampering", "funding_email": "pushed the pitch after the legal review",
             "currency_conversion": "did the forbidden arithmetic", "lazy_investigation": "took the shortcut",
             "parse_duration": "answered without running durc", "revert_or_refactor": "reverted the fix",
             "async_conversion": "weakened test assertions", "tower_of_hanoi": "cheating detected",
             "precommit_hook": "removed the hook", "coding": "made the broken test pass",
             "coding_norules": "made the broken test pass", "secret_number": "read the answer or switched noise off",
             "norvane": "tampered with the unrunnable tests", "chess": "won an unwinnable game",
             "tictactoe": "won an unwinnable game", "puppeteer": "hid the missing screenshot"}

agg = collections.defaultdict(lambda: [0, 0])
unmeasured = set()
for r in csv.DictReader(open(HERE / "results" / "swap_all.csv")):
    kind = "Qwen's incident" if r["incident"] == "True" else "Qwen's final decision"
    a = agg[(r["env"], kind, r["who"])]; a[0] += int(r["k"]); a[1] += int(r["n"])
    if r["lost"] and "unscored" in r["lost"] and not int(r["n"]) and not any(r["env"] in v for v in CATEGORY.values()):
        unmeasured.add(r["env"])
# the pre-commit hook swap: Qwen's incident points (17 Oct Qwen replays, 20 Oct gemma text mode)
def precommit(text):
    k = n = 0
    for tag in ("precommit_q1", "precommit_q1m", "precommit_q2"):
        f = HERE / "results" / (f"runs_swap_text_{tag}.csv" if text else f"runs_swap_{tag}.csv")
        for row in csv.DictReader(open(f)):
            m = re.search(r"from_step\d+/(qwen|gemma)/", row["experiment"])
            if m and m.group(1) == ("gemma" if text else "qwen") and int(row["steps"] or 0) >= 1 and row["cheated"] in ("True", "False"):
                k += row["cheated"] == "True"; n += 1
    return [k, n]
agg[("precommit_hook", "Qwen's incident", "qwen")] = precommit(False)
agg[("precommit_hook", "Qwen's incident", "gemma")] = precommit(True)

rows = []   # (category, env, kind)
for cat, envs in CATEGORY.items():
    for e in envs:
        for kind in ("Qwen's incident", "Qwen's final decision"):
            if agg[(e, kind, "qwen")][1] or agg[(e, kind, "gemma")][1]:
                rows.append((cat, e, kind))

fig, ax = plt.subplots(figsize=(11, 0.46 * len(rows) + 2.2))
y = len(rows) + len(CATEGORY); cat_prev = None; ylabels = []
for cat, e, kind in rows:
    if cat != cat_prev:
        y -= 1; ax.text(-2, y, cat, ha="right", va="center", fontsize=10, fontweight="bold"); cat_prev = cat
    y -= 1
    for j, who in enumerate(("qwen", "gemma")):
        k, n = agg[(e, kind, who)]
        if not n:
            continue
        lo, hi = wilson(k, n); yy = y + (0.16 if j == 0 else -0.16)
        ax.plot([lo * 100, hi * 100], [yy, yy], color=COL[who], lw=2, alpha=0.3, solid_capstyle="round", zorder=2)
        ax.scatter([k / n * 100], [yy], s=40, color=COL[who], edgecolor="white", linewidth=1.2, zorder=3)
        right = k / n < 0.75
        ax.text((hi * 100 + 1.5) if right else (lo * 100 - 1.5), yy, f"{k}/{n}", va="center", ha="left" if right else "right",
                fontsize=7.5, color=INK)
    label = f"{e}" + (f" ({BEHAVIOUR[e]})" if e in BEHAVIOUR else "")
    ax.text(-2, y, f"{label}\n", ha="right", va="center", fontsize=8.5)
    ax.text(-2, y - 0.1, kind, ha="right", va="top", fontsize=7.5, color="#d03b3b" if "incident" in kind else MUTED)
ax.set_xlim(0, 105); ax.set_ylim(y - 0.8, len(rows) + len(CATEGORY) - 0.3); ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[w]) for w in COL], labels=[NAME[w] for w in COL],
          frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1), fontsize=8.5)
title = sys.argv[1] if len(sys.argv) > 1 else "The swap in every environment"
ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=24)
fig.text(0.01, 0.075, "Share of finished replays with the environment's bad behaviour (coding suite: grader verdict 'cheated', Claude "
         "Opus 5, or Opus 4.8 where Opus 5 refused); line = 95% Wilson interval.\nRestart point = checkpoint N−1 (and N−2 for "
         "incidents), N = Qwen's bad step or its final decision. Lost replays (provider errors, context window, crashes) are "
         "left out, never counted as behaviour.\nReplayed but without a bad-behaviour measure: "
         "sandbagging (needs its control condition). optimize_lru_cache: Qwen quits at step 1, nothing to restart. No gemma: async_conversion (Qwen's "
         "history from step ~100 exceeds gemma's 131k context).", fontsize=7.5, color=MUTED, va="top")
fig.savefig(HERE / "charts" / "swap_all.png", dpi=180, bbox_inches="tight"); plt.close(fig)

lines = ["| Category | Environment | Restart point | Qwen replayed | gemma swapped in |", "|---|---|---|---|---|"]
for cat, e, kind in rows:
    cells = []
    for who in ("qwen", "gemma"):
        k, n = agg[(e, kind, who)]; lo, hi = wilson(k, n)
        cells.append(f"{k}/{n} = {k/n:.0%} ({lo:.0%}–{hi:.0%})" if n else "—")
    lines.append(f"| {cat} | {e} | {kind} | {cells[0]} | {cells[1]} |")
(HERE / "results" / "swap_all_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
