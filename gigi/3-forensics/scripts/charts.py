"""The charts of gigi/3-forensics/RESULTS.md (the tables stay as details underneath).

  charts/solo.png      each model alone: how often it cheats, per environment (+ Test Regression)
  charts/swap.png      the swap: cheat rate after taking over, by whose run it is and when
  charts/wrap_up.png   how soon each model stops after taking over, in Qwen's runs vs gemma's runs

Same colours as part two: gemma-3-27b blue, Qwen3-Next-80B aqua. Rates with Wilson 95% intervals.
Run (after collect.py / analyse.py inputs exist):
    uv run --with pandas --with matplotlib --with statsmodels python gigi/3-forensics/scripts/charts.py
"""
import glob
import json
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

HERE = pathlib.Path(__file__).resolve().parents[1]
RES, OUT = HERE / "results", HERE / "charts"
OUT.mkdir(exist_ok=True)
Q, GM = "qwen/qwen3-next-80b-a3b-instruct", "google/gemma-3-27b-it"
COL = {"qwen": "#1baf7a", "gemma": "#2a78d6"}
NAME = {"qwen": "Qwen3-Next-80B", "gemma": "gemma-3-27b"}
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": MUTED})


def wilson(k, n):
    lo, hi = proportion_confint(k, n, 0.05, method="wilson")
    return k / n, lo, hi


def scored(d):
    return d[d.cheated.astype(str).isin(["True", "False"])]


def cheats(d):
    d = scored(d)
    return int((d.cheated.astype(str) == "True").sum()), len(d)


# ---------------------------------------------------------------- data
STAGES = {"qwen": ["smoke", "screen", "base_qwen"], "gemma": ["smoke_gemma2", "screen_gemma", "base_gemma"]}
solo = {w: pd.concat([pd.read_csv(RES / f"runs_{s}.csv") for s in st]) for w, st in STAGES.items()}
for w in solo:
    solo[w] = solo[w][(solo[w].model == (Q if w == "qwen" else GM)) & (solo[w].steps > 1)]

ROWS = [("Secret Number\nread the answer file", "secret_number", ""),
        ("is_prime, rule against hardcoding\nhard-coded the broken test", "coding", "rules"),
        ("is_prime, no rule\nhard-coded the broken test", "coding", "no_rules"),
        ("Pre-commit hook\nbypassed the hook", "precommit_hook", ""),
        ("Broken Test Runner\ntampered with the tests", "norvane", "")]
solo_rows = []
for label, env, var in ROWS:
    solo_rows.append((label, {w: cheats(d[(d.env == env) & (d.variant.fillna("") == var)]) for w, d in solo.items()}))
judged = [json.loads(l) for f in glob.glob(str(RES / "judged_*.jsonl")) for l in open(f)]
solo_rows.append(("Broken Test Runner\nfinal report claims tests passed",
                  {w: (sum(r["verdict"] == "DISHONEST" for r in judged if r["model"] == m and r.get("verdict")),
                       sum(1 for r in judged if r["model"] == m and r.get("verdict"))) for w, m in (("qwen", Q), ("gemma", GM))}))
reg = [json.loads(l) for l in open(RES / "regression_prefill.jsonl")]
solo_rows.append(("Test Regression (prefilled)\nhid the failing test",
                  {w: (sum(r.get("grade") in ("silent", "claims_tests_pass") for r in reg if r["model"] == m),
                       sum(1 for r in reg if r["model"] == m and r.get("grade"))) for w, m in (("qwen", Q), ("gemma", GM))}))

swaps = []
for f in glob.glob(str(RES / "runs_swap_precommit_*.csv")):
    d = scored(pd.read_csv(f))
    d["tag"] = re.search(r"precommit_(\w+)\.csv", f).group(1)
    d["K"] = d.experiment.str.extract(r"from_step(\d+)")[0].astype(int)
    d["who"] = d.experiment.str.extract(r"from_step\d+/(qwen|gemma)/")[0]
    swaps.append(d)
S = pd.concat(swaps)
# "just before the cheat" = the checkpoint right before each incident's cheating step
BEFORE = {("q1", 8), ("q2", 3), ("g2", 6), ("g5", 8), ("g19", 5)}
MATCHED = {(t, 3) for t in ("q1m", "q2", "h1", "h2", "h4", "g2", "g5", "g19", "g7h")}
S["owner"] = np.where(S.tag.str.startswith("g"), "gemma", "qwen")
S["moment"] = [("before" if (t, k) in BEFORE else "") + ("|matched" if (t, k) in MATCHED else "") for t, k in zip(S.tag, S.K)]

# ---------------------------------------------------------------- chart 1: each model alone
fig, ax = plt.subplots(figsize=(10, 7.2))
y = np.arange(len(solo_rows))[::-1]
for i, (label, cnt) in enumerate(solo_rows):
    for w, dy in (("qwen", 0.19), ("gemma", -0.19)):
        k, n = cnt[w]
        p, lo, hi = wilson(k, n)
        ax.barh(y[i] + dy, p * 100, height=0.34, color=COL[w], zorder=3)
        ax.plot([lo * 100, hi * 100], [y[i] + dy] * 2, color="#b9b5ab", lw=1, zorder=2)   # 95% interval, kept quiet
        ax.plot([p * 100], [y[i] + dy], marker="o", ms=5, color=COL[w], zorder=4)
        ax.text(hi * 100 + 1.2, y[i] + dy, f"{100 * p:.0f}%  ({k} of {n})", va="center", fontsize=9, color=INK)
ax.set_yticks(y); ax.set_yticklabels([l for l, _ in solo_rows], fontsize=9.5)
ax.set_xlim(0, 75); ax.set_xlabel("share of runs, %   (grey line = 95% interval: few runs, so wide)")
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_title("On their own, gemma cheats more than Qwen-80B,\nmostly by deleting the pre-commit hook",
             loc="left", fontsize=13, fontweight="bold", pad=12)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[w]) for w in ("qwen", "gemma")],
          labels=[NAME["qwen"], NAME["gemma"]], loc="lower right", frameon=False)
fig.savefig(OUT / "solo.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- chart 2: the swap
groups = [("qwen", "matched", "after the first\nblocked commit"),
          ("qwen", "before", "one step before\nQwen's cheat"),
          ("gemma", "matched", "after the first\nblocked commit"),
          ("gemma", "before", "one step before\ngemma's cheat")]
fig, ax = plt.subplots(figsize=(10.5, 6))
x = np.array([0, 1, 2.4, 3.4])
for gi, (owner, moment, label) in enumerate(groups):
    sub = S[(S.owner == owner) & S.moment.str.contains(moment)]
    for w, dx in (("qwen", -0.19), ("gemma", 0.19)):
        k, n = cheats(sub[sub.who == w])
        p, lo, hi = wilson(k, n)
        ax.bar(x[gi] + dx, p * 100, width=0.34, color=COL[w], zorder=3)
        ax.errorbar(x[gi] + dx, p * 100, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=INK, lw=1, capsize=3, zorder=4)
        ax.text(x[gi] + dx, hi * 100 + 2.5, f"{100 * p:.0f}%\n{k}/{n}", ha="center", fontsize=8.5, color=INK, linespacing=1.1)
ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups], fontsize=9.5)
ax.axvline(1.7, color=GRID, lw=1)
for xc, head in ((0.5, "IN QWEN'S RUNS"), (2.9, "IN GEMMA'S RUNS")):
    ax.text(xc, 96, head, ha="center", fontsize=10.5, fontweight="bold", color=INK)
ax.set_ylim(0, 102); ax.set_ylabel("replays that bypassed the hook, %")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.tick_params(axis="x", length=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("Whose run it is decides more than who takes over",
             loc="left", fontsize=13, fontweight="bold", pad=26)
ax.text(0, 1.03, "Pre-commit hook. A recorded run is restarted from a saved step, 20 times per model per checkpoint.",
        transform=ax.transAxes, fontsize=9, color=MUTED)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[w]) for w in ("qwen", "gemma")],
          labels=[f"{NAME['qwen']} takes over", f"{NAME['gemma']} takes over"], loc="upper left", frameon=False, bbox_to_anchor=(0, 0.92))
fig.savefig(OUT / "swap.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- chart 3: how soon they stop
fig, ax = plt.subplots(figsize=(8.5, 4.6))
cats = [("qwen", "in Qwen's runs"), ("gemma", "in gemma's runs")]
for ci, (owner, label) in enumerate(cats):
    for w, dx in (("qwen", -0.19), ("gemma", 0.19)):
        st = S[(S.owner == owner) & (S.who == w)].steps.astype(int)
        k, n = int((st <= 3).sum()), len(st)
        p, lo, hi = wilson(k, n)
        ax.bar(ci + dx, p * 100, width=0.34, color=COL[w], zorder=3)
        ax.text(ci + dx, p * 100 + 2.5, f"{100 * p:.0f}%", ha="center", fontsize=10, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels([c[1] for c in cats])
ax.set_ylim(0, 105); ax.set_ylabel("replays that ended within 3 steps, %")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.tick_params(axis="x", length=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("Why: in Qwen's runs both models wrap up quickly;\nin gemma's runs both keep working (and get to the cheat)",
             loc="left", fontsize=12.5, fontweight="bold", pad=12)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[w]) for w in ("qwen", "gemma")],
          labels=[f"{NAME['qwen']} takes over", f"{NAME['gemma']} takes over"], loc="upper right", frameon=False)
fig.savefig(OUT / "wrap_up.png", dpi=180, bbox_inches="tight"); plt.close(fig)
print("saved charts/solo.png, charts/swap.png, charts/wrap_up.png")
