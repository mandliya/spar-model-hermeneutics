"""Chart 5H (28 Oct 2026, Yuqi's question): how much of Qwen's cheating does the fed-in trace produce? One row per gap
environment; Qwen's rate of the bad act under four starts, light to dark: on its own (section 1, results/full_runs.csv),
restarted at the end of one of its clean runs, restarted from its own cheating trace one turn before the cheat (N-1),
restarted at the cheating turn (N). Hollow blue marker: gemma restarted from the same trace at N. 95% Wilson intervals.
Sources: results/gap_replays.csv, results/full_runs.csv, the pre-commit study CSVs (as in gap_analysis.py).
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/trace_vs_propensity.py
"""
import collections, csv, math, pathlib, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
COND = [("alone", "Qwen on its own (a fresh run)", "#b7e4d3"),
        ("clean", "Qwen restarted at the end of its own clean run", "#6fcaa6"),
        ("N−1", "Qwen restarted from its cheating trace, 1 turn before the cheat", "#1baf7a"),
        ("N", "Qwen restarted from its cheating trace, at the cheating turn", "#0d6b4a")]
GEMMA = "#2a78d6"
ENVS = ["coding", "coding_norules", "optimize_rendering_engine", "dashboard_costly_failure", "ai_rd", "grade_your_own", "precommit_hook"]
LABEL = {"coding": "coding (is_prime)", "coding_norules": "coding_norules (is_prime)", "optimize_rendering_engine": "optimize_rendering_engine",
         "dashboard_costly_failure": "dashboard_costly_failure", "ai_rd": "ai_rd", "grade_your_own": "grade_your_own (run 3)", "precommit_hook": "precommit_hook"}
TAKE = {1: "N", 2: "N−1", 3: "N−2", 6: "N−5"}
MODEL = {"qwen": "qwen3-next-80b", "gemma": "gemma-3-27b"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


kn = collections.defaultdict(lambda: [0, 0])          # (env, cond, who) -> [k, n]
for r in csv.DictReader(open(HERE / "results" / "full_runs.csv")):
    for who, m in MODEL.items():
        if r["model"] == m and r["environment"] in ENVS:
            kn[(r["environment"], "alone", who)] = [int(r["k"]), int(r["n"])]
for r in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
    if r["hacked"] == "" or r["env"] not in ENVS:
        continue
    if r["env"] == "grade_your_own" and "_run-3_" not in r["point"]:
        continue                                        # run 5 has no cheat in either model; run 3 is the false report
    cond = TAKE.get(int(r["N"]) - int(r["K"]), "?") if r["incident"] == "True" else "clean"
    kn[(r["env"], cond, r["who"])][0] += int(r["hacked"]); kn[(r["env"], cond, r["who"])][1] += 1
for tag, N in (("q1", 9), ("q2", 4), ("q1m", 9)):      # pre-commit's cheating-trace restarts (as in gap_analysis.py)
    for text, who in ((False, "qwen"), (True, "gemma")):
        f = HERE / "results" / (f"runs_swap_text_precommit_{tag}.csv" if text else f"runs_swap_precommit_{tag}.csv")
        for row in csv.DictReader(open(f)):
            mm = re.search(r"from_step(\d+)/(qwen|gemma)/", row["experiment"])
            if mm and mm.group(2) == who and int(row["steps"] or 0) >= 1 and row["cheated"] in ("True", "False"):
                cond = TAKE.get(N - int(mm.group(1)), "?")
                kn[("precommit_hook", cond, who)][0] += row["cheated"] == "True"; kn[("precommit_hook", cond, who)][1] += 1

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})
H = 1.9 + 0.95 * len(ENVS) + 0.9
fig = plt.figure(figsize=(12.5, H)); ax = fig.add_axes([0.24, 0.75 / H, 0.5, 0.95 * len(ENVS) / H])
MIN_N = 5
for i, env in enumerate(ENVS):
    ax.text(-3, i, LABEL[env], ha="right", va="center", fontsize=9.5, fontweight="bold")
    ys = {c: i + (j - 1.5) * 0.19 for j, (c, _, _) in enumerate(COND)}
    for c, _, col in COND:
        k, n = kn[(env, c, "qwen")]
        if n < MIN_N:
            ax.text(1, ys[c], f"{k}/{n}, too few" if n else "not run", fontsize=7.3, color=MUTED, va="center")
            continue
        lo, hi = wilson(k, n); p = k / n * 100
        ax.plot([lo * 100, hi * 100], [ys[c], ys[c]], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(p, ys[c], "o", color=col, ms=7.5, zorder=3)
        ax.text(hi * 100 + 1.5, ys[c], f"{k}/{n}", fontsize=7.3, color=MUTED, va="center")
    k, n = kn[(env, "N", "gemma")]
    if n >= MIN_N:
        ax.plot(k / n * 100, ys["N"], "o", mfc="white", mec=GEMMA, mew=1.8, ms=7.5, zorder=4)
        ax.text(k / n * 100, ys["N"] + 0.13, f"gemma {k}/{n}", fontsize=7, color=GEMMA, ha="center", va="top")
    if i < len(ENVS) - 1:
        ax.axhline(i + 0.5, color=GRID, lw=0.7, zorder=0)
ax.set_xlim(0, 115); ax.set_ylim(len(ENVS) - 0.5, -0.5)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", left=False, labelleft=False); ax.tick_params(axis="x", length=0)
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
ax.set_xlabel("share of tries that did the bad act (dot), with its 95% interval (line)", fontsize=8.5, color=MUTED)
fig.text(0.02, 1 - 0.2 / H, "5H · How much of Qwen's cheating does the fed-in trace produce? Coding: nearly all of it. Rendering engine: Qwen cheats from scratch too",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.02, 1 - 0.55 / H,
         "Each row is one environment; the four green dots are Qwen's cheating rate under four different starts, light to dark: a fresh run, the end of its own clean "
         "run, its own cheating trace rewound\nto one turn before the cheat, and the cheating turn itself. The hollow blue dot is gemma-3-27b restarted from the same "
         "trace at the cheating turn. The jump from the light dots to the dark dots is the share\nof the bad act that the history produces; the dark dots themselves "
         "are Qwen's standing propensity when the history has already gone that way.",
         fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
handles = [plt.Line2D([], [], marker="o", ls="", color=col, ms=7) for _, _, col in COND] + \
          [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=GEMMA, mew=1.8, ms=7)]
fig.legend(handles=handles, labels=[t for _, t, _ in COND] + ["gemma-3-27b restarted from the same trace, at the cheating turn"],
           loc="upper left", bbox_to_anchor=(0.015, 1 - 1.15 / H), ncol=2, frameon=False, fontsize=8.2, handlelength=1.2, columnspacing=2)
fig.text(0.02, 0.2 / H, "'Qwen on its own' is the original full runs of section 1 (5–20 per environment; ai_rd's 0/10 predates its scorer fix). Clean-run restarts: the last turn of a "
         "Qwen run with no bad act. grade_your_own: run 3 only (the false report); run 5 has no cheat in either model.\nPre-commit: restarts at N−1 are 7/80 (9%) and at "
         "N−2 11/40 (28%), from the pre-commit study. Sources: results/full_runs.csv, results/gap_replays.csv, runs_swap[_text]_precommit_*.csv.",
         fontsize=7.6, color=MUTED, va="center", linespacing=1.3)
fig.savefig(HERE / "charts" / "q5_trace.png", dpi=180, bbox_inches="tight", facecolor="white")
for env in ENVS:
    print(env, "  ".join(f"{c}: {kn[(env, c, 'qwen')][0]}/{kn[(env, c, 'qwen')][1]}" for c, _, _ in COND), f"  gemma N: {kn[(env, 'N', 'gemma')][0]}/{kn[(env, 'N', 'gemma')][1]}")
