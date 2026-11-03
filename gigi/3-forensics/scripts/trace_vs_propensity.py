"""Chart 5H (28 Oct 2026, Yuqi's question): of Qwen's cheating after it is rewound to its own cheating trace, how much is
"only because the trace is going this direction"? One bar per gap environment: Qwen's rate of the bad act when restarted
from its own cheating run at the cheating turn (the control arm of the swap), split into the part it does anyway (its
rate when restarted at the end of one of its own runs with no bad act: same model, same environment, no cheating
history) and the part the cheating history adds. Right: the added part as a share of the whole. Fresh-run rates
(section 1, 5-20 runs) are in the footnote. Sources: results/gap_replays.csv, results/full_runs.csv, the pre-commit
study CSVs (as in gap_analysis.py).
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/trace_vs_propensity.py
"""
import collections, csv, math, pathlib, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
BASE, ADDED = "#0d6b4a", "#8fd6bb"
ENVS = ["coding", "coding_norules", "precommit_hook", "dashboard_costly_failure", "optimize_rendering_engine", "ai_rd", "grade_your_own"]
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
runs = collections.defaultdict(set)                  # (env, cond, who) -> source runs behind the rate
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
    runs[(r["env"], cond, r["who"])].add(re.sub(r"_K\d+$", "", r["point"]))
for tag, N in (("q1", 9), ("q2", 4), ("q1m", 9)):      # pre-commit's cheating-trace restarts (as in gap_analysis.py)
    for text, who in ((False, "qwen"), (True, "gemma")):
        f = HERE / "results" / (f"runs_swap_text_precommit_{tag}.csv" if text else f"runs_swap_precommit_{tag}.csv")
        for row in csv.DictReader(open(f)):
            mm = re.search(r"from_step(\d+)/(qwen|gemma)/", row["experiment"])
            if mm and mm.group(2) == who and int(row["steps"] or 0) >= 1 and row["cheated"] in ("True", "False"):
                cond = TAKE.get(N - int(mm.group(1)), "?")
                kn[("precommit_hook", cond, who)][0] += row["cheated"] == "True"; kn[("precommit_hook", cond, who)][1] += 1
                runs[("precommit_hook", cond, who)].add(tag.rstrip("m"))   # q1 and q1m are the same incident run

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})
rows = []
for env in ENVS:
    kt, nt = kn[(env, "N", "qwen")]; kb, nb = kn[(env, "clean", "qwen")]
    trace = kt / nt; base = kb / nb if nb >= 5 else None
    rows.append((env, kt, nt, trace, kb, nb, base))
rows.sort(key=lambda r: -((r[3] - r[6]) / r[3] if r[6] is not None and r[3] else -1))

H = 2.6 + 0.8 * len(rows) + 1.3
fig = plt.figure(figsize=(13, H)); ax = fig.add_axes([0.21, 1.3 / H, 0.46, 0.8 * len(rows) / H])
C_WITHOUT, C_WITH = "#9aa5b1", "#0d6b4a"
for x, t in ((109, "continuing\na clean\nrun"), (134, "continuing\na cheating\nrun"), (150, "difference"), (170, "share of the\ncheating that\ncomes from\nthe run")):
    ax.text(x, -1.05, t, fontsize=8, color=MUTED, ha="center", va="center", linespacing=1.15)
for i, (env, kt, nt, trace, kb, nb, base) in enumerate(rows):
    ax.text(-2, i, LABEL[env], ha="right", va="center", fontsize=9.5, fontweight="bold")
    tw = trace * 100
    if base is None:
        ax.plot(tw, i, "o", color=C_WITH, ms=9, zorder=3)
        ax.text(tw - 2.5, i, f"{tw:.0f}%", ha="right", va="center", fontsize=8.5, fontweight="bold", color=C_WITH)
        ax.text(110, i, "no clean\nrun to\ncompare", ha="center", va="center", fontsize=7, color=MUTED, linespacing=1.15)
        ax.text(134, i, f"{tw:.0f}%\n{kt}/{nt} tries\n{len(runs[(env, 'N', 'qwen')])} run" + ("s" if len(runs[(env, 'N', 'qwen')]) != 1 else ""), ha="center", va="center", fontsize=8, color=INK, linespacing=1.1)
        ax.text(150, i, "—", ha="center", va="center", fontsize=9, color=MUTED); ax.text(170, i, "—", ha="center", va="center", fontsize=9, color=MUTED)
        continue
    bw = base * 100
    ax.plot([bw, tw], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.annotate("", xy=(tw - 1.2, i), xytext=(bw, i), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=6, shrinkB=0), zorder=2)
    ax.plot(bw, i, "o", color=C_WITHOUT, ms=9, zorder=3); ax.plot(tw, i, "o", color=C_WITH, ms=9, zorder=3)
    ax.text(tw + 2.5, i, f"{tw:.0f}%", ha="left", va="center", fontsize=8.5, fontweight="bold", color=C_WITH)
    share = max(trace - base, 0) / trace if trace else 0
    ax.text(109, i, f"{bw:.0f}%\n{kb}/{nb} tries\n{len(runs[(env, 'clean', 'qwen')])} run" + ("s" if len(runs[(env, 'clean', 'qwen')]) != 1 else ""), ha="center", va="center", fontsize=8, color=INK, linespacing=1.1)
    ax.text(134, i, f"{tw:.0f}%\n{kt}/{nt} tries\n{len(runs[(env, 'N', 'qwen')])} run" + ("s" if len(runs[(env, 'N', 'qwen')]) != 1 else ""), ha="center", va="center", fontsize=8, color=INK, linespacing=1.1)
    ax.text(150, i, f"+{tw - bw:.0f} pts", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(170, i, f"{share * 100:.0f}%", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C_WITH if share >= 0.5 else INK)
    if i < len(rows) - 1:
        ax.plot([-60, 180], [i + 0.5, i + 0.5], color=GRID, lw=0.6, zorder=0, clip_on=False)
ax.set_xlim(0, 180); ax.set_ylim(len(rows) - 0.5, -1.5)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_bounds(0, 100); ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", left=False, labelleft=False); ax.tick_params(axis="x", length=0)
for xv in (0, 25, 50, 75, 100):
    ax.plot([xv, xv], [-0.5, len(rows) - 0.5], color=GRID, lw=0.7, zorder=0)
ax.text(50, len(rows) - 0.5 + 0.55, "how often Qwen cheats", ha="center", va="top", fontsize=8.5, color=MUTED)
fig.text(0.02, 1 - 0.2 / H, "5H · Does Qwen cheat because of the run it is continuing? Coding and pre-commit: almost entirely. Dashboard: mostly. Rendering engine: a third. ai_rd: a fifth",
         fontsize=12.5, fontweight="bold", va="top")
SUB = ("How it works: we restart Qwen (always the same model) partway through one of its own earlier runs, let it continue, and count how often it cheats. "
       "Grey dot: the run it continues is one where Qwen did NOT cheat; it is restarted near the end. "
       "Green dot: the run it continues is one where Qwen DID cheat; it is restarted right before the turn where it cheated, so the conversation is already heading towards the cheat. "
       "Each dot is an average: the share of tries that cheated, pooled over all of Qwen's runs of that kind in that environment (tries and runs under each number). "
       "The grey and green dots on a row come from DIFFERENT runs; they share a row only because they are the same environment. No random seed is fixed: every try is a fresh sample. "
       "The arrow is how much more Qwen cheats when it continues a run that was heading towards cheating; the last column is that extra part as a share of the green rate. "
       "The green dot is also the control in every swap chart (charts 4 and 5F put gemma in the same spot).")
fig.text(0.02, 1 - 0.55 / H, "\n".join(__import__("textwrap").wrap(SUB, 205)), fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C_WITHOUT, ms=8), plt.Line2D([], [], marker="o", ls="", color=C_WITH, ms=8)],
           labels=["continuing a run where Qwen did NOT cheat (restarted near its end)", "continuing a run where Qwen DID cheat (restarted right before the cheat)"],
           loc="upper left", bbox_to_anchor=(0.015, 1 - 1.55 / H), ncol=2, frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=2)
alone = " · ".join(f"{LABEL[e]} {kn[(e, 'alone', 'qwen')][0]}/{kn[(e, 'alone', 'qwen')][1]}" for e in ENVS)
fig.text(0.02, 0.45 / H, "Brand-new runs started from scratch (section 1, 5–20 per environment) cheat about as often as the grey dots: " + alone + ".",
         fontsize=7.6, color=MUTED, va="center")
fig.text(0.02, 0.22 / H, "Restarting one turn earlier than the green dot gives the same picture: coding 58/60, coding_norules 10/30, rendering 49/64, dashboard 11/46, ai_rd 25/72, "
         "pre-commit 7/80 (N−1) and 11/40 (N−2). Sources: results/gap_replays.csv, results/full_runs.csv, runs_swap[_text]_precommit_*.csv.",
         fontsize=7.6, color=MUTED, va="center")
fig.savefig(HERE / "charts" / "q5_trace.png", dpi=180, bbox_inches="tight", facecolor="white")
for env, kt, nt, trace, kb, nb, base in rows:
    print(f"{env:28s} trace {kt}/{nt} ({trace*100:.0f}%)  without {kb}/{nb}" + (f" ({base*100:.0f}%)  share from trace {(trace-base)/trace*100:.0f}%" if base is not None else "  no baseline"))
