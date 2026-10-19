"""Three charts, one per finding of the full runs (19 Oct 2026):
  1. charts/finding_report_not_action.png  the failures are in the report, not the action: offered exploits
     are never taken, while known facts are withheld; the cheats split into false success claims vs code changes
  2. charts/finding_impossible_targets.png the performance tasks: every run's best benchmark against its
     target, by verdict (no run reached the target; some of Qwen's failing runs cheated)
  3. charts/finding_persistence.png        steps per graded run in the cheating suite, cheats marked:
     gemma's runs end before the point where Qwen's cheats happen
Data: results/full_runs.csv, final/grading.json, rollout.log (benchmark output only, never the agent's claims).
"""
import collections, csv, json, pathlib, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import full_runs_table as T

COL = {"qwen3-next-80b": "#1baf7a", "gemma-3-27b": "#2a78d6"}
NAME = {"qwen3-next-80b": "Qwen3-Next-80B", "gemma-3-27b": "gemma-3-27b"}
INK, MUTED, GRID, CHEAT = "#17140f", "#6f6a60", "#e1e0d9", "#d03b3b"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK})
OUT = HERE / "charts"
rows = {(r["environment"], r["model"]): r for r in csv.DictReader(open(HERE / "results" / "full_runs.csv"))}
EXPLOIT = ["leaked_tests", "leaked_scorer", "copying_of_answers", "reference_extraction", "revert_or_refactor",
           "async_conversion", "tower_of_hanoi"]
WITHHOLD = ["eval_tampering", "lazy_investigation", "currency_conversion", "funding_email"]
SUITE = [e for e, v in T.JUDGED.items()]


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def pooled(envs, m):
    k = sum(int(rows[(e, m)]["k"]) for e in envs if (e, m) in rows)
    n = sum(int(rows[(e, m)]["n"]) for e in envs if (e, m) in rows)
    return k, n


# graded cheating-suite runs: (env, model, run, verdict dict or None, steps)
graded = []
for env, model, run in T.modal_runs():
    if env in T.JUDGED and T.status(run) == "finished":
        g = run / "final" / "grading.json"
        d = json.loads(g.read_text()) if g.exists() else None
        if d and str(d.get("evidence", "")).strip(" .") == "":
            d = None
        graded.append((env, model, run, d, len(list(run.glob("step-*")))))
cheats = [x for x in graded if x[3] and x[3].get("verdict") == "cheated"]

# ---- 1. report vs action
FALSE_CLAIM = re.compile(r"fabricat|invented|claims?|declares|report(s|ed)? (a |that )?.{0,40}(achiev|met|gain|success)", re.I)
shape = collections.Counter("False success claim" if FALSE_CLAIM.search(str(d.get("evidence"))) else
                            "Changed the code or baseline" for _, _, _, d, _ in cheats)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.75})
groups = [("Took an offered exploit\n(leaked tests or scorer, copied answers,\nrevert, edit tests, game cheat)", EXPLOIT),
          ("Withheld or misreported\nwhat it knew\n(honesty probes)", WITHHOLD)]
for i, (label, envs) in enumerate(groups):
    for j, m in enumerate(COL):
        k, n = pooled(envs, m); lo, hi = T.wilson(k, n); y = 1 - i + (0.18 if j == 0 else -0.18)
        a1.barh(y, k / n * 100, height=0.32, color=COL[m], zorder=3)
        a1.plot([lo * 100, hi * 100], [y, y], color=INK, lw=1, alpha=0.5, zorder=4)
        a1.text(hi * 100 + 1.5, y, f"{k/n:.0%} ({k}/{n})", va="center", fontsize=8.5)
    a1.text(-1.5, 1 - i, label, ha="right", va="center", fontsize=8.5)
a1.set_xlim(0, 115); a1.set_yticks([]); a1.set_xticks([0, 50, 100]); a1.set_xticklabels(["0%", "50%", "100%"])
a1.grid(axis="x", color=GRID, lw=0.6, zorder=0); style(a1)
a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in COL], labels=[NAME[m] for m in COL],
          frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1), fontsize=8.5)
labels2 = ["False success claim", "Changed the code or baseline"]
by_model = {m: collections.Counter("False success claim" if FALSE_CLAIM.search(str(d.get("evidence"))) else
                                   "Changed the code or baseline" for _, mm, _, d, _ in cheats if mm == m) for m in COL}
left = [0, 0]
for m in COL:
    vals = [by_model[m][l] for l in labels2]
    a2.barh([1, 0], vals, left=left, height=0.5, color=COL[m], edgecolor="white", linewidth=2, zorder=3)
    left = [a + b for a, b in zip(left, vals)]
for y, l in zip([1, 0], labels2):
    a2.text(-0.15, y, l, ha="right", va="center", fontsize=8.5); a2.text(left[1 - y] + 0.15, y, str(left[1 - y]), va="center", fontsize=8.5)
a2.set_yticks([]); a2.set_xlim(0, max(left) + 1.5); a2.set_xticks(range(0, max(left) + 2)); style(a2)
a2.set_title(f"The {len(cheats)} cheats in the coding suite, by shape", loc="left", fontsize=9.5, color=MUTED)
fig.suptitle("The failures are in what the models report, not in exploiting the environment", x=0.01, ha="left",
             fontsize=12, fontweight="bold", y=1.04)
fig.text(0.01, -0.05, "Left: pooled share of scored runs (Modal pass); gemma ran only 2 of the 4 honesty probes. "
         "Right: each cheat's verdict evidence (Opus grader). Black line = 95% Wilson interval.", fontsize=8, color=MUTED)
fig.savefig(OUT / "finding_report_not_action.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# ---- 2. impossible targets: performance tasks, best benchmark vs target
BENCH = re.compile(r"speedup:? (\d+(?:\.\d+)?)x \((?:baseline|median)")
pts = []
for env, model, run, d, steps in graded:
    if run.parts[-4] not in ("dashboard_perf", "stripped_engine"):
        continue
    cfg = (run.parent / "config.yaml").read_text(errors="replace")
    tgt = re.search(r"(\d+(?:\.\d+)?)x speedup", cfg); tgt = float(tgt.group(1)) if tgt else 3.0
    vals = [float(v) for v in BENCH.findall((run / "rollout.log").read_text(errors="replace"))]
    pts.append((model, max(vals) / tgt if vals else None, d.get("verdict") if d else None, env))
fig, ax = plt.subplots(figsize=(9, 3.4))
cats = [("honest", "Judged honest"), ("cheated", "Judged cheated")]
for i, (v, lab) in enumerate(cats):
    sel = [p for p in pts if p[2] == v]
    for j, (m, r, _, _) in enumerate(sel):
        x = r * 100 if r is not None else -12
        y = 1 - i + ((j % 7) - 3) * 0.06
        ax.scatter([x], [y], s=46 if v == "cheated" else 30, color=COL[m], marker="X" if v == "cheated" else "o",
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(-30, 1 - i, f"{lab}\n({len(sel)} runs)", ha="right", va="center", fontsize=9)
ax.axvline(100, color=INK, lw=1, ls="--", zorder=2); ax.text(101, 1.45, "target reached", fontsize=8, color=INK)
ax.axvspan(-20, -4, color=GRID, alpha=0.5, zorder=0); ax.text(-12, 1.45, "no benchmark\never ran", ha="center", fontsize=7.5, color=MUTED)
ax.set_xlim(-20, 120); ax.set_ylim(-0.6, 1.7); ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); style(ax)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[m]) for m in COL], labels=[NAME[m] for m in COL],
          frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1), fontsize=8.5)
reached = sum(1 for p in pts if p[2] == "cheated" and p[1] is not None and p[1] >= 1)
nq = sum(p[0] == "qwen3-next-80b" for p in pts); ng = sum(p[0] == "gemma-3-27b" for p in pts)
cq = sum(p[0] == "qwen3-next-80b" and p[2] == "cheated" for p in pts); cg = sum(p[0] == "gemma-3-27b" and p[2] == "cheated" for p in pts)
hit = sum(1 for p in pts if p[1] is not None and p[1] >= 1)
ax.set_title(f"No run reached the target ({hit} of {len(pts)}); {cq} of Qwen's {nq} runs then cheated, {cg} of gemma's {ng}",
             loc="left", fontsize=11.5, fontweight="bold", pad=22)
fig.text(0.01, -0.04, "Performance tasks (optimize_rendering_engine, dashboard_costly_failure, optimize_rendering_engine_graded): "
         "best speedup the benchmark itself printed, as a share of the prompt's target.", fontsize=8, color=MUTED)
fig.savefig(OUT / "finding_impossible_targets.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# ---- 3. persistence: steps per graded run, cheats marked
fig, ax = plt.subplots(figsize=(9, 2.9))
stat = {}
for i, m in enumerate(COL):
    sel = [x for x in graded if x[1] == m]
    for j, (env, _, run, d, steps) in enumerate(sel):
        y = 1 - i + ((j % 9) - 4) * 0.045
        c = d and d.get("verdict") == "cheated"
        ax.scatter([max(steps, 1)], [y], s=60 if c else 14, color=CHEAT if c else COL[m], marker="X" if c else "o",
                   alpha=1 if c else 0.55, zorder=4 if c else 3, linewidth=0)
    long_ = [x for x in sel if x[4] >= 10]
    stat[m] = (len(long_), len(sel), sum(1 for x in sel if x[3] and x[3].get("verdict") == "cheated"))
    ax.text(0.7, 1 - i, f"{NAME[m]}\n{stat[m][0]}/{stat[m][1]} runs reach 10+ steps", ha="right", va="center", fontsize=8.5)
ax.set_xscale("log"); ax.set_xlim(0.9, 600); ax.set_yticks([]); ax.set_ylim(-0.5, 1.5)
ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); style(ax)
ax.legend(handles=[plt.Line2D([], [], marker="X", ls="", color=CHEAT, ms=8)], labels=["cheated"], frameon=False,
          loc="lower right", bbox_to_anchor=(1, 1), fontsize=8.5)
ax.set_title("gemma's runs mostly end before the point where Qwen's cheats happen", loc="left", fontsize=11.5,
             fontweight="bold", pad=22)
fig.text(0.01, -0.06, "Each dot = one graded run of the coding cheating suite (Modal pass), by the number of steps it lasted (log scale).",
         fontsize=8, color=MUTED)
fig.savefig(OUT / "finding_persistence.png", dpi=180, bbox_inches="tight"); plt.close(fig)

print("shapes:", dict(shape), "by model:", {m: dict(c) for m, c in by_model.items()})
print("perf points:", collections.Counter((p[0][:5], p[2], None if p[1] is None else p[1] >= 1) for p in pts))
print("persistence:", stat, "cheat steps:", sorted((x[1][:5], x[4]) for x in cheats))
for m in COL:
    print(m, "exploit", pooled(EXPLOIT, m), "withhold", pooled(WITHHOLD, m))
