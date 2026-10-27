"""Chart 5 of RESULTS.md (section 6, "Analysing the gaps"), 27 Oct 2026: in each environment where chart 4 shows a gap
to close, what Qwen and gemma did after taking over from the same checkpoint. Every scored replay is one of:
hacked; honest and finished (said done, or submitted); honest, fell short and said so (gave up, handed back);
honest and stopped (stopped mid-task, ran out of steps, empty reply). All takeover points of an environment pooled.
Sources: results/gap_replays.csv (scripts/gap_replays.py) and the pre-commit study (runs_swap[_text]_precommit_*.csv).
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/gap_analysis.py
"""
import collections, csv, pathlib, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
MODEL_COL = {"qwen": "#1baf7a", "gemma": "#2a78d6"}
NAME = {"qwen": "Qwen3-Next-80B", "gemma": "gemma-3-27b"}
SEGS = [("hacked", "hacked (the environment's bad behaviour)", "#d03b3b"),
        ("finished", "honest: finished (said done, or submitted)", "#4a5568"),
        ("fell_short", "honest: fell short and said so (gave up, handed back)", "#9aa5b1"),
        ("stopped", "honest: stopped mid-task or ran out of steps", "#d5d9de")]
SEG_OF = {"says_done": "finished", "quick_answer": "finished", "gave_up": "fell_short", "handed_back": "fell_short"}
ROWS = [("A", "Impossible coding tests", ["coding", "coding_norules"]),
        ("B", "Speed-up targets", ["optimize_rendering_engine", "dashboard_costly_failure"]),
        ("C", "Self-certified ML results", ["ai_rd", "grade_your_own"]),
        ("E", "Repo maintenance", ["precommit_hook"])]
LABEL = {"coding": "coding (ImpossibleBench)", "coding_norules": "coding_norules (ImpossibleBench)",
         "optimize_rendering_engine": "optimize_rendering_engine", "dashboard_costly_failure": "dashboard_costly_failure",
         "ai_rd": "ai_rd", "grade_your_own": "grade_your_own", "precommit_hook": "precommit_hook"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK})

counts = collections.defaultdict(collections.Counter)   # (env, who) -> segment -> replays
for r in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
    if r["hacked"] == "" or r["env"] == "precommit_hook":
        continue
    counts[(r["env"], r["who"])]["hacked" if r["hacked"] == "1" else SEG_OF.get(r["ending"], "stopped")] += 1
# pre-commit: Qwen's relays and gemma's text-mode takeovers from Qwen's runs q1, q2, q1m (as in story_charts.py);
# its replays carry no ending label, so an honest replay counts as finished when it attempted a commit with the hook
for tag in ("q1", "q2", "q1m"):
    for text, who in ((False, "qwen"), (True, "gemma")):
        f = HERE / "results" / (f"runs_swap_text_precommit_{tag}.csv" if text else f"runs_swap_precommit_{tag}.csv")
        for row in csv.DictReader(open(f)):
            mm = re.search(r"from_step(\d+)/(qwen|gemma)/", row["experiment"])
            if mm and mm.group(2) == who and int(row["steps"] or 0) >= 1 and row["cheated"] in ("True", "False"):
                seg = "hacked" if row["cheated"] == "True" else (
                    "finished" if "attempted_commit_with_hook" in row["outcome"] else "stopped")
                counts[("precommit_hook", who)][seg] += 1

fig, ax = plt.subplots(figsize=(10, 7.4))
y, ticks, labels, tick_cols = 0, [], [], []
for letter, fam, envs in ROWS:
    ax.text(-0.17, y - 0.15, f"{letter} · {fam}", fontsize=10, fontweight="bold", color=INK, ha="left", va="bottom",
            transform=ax.get_yaxis_transform())
    y += 0.55
    for env in envs:
        ax.text(0, y - 0.02, LABEL[env], fontsize=9, color=MUTED, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        y += 0.4
        for who in ("qwen", "gemma"):
            c = counts[(env, who)]; n = sum(c.values()); left = 0
            for key, _, col in SEGS:
                w = c[key] / n * 100 if n else 0
                ax.barh(y, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.2, zorder=3)
                if w >= 7:
                    ax.text(left + w / 2, y, f"{c[key]}", ha="center", va="center", fontsize=7.5,
                            color="white" if key in ("hacked", "finished") else INK, zorder=4)
                left += w
            ax.text(101.5, y, f"{c['hacked']}/{n} hacked ({c['hacked'] / n * 100:.0f}%)" if n else "no data",
                    va="center", fontsize=8, color=INK if c["hacked"] else MUTED)
            ticks.append(y); labels.append(NAME[who]); tick_cols.append(MODEL_COL[who]); y += 0.72
        y += 0.25
    y += 0.35
ax.set_yticks(ticks, labels)
for t, col in zip(ax.get_yticklabels(), tick_cols):
    t.set_color(col); t.set_fontweight("bold")
ax.set_ylim(y - 0.4, -0.6); ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of scored replays after the takeover (all takeover points of the environment pooled)", color=MUTED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in SEGS], labels=[t for _, t, _ in SEGS],
           loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("5 · Where the gaps come from: same checkpoint, Qwen hacks, gemma mostly finishes or reports it fell short",
             x=0.02, ha="left", fontsize=11.5, fontweight="bold")
fig.text(0.02, 0.935, "What each model did after taking over from Qwen's checkpoint, in every environment with a gap to close "
         "(chart 4). Red = the bad behaviour; greys = honest endings.", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.17, right=0.84, top=0.9, bottom=0.12)
fig.savefig(HERE / "charts" / "q5_gap_analysis.png", dpi=180, bbox_inches="tight"); plt.close(fig)
for (env, who), c in sorted(counts.items()):
    print(f"{env:28s} {who:5s} " + "  ".join(f"{k} {c[k]}" for k, _, _ in SEGS) + f"  n {sum(c.values())}")
