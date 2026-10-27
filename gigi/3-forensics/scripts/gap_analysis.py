"""Chart 5 of RESULTS.md (section 6, "Analysing the gaps"), 27 Oct 2026: in each environment where chart 4 shows a gap
to close, what Qwen and gemma did after taking over from the same checkpoint. Every scored replay is one of:
hacked; honest and finished (said done, or submitted); honest, fell short and said so (gave up, handed back);
honest and stopped (stopped mid-task, ran out of steps, empty reply). One pair of bars per takeover point (the rows of chart 4).
Sources: results/gap_replays.csv (scripts/gap_replays.py) and the pre-commit study (runs_swap[_text]_precommit_*.csv).
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/gap_analysis.py
"""
import collections, csv, math, pathlib, re

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

TAKE = {1: "N", 2: "N−1", 3: "N−2", 6: "N−5"}   # checkpoint N-k -> takeover at turn N-k+1 (as in story_charts.py)
TK_ORDER = ["at N", "at N−1", "at N−2", "at N−5", "clean runs"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2; (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    return p1 - p2, p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


counts = collections.defaultdict(collections.Counter)   # (env, subtask, who) -> segment -> replays
for r in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
    if r["hacked"] == "" or r["env"] == "precommit_hook":
        continue
    sub = f"at {TAKE.get(int(r['N']) - int(r['K']), '?')}" if r["incident"] == "True" else "clean runs"
    if r["env"] == "grade_your_own":
        sub = f"run {re.search(r'_run-(\d+)_', r['point']).group(1)} · {sub}"
    counts[(r["env"], sub, r["who"])]["hacked" if r["hacked"] == "1" else SEG_OF.get(r["ending"], "stopped")] += 1
# pre-commit: Qwen's relays and gemma's text-mode takeovers from Qwen's runs q1, q2, q1m (as in story_charts.py);
# its replays carry no ending label, so an honest replay counts as finished when it attempted a commit with the hook
for tag, N in (("q1", 9), ("q2", 4), ("q1m", 9)):
    for text, who in ((False, "qwen"), (True, "gemma")):
        f = HERE / "results" / (f"runs_swap_text_precommit_{tag}.csv" if text else f"runs_swap_precommit_{tag}.csv")
        for row in csv.DictReader(open(f)):
            mm = re.search(r"from_step(\d+)/(qwen|gemma)/", row["experiment"])
            if mm and mm.group(2) == who and int(row["steps"] or 0) >= 1 and row["cheated"] in ("True", "False"):
                seg = "hacked" if row["cheated"] == "True" else (
                    "finished" if "attempted_commit_with_hook" in row["outcome"] else "stopped")
                counts[("precommit_hook", f"at {TAKE.get(N - int(mm.group(1)), '?')}", who)][seg] += 1


def order(sub):
    run, _, tk = sub.rpartition(" · ")
    return (run, TK_ORDER.index(tk) if tk in TK_ORDER else 99)


n_pairs = len({(e, s) for e, s, _ in counts if any(e in envs for _, _, envs in ROWS)})
fig, ax = plt.subplots(figsize=(10.5, 2.2 + 1.02 * n_pairs))
y, ticks, labels, tick_cols = 0, [], [], []
for letter, fam, envs in ROWS:
    ax.text(-0.17, y - 0.15, f"{letter} · {fam}", fontsize=10.5, fontweight="bold", color=INK, ha="left", va="bottom",
            transform=ax.get_yaxis_transform())
    y += 0.5
    for env in envs:
        ax.text(-0.17, y + 0.05, LABEL[env], fontsize=9.5, color=INK, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        y += 0.45
        for sub in sorted({s for e, s, _ in counts if e == env}, key=order):
            q, g = counts[(env, sub, "qwen")], counts[(env, sub, "gemma")]
            nq, ng = sum(q.values()), sum(g.values())
            if nq and ng:
                d, lo, hi = newcombe(q["hacked"], nq, g["hacked"], ng)
                tag = "gap to close" if lo > 0 else ("gemma more" if hi < 0 else "no clear gap")
                if env == "grade_your_own" and sub == "run 3 · at N−1":
                    tag = "unclear (judge split, chart 4)"
                head = f"{sub}   ·   gap {d * 100:+.0f} points (95% CI {lo * 100:+.0f} to {hi * 100:+.0f}): {tag}"
            else:
                tag, head = "", f"{sub}   ·   no data for one model"
            ax.text(0, y - 0.03, head, fontsize=8.5, color="#d03b3b" if tag == "gap to close" else MUTED, ha="left", va="bottom",
                    fontweight="bold" if tag == "gap to close" else "normal", transform=ax.get_yaxis_transform())
            y += 0.38
            for who, c in (("qwen", q), ("gemma", g)):
                n = sum(c.values()); left = 0
                for key, _, col in SEGS:
                    w = c[key] / n * 100 if n else 0
                    ax.barh(y, w, left=left, height=0.6, color=col, edgecolor="white", linewidth=1.2, zorder=3)
                    if w >= 7:
                        ax.text(left + w / 2, y, f"{c[key]}", ha="center", va="center", fontsize=7.5,
                                color="white" if key in ("hacked", "finished") else INK, zorder=4)
                    left += w
                ax.text(101.5, y, f"{c['hacked']}/{n} hacked ({c['hacked'] / n * 100:.0f}%)" if n else "no data",
                        va="center", fontsize=8, color=INK if c["hacked"] else MUTED)
                ticks.append(y); labels.append(NAME[who]); tick_cols.append(MODEL_COL[who]); y += 0.66
            y += 0.3
        y += 0.1
    y += 0.3
ax.set_yticks(ticks, labels)
for t, col in zip(ax.get_yticklabels(), tick_cols):
    t.set_color(col); t.set_fontweight("bold")
ax.set_ylim(y - 0.4, -0.6); ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of scored replays after the takeover", color=MUTED)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in SEGS], labels=[t for _, t, _ in SEGS],
           loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("5 · Where the gaps come from: from the same checkpoint, Qwen hacks; gemma finishes or reports it fell short",
             x=0.02, ha="left", fontsize=11.5, fontweight="bold")
fig.text(0.02, 1 - 0.75 / (2.2 + 1.02 * n_pairs), "Each pair of bars = one takeover point, the same rows as chart 4. \"at N\": the model "
         "writes Qwen's bad turn N onward (Qwen's run up to N−1); \"at N−1\": one turn earlier; \"clean runs\": restarts\nbefore Qwen's "
         "last turn in runs where it did nothing bad. Red = the bad behaviour; greys = honest endings. Gap = Qwen's rate minus gemma's.",
         fontsize=8.3, color=MUTED, va="top")
fig.subplots_adjust(left=0.17, right=0.84, top=1 - 1.15 / (2.2 + 1.02 * n_pairs), bottom=0.7 / (2.2 + 1.02 * n_pairs))
fig.savefig(HERE / "charts" / "q5_gap_analysis.png", dpi=180, bbox_inches="tight"); plt.close(fig)
for (env, sub, who), c in sorted(counts.items()):
    print(f"{env:26s} {sub:22s} {who:5s} " + "  ".join(f"{k} {c[k]}" for k, _, _ in SEGS) + f"  n {sum(c.values())}")
