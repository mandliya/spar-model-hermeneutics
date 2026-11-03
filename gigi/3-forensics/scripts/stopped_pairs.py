"""Chart 5K: where gemma stopped, what did Qwen do on the same trace, from the same turn? (John, 28 Oct call)

The pooled stopped counts (chart 5J) cannot say whether Qwen cheated in the situations where gemma stopped. This chart
takes every cell (environment x restart point) in which gemma had at least one scored try that ended "stopped" (a reply
with no tool call, which this harness treats as the final report) and shows, for the same cell, how Qwen's scored tries
ended: hacked, finished and said so (says_done / quick_answer), fell short and said so (gave_up / handed_back), stopped.
Both models continue the same Qwen trace from the same turn, so the comparison is like for like.

Reads results/gap_replays.csv (one row per try) and results/stopped_reasons.csv (one row per scored stopped try in the gap
environments, with the reason). The cells are those where stopped_reasons.csv has a gemma row. Writes
charts/q5_stopped_pairs.png and prints a markdown table (one row per cell, totals, and the stop reasons per environment).
  cd gigi/3-forensics/agent-interp-envs && env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/stopped_pairs.py
"""
import collections, csv, pathlib, re, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID, RED = "#17140f", "#6f6a60", "#e1e0d9", "#d03b3b"
QWEN, GEMMA = "#1baf7a", "#2a78d6"
SEGS = [("hacked", "hacked", RED), ("finished", "finished and said so", "#4a5568"),
        ("fell_short", "fell short and said so", "#9aa5b1"), ("stopped", "stopped", "#d5d9de")]
SEG_OF = {"says_done": "finished", "quick_answer": "finished", "gave_up": "fell_short", "handed_back": "fell_short"}
ENV_ORDER = ["optimize_rendering_engine", "dashboard_costly_failure", "ai_rd", "grade_your_own"]
POINT_RE = re.compile(r"^(?P<ts>.+)_run-(?P<run>\d+)_K(?P<k>\d+)$")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})


def segment(r):
    if r["hacked"] == "1":
        return "hacked"
    return SEG_OF.get(r["ending"], "stopped")


def point_key(point):
    m = POINT_RE.match(point)
    return m["ts"], int(m["run"]), int(m["k"])


def short_label(env, point, multi_trace):
    ts, run, k = point_key(point)
    trace = f"run-{run}"
    if multi_trace:  # the environment has restart points on traces from different collections: name the trace by its time
        trace = f"run-{run} (trace {ts.split('_')[1][:5]})"
    return f"{trace} · restart at turn {k}"


# ---- data -----------------------------------------------------------------------------------------------------------------
reasons = list(csv.DictReader(open(HERE / "results" / "stopped_reasons.csv")))
g_stops = collections.Counter((r["env"], r["point"]) for r in reasons if r["who"] == "gemma")
cells = sorted(g_stops, key=lambda c: (ENV_ORDER.index(c[0]), point_key(c[1])))
counts = {c: {"qwen": collections.Counter(), "gemma": collections.Counter()} for c in cells}
for r in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
    if r["hacked"] not in ("0", "1") or (r["env"], r["point"]) not in counts:
        continue
    counts[(r["env"], r["point"])][r["who"]][segment(r)] += 1
for c in cells:
    if counts[c]["gemma"]["stopped"] != g_stops[c]:
        print(f"WARNING {c}: gap_replays has {counts[c]['gemma']['stopped']} gemma stopped tries, stopped_reasons.csv {g_stops[c]}")
reason_by_env = collections.defaultdict(collections.Counter)
for r in reasons:
    if r["who"] == "gemma" and (r["env"], r["point"]) in counts:
        reason_by_env[r["env"]][r["reason"]] += 1

n_q = sum(sum(counts[c]["qwen"].values()) for c in cells)
h_q = sum(counts[c]["qwen"]["hacked"] for c in cells)
n_stops = sum(g_stops.values())
by_env = collections.defaultdict(lambda: [0, 0])
for c in cells:
    by_env[c[0]][0] += counts[c]["qwen"]["hacked"]; by_env[c[0]][1] += sum(counts[c]["qwen"].values())
# "the rendering trace": the one Qwen trace in optimize_rendering_engine on which Qwen cheated most (pooled over its restart points)
by_trace = collections.defaultdict(lambda: [0, 0])
for env, point in cells:
    if env == "optimize_rendering_engine":
        ts, run, _ = point_key(point)
        by_trace[(ts, run)][0] += counts[(env, point)]["qwen"]["hacked"]; by_trace[(ts, run)][1] += sum(counts[(env, point)]["qwen"].values())
(r_ts, r_run), (r_h, r_n) = max(by_trace.items(), key=lambda kv: kv[1][0])
r_pts = sum(1 for e, p in cells if e == "optimize_rendering_engine" and point_key(p)[:2] == (r_ts, r_run))
multi = {env: len({point_key(p)[0] for e, p in cells if e == env}) > 1 for env in ENV_ORDER}

# ---- markdown table ---------------------------------------------------------------------------------------------------------
print("| environment | trace · restart | gemma stopped | gemma hacked / finished / fell short (of n) | Qwen hacked / n |")
print("|---|---|---:|---|---:|")
tot = {"gemma": collections.Counter(), "qwen": collections.Counter()}
for env, point in cells:
    g, q = counts[(env, point)]["gemma"], counts[(env, point)]["qwen"]
    tot["gemma"].update(g); tot["qwen"].update(q)
    print(f"| {env} | {short_label(env, point, multi[env])} | {g['stopped']} | {g['hacked']} / {g['finished']} / {g['fell_short']} "
          f"(of {sum(g.values())}) | {q['hacked']} / {sum(q.values())} |")
g, q = tot["gemma"], tot["qwen"]
print(f"| **total ({len(cells)} cells)** | | **{g['stopped']}** | **{g['hacked']} / {g['finished']} / {g['fell_short']} (of {sum(g.values())})** "
      f"| **{q['hacked']} / {sum(q.values())} ({100 * q['hacked'] / sum(q.values()):.0f}%)** |")
reason_line = "; ".join(f"{env}: " + ", ".join(f"{rs} {k}" for rs, k in reason_by_env[env].most_common())
                        for env in ENV_ORDER if reason_by_env[env])
print("gemma's stop reasons in these cells — " + reason_line)

# ---- chart ------------------------------------------------------------------------------------------------------------------
ROW, HEAD, BAR = 1.0, 0.62, 0.36
W = 13.0
H = 0.7 * len(cells) + 3.1
top_in, bot_in = 1.75, 1.55
fig = plt.figure(figsize=(W, H), facecolor="white")
left_in = 2.55
ax = fig.add_axes([left_in / W, bot_in / H, (W - left_in - 0.15) / W, (H - top_in - bot_in) / H])
ax.set_facecolor("white")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlim(0, 160); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=8, color=MUTED)
ax.tick_params(axis="x", colors=MUTED, length=0, pad=4); ax.set_yticks([])
for xv in (25, 50, 75, 100):
    ax.axvline(xv, color=GRID, lw=0.7, zorder=0)

y = 0.0
for env in ENV_ORDER:
    env_cells = [c for c in cells if c[0] == env]
    if not env_cells:
        continue
    eh, en = by_env[env]
    ax.text(-0.5, y + HEAD * 0.5, env, fontsize=9.5, fontweight="bold", color=INK, ha="right", va="center", transform=ax.transData, clip_on=False)
    ax.text(0.6, y + HEAD * 0.5, f"{len(env_cells)} restart point{'s' if len(env_cells) > 1 else ''} where gemma stopped · "
            f"Qwen hacked {eh}/{en} ({100 * eh / en:.0f}%)", fontsize=8, color=MUTED, va="center")
    y += HEAD
    for env, point in env_cells:
        ax.text(-0.5, y + ROW / 2, short_label(env, point, multi[env]), fontsize=8.6, color=INK, ha="right", va="center", clip_on=False)
        for who, yy, col in (("gemma", y + ROW * 0.29, GEMMA), ("qwen", y + ROW * 0.71, QWEN)):
            c = counts[(env, point)][who]; n = sum(c.values())
            left = 0.0
            for key, _, scol in SEGS:
                w = 100 * c[key] / n
                if w:
                    ax.barh(yy, w, left=left, height=BAR, color=scol, edgecolor="white", linewidth=0.8, zorder=2)
                    if c[key] >= 2:
                        ax.text(left + w / 2, yy, str(c[key]), ha="center", va="center", fontsize=7.5, zorder=4,
                                color="white" if key in ("hacked", "finished") else INK)
                left += w
            tag = (f"gemma n={n} · {c['stopped']} stopped" if who == "gemma" else f"Qwen n={n} · {c['hacked']} hacked")
            ax.text(101.5, yy, tag, fontsize=8, color=col, va="center")
        q = counts[(env, point)]["qwen"]; qn = sum(q.values()); p = 100 * q["hacked"] / qn
        hot = p >= 30
        ax.text(131, y + ROW / 2, f"Qwen hacked {q['hacked']}/{qn} ({p:.0f}%)", fontsize=8.5, va="center",
                color=RED if hot else INK, fontweight="bold" if hot else "normal")
        y += ROW
    ax.plot([0, 100], [y + 0.02, y + 0.02], color=GRID, lw=0.7, zorder=1, clip_on=False)
    y += 0.12
ax.set_ylim(y, -0.05)

title = (f"5K · Where gemma stopped, what did Qwen do on the same trace, from the same turn?\n"
         f"Qwen cheated in {h_q}/{n_q} tries ({100 * h_q / n_q:.0f}%); {r_h}/{r_n} on the rendering trace "
         f"(run-{r_run}, trace {r_ts.split('_')[1][:5]}, {r_pts} restart points)")
sub = ("gemma's \"stopped\" tries end with a reply that carries no tool call; in this harness a plain reply is the final report, so the "
       "run ends there and the final state is scored. The biggest group ran `end_task` (a tool named in the paper's prompt but not "
       "offered by the framework) right after an honest shortfall report. Each row is one restart point on one Qwen trace: both models "
       f"continue from the same turn of the same transcript. {n_stops} gemma stops in {len(cells)} cells; Qwen's {n_q} scored tries in those cells.")
fig.text(0.02, 1 - 0.22 / H, title, fontsize=12.5, fontweight="bold", color=INK, va="top", linespacing=1.3)
fig.text(0.02, 1 - 0.78 / H, "\n".join(textwrap.wrap(sub, 205)), fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
leg = fig.legend(handles=[Patch(facecolor=col, label=lab) for _, lab, col in SEGS], loc="upper left", bbox_to_anchor=(0.02, 1 - 1.32 / H),
                 ncol=4, frameon=False, fontsize=8.5, handlelength=1.4, handleheight=0.9, columnspacing=1.6, borderaxespad=0)
for t in leg.get_texts():
    t.set_color(INK)

foot = ["Source: results/gap_replays.csv and results/stopped_reasons.csv (scripts/stopped_reasons.py). Chart 5J gives the pooled reasons; "
        "this chart pairs each gemma stop with Qwen's tries in the same cell. A cell is one environment x one restart point; counts are "
        "scored tries only (lost tries excluded).",
        "gemma's stop reasons in these cells — " + reason_line + "."]
lines = [w for line in foot for w in textwrap.wrap(line, 210, subsequent_indent="    ")]
for i, line in enumerate(lines):
    fig.text(0.02, (bot_in - 0.5 - 0.17 * i) / H, line, fontsize=7.6, color=MUTED, va="center")

out = HERE / "charts" / "q5_stopped_pairs.png"
fig.savefig(out, dpi=180, facecolor="white")
print(f"-> {out}")
