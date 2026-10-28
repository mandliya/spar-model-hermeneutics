"""Charts 5I and 5J of RESULTS.md (29 Oct 2026), the two "what is missing" pictures under 5G.
5I: the final loss table by cause: planned swap tries not scored, per cause and model, from results/gap_replays.csv and
    the plan (swap_all_score.PLAN; target 10 per cell, 60 for dashboard_costly_failure's Qwen cheating cells). Fixable
    causes (provider failure, crash, cut off, budget stop) count only where a cell is still short after the reruns.
5J: why the scored "stopped" tries stopped (results/stopped_reasons.csv, scripts/stopped_reasons.py), per model.
Also prints the markdown table for 5I.
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/loss_charts.py
"""
import collections, csv, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import swap_all_score as S

INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
MODEL_COL = {"qwen": "#1baf7a", "gemma": "#2a78d6"}
UNFIX = {"context window exceeded", "no measure", "no value for the measure", "step cap, no report"}
CAUSE_LABEL = {"context window exceeded": "context window exceeded\n(gemma 128k, Qwen 262k)",
               "no measure": "no measure at this point\n(sandbagging)",
               "no value for the measure": "no value for the measure\n(chess, puppeteer)",
               "step cap, no report": "step cap, no report",
               "fixable": "fixable (provider failure, crash,\ncut off, budget stop): none left"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})

# ---- 5I ---------------------------------------------------------------------------------------------------------------
scored, lost = collections.Counter(), collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
    k = (r["env"], r["point"], r["who"])
    if r["hacked"] != "":
        scored[k] += 1
    else:
        lost[k][r["status"]] += 1
short = collections.defaultdict(int); cells = collections.defaultdict(list)
n_cells = planned = done = 0
for p in S.PLAN:
    name = S.point_name(p)
    for who in ("qwen", "gemma"):
        k = (p["env"], name, who)
        target = 60 if (p["env"] == "dashboard_costly_failure" and p["incident"] and who == "qwen") else p.get("reps", 10)
        n_cells += 1; planned += target; done += min(scored[k], target)
        gap = target - scored[k]
        if gap <= 0:
            continue
        ux = {s: v for s, v in lost[k].items() if s in UNFIX}
        cause = max(ux, key=ux.get) if ux else "fixable"
        short[(cause, who)] += gap; cells[(cause, who)].append(f"{p['env']} K{p['K']} {scored[k]}/{target}")
causes = [c for c in ("context window exceeded", "no measure", "no value for the measure", "step cap, no report", "fixable")]
fig = plt.figure(figsize=(11, 4.6)); ax = fig.add_axes([0.3, 0.16, 0.6, 0.6])
xmax = max(short.values()) * 1.3
for i, c in enumerate(causes):
    ax.text(-xmax * 0.02, i, CAUSE_LABEL[c], ha="right", va="center", fontsize=8.8, linespacing=1.2, fontweight="bold" if c != "fixable" else "normal",
            color=INK if c != "fixable" else MUTED)
    for j, who in enumerate(("qwen", "gemma")):
        v = short[(c, who)]; y = i + (j - 0.5) * 0.36
        ax.barh(y, v, height=0.32, color=MODEL_COL[who], zorder=3)
        ax.text(v + xmax * 0.01, y, f"{v} tries in {len(cells[(c, who)])} cells" if v else "0", fontsize=8, va="center", color=INK if v else MUTED)
ax.set_xlim(0, xmax); ax.set_ylim(len(causes) - 0.5, -0.5)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", left=False, labelleft=False); ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0); ax.set_xlabel("planned swap tries that produced no scorable result", fontsize=8.5, color=MUTED)
fig.text(0.02, 0.97, f"5I · What is missing from the swap data, and why: {planned - done} of {planned} planned tries ({(planned - done) / planned * 100:.0f}%), all for reasons a rerun cannot fix",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.02, 0.885, f"{n_cells} cells (one restart point × one model), 10 tries each (60 for the dashboard's Qwen cheating cells). Every try lost to a provider failure, a crash, a cut-off "
         "download or a budget stop was rerun on 27–28 Oct;\nwhat remains is the model's own window, environments with nothing to measure at that point, or a 500-step run with no report. "
         "A scored 'stopped' try is not a loss (chart 5J).", fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COL[w]) for w in ("qwen", "gemma")], labels=["Qwen3-Next-80B", "gemma-3-27b"],
           loc="upper left", bbox_to_anchor=(0.015, 0.8), ncol=2, frameon=False, fontsize=8.5, handlelength=1.3)
fig.text(0.02, 0.03, "Source: results/gap_replays.csv (scripts/gap_replays.py) and results/loss_audit.md, which lists every cell.", fontsize=7.6, color=MUTED)
fig.savefig(HERE / "charts" / "q5_losses.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)

print(f"{n_cells} cells, {planned} planned, {done} scored ({done / planned * 100:.1f}%)")
print("| Cause | Fixable? | gemma: tries · cells | Qwen: tries · cells | Cells (scored/target) |\n|---|---|---|---|---|")
for c in causes:
    g, q = short[(c, "gemma")], short[(c, "qwen")]
    where = "; ".join(cells[(c, "gemma")] + [x + " (Qwen)" for x in cells[(c, "qwen")]])[:400]
    print(f"| {CAUSE_LABEL[c].replace(chr(10), ' ')} | {'yes, all rerun' if c == 'fixable' else 'no'} | {g} · {len(cells[(c, 'gemma')])} | {q} · {len(cells[(c, 'qwen')])} | {where or '—'} |")

# ---- 5J ---------------------------------------------------------------------------------------------------------------
REASON_ORDER = ["repeated itself until the token limit", "tried to end the task, then said the session was over", "announced its next action, no call",
                "pasted code or a prompt as text", "plan or summary, no action", "waited for the user or for output", "ran out of steps",
                "malformed or unavailable tool call", "empty reply"]
REASON_LABEL = {"repeated itself until the token limit": "repeated one line until the token limit\n(generation collapse; all dashboard)",
                "tried to end the task, then said the session was over": "ran `end_task` (in the prompt, not a tool), then `exit`,\nthen said the session was over (all dashboard)",
                "announced its next action, no call": "announced its next step, made no tool call",
                "pasted code or a prompt as text": "pasted code or a prompt as text\ninstead of writing it",
                "plan or summary, no action": "laid out a plan or a summary, no action, no claim",
                "waited for the user or for output": "waited for the user or for output",
                "ran out of steps": "ran out of steps", "malformed or unavailable tool call": "malformed tool call", "empty reply": "empty reply"}
cnt = collections.Counter(); tot = collections.Counter()
for r in csv.DictReader(open(HERE / "results" / "stopped_reasons.csv")):
    cnt[(r["reason"], r["who"])] += 1; tot[r["who"]] += 1
reasons = [x for x in REASON_ORDER if cnt[(x, "qwen")] + cnt[(x, "gemma")]]
fig = plt.figure(figsize=(11, 2.2 + 0.62 * len(reasons))); ax = fig.add_axes([0.36, 0.9 / (2.2 + 0.62 * len(reasons)), 0.55, 0.62 * len(reasons) / (2.2 + 0.62 * len(reasons))])
xmax = max(cnt.values()) * 1.3
for i, x in enumerate(reasons):
    ax.text(-xmax * 0.02, i, REASON_LABEL[x], ha="right", va="center", fontsize=8.6, linespacing=1.2)
    for j, who in enumerate(("qwen", "gemma")):
        v = cnt[(x, who)]; y = i + (j - 0.5) * 0.36
        ax.barh(y, v, height=0.32, color=MODEL_COL[who], zorder=3)
        ax.text(v + xmax * 0.01, y, str(v) if v else "0", fontsize=8, va="center", color=INK if v else MUTED)
ax.set_xlim(0, xmax); ax.set_ylim(len(reasons) - 0.5, -0.5)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", left=False, labelleft=False); ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0); ax.set_xlabel("scored tries in the gap environments whose last reply had no tool call and no report", fontsize=8.5, color=MUTED)
Hj = 2.2 + 0.62 * len(reasons)
fig.text(0.02, 1 - 0.12 / Hj, f"5J · Why the 'stopped' tries stopped: Qwen {tot['qwen']}, gemma {tot['gemma']}. These are scored tries (verdict: not hacked), not losses",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.02, 1 - 0.45 / Hj, "In this harness a reply with no tool call is the final report, so the run ends and the final state is scored. Qwen's stops are almost all one failure: "
         "on the dashboard's 60–100-turn histories it repeats\none line until the token limit. gemma's biggest group is honest: it reports the shortfall, then tries `end_task` "
         "(which the paper's prompt names but the framework does not provide), then `exit`, then says the session ended.",
         fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COL[w]) for w in ("qwen", "gemma")], labels=["Qwen3-Next-80B", "gemma-3-27b"],
           loc="upper left", bbox_to_anchor=(0.015, 1 - 0.78 / Hj), ncol=2, frameon=False, fontsize=8.5, handlelength=1.3)
fig.text(0.02, 0.2 / Hj, "Source: results/stopped_reasons.csv (scripts/stopped_reasons.py), classified from each try's last assistant message; the gap environments only.", fontsize=7.6, color=MUTED)
fig.savefig(HERE / "charts" / "q5_stopped.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"5J: Qwen {tot['qwen']}, gemma {tot['gemma']} stopped tries")
