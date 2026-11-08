"""John's three end-of-run rules on all 29 tasks, one chart (8 Nov 2026, GG: "show them to me in a graph with all the
environments in which it was run"). Llama 3.3 70B (fp16, CoreWeave) under: no check (hero_solo*), the soft check = notice in
the system prompt + one "do you want to end?" question (hero2_llama*), the hard rule = only end_task ends the run, every
tool-less reply gets a reminder (hero4_llama*). Two panels, 29 rows each: left, the share of runs that ended on a plain reply
(by accident); right, the typical number of steps per run (median), with gemma-3 27B under the soft check as the reference
for a model that does the work. Reuses the run classifier of scripts/hero4_all29.py.
Run from gigi/3-forensics/agent-interp-envs:  env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/john_rules_all29.py
Writes charts/john_rules_all29.png and results/john_rules_all29.md.
"""
import collections, os, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import confirm_effect_charts as C
# hero4_all29.py renders its charts on import, so load only its definitions (everything before the first chart)
import types
_src = (H / "scripts" / "hero4_all29.py").read_text()
A = types.ModuleType("hero4_all29_defs"); A.__dict__["__file__"] = str(H / "scripts" / "hero4_all29.py")
exec(_src[:_src.index("fig, ax, rows = frame(")], A.__dict__)
A.med = lambda v: (sorted(v)[len(v) // 2] if v else "-")

RULES = [  # label, per-run csvs, soft?, colour
    ("no check (4 Nov)", A.PASSES[0][2], True, "#b5aea4"),
    ("soft check: notice in the prompt + one question (4-5 Nov)", A.PASSES[1][2], True, "#a99cf5"),
    ("hard rule: only end_task ends the run, reminders otherwise (7-8 Nov)", A.PASSES[2][2], False, "#6c4ff0"),
]
GEMMA = ("gemma-3 27B, soft check (reference)", ["hero2_full_runs__gemma-3-27b_per_run.csv", "hero2_full_runs__gemma-3-27b__other21_per_run.csv"], True, "#2f6fd6")

data = []   # (label, colour, endings, steps)
for label, files, soft, col in RULES + [GEMMA]:
    e, s = A.endings(files, soft)
    data.append((label, col, e, s))
fix_done = (A.R / "hero4_full_runs__llama-3.3-70b__fix_per_run.csv").exists()

rows, heads, ymax = C.layout()
envs = [e for e in rows]
fig = plt.figure(figsize=(14, 15.5))
fig.text(0.015, 0.985, "John's three end rules on all 29 tasks: Llama 3.3 70B ends its runs on purpose under each rule, and takes no more steps",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.015, 0.962,
         "Left: share of a task's runs that ended on a plain reply, the model simply stopped calling tools (under the hard rule that cannot end a run, so 0%). Right: typical steps per run (median).\n"
         "No check: a reply without a tool call ends the run. Soft check: the system prompt says so and the first such reply gets one question. Hard rule: only end_task\n"
         "ends the run, every tool-less reply gets a reminder (10 at most), then the step cap. gemma-3 27B ran with the soft check and shows what a model that does the work looks like.\n"
         "Llama 3.3 70B Instruct fp16 on CoreWeave; gemma fp8 on DeepInfra; all via OpenRouter. 95 + 286 runs per Llama rule. Counts on the right: median steps, no check / soft / hard.",
         fontsize=8.3, color="#555", va="top", linespacing=1.45)

def panel(ax, xlabel):
    ax.set_ylim(ymax, 0); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9, color="#444", labelpad=6)
    for title, y in heads:
        ax.axhline(y + 0.55, color="#ddd", lw=0.8, zorder=0)

axL = fig.add_axes([0.26, 0.085, 0.26, 0.815]); axR = fig.add_axes([0.575, 0.085, 0.30, 0.815])
panel(axL, "runs that ended on a plain reply (the model stopped calling tools)"); panel(axR, "typical steps per run (median)")
axL.set_xlim(-3, 108); axL.set_xticks([0, 25, 50, 75, 100]); axL.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
axR.set_xlim(0, 55); axR.set_xticks([0, 10, 20, 30, 40, 50]); axR.set_xticklabels(["0", "10", "20", "30", "40", "50+"])
off = [-0.22, 0.0, 0.22, 0.0]
for title, y in heads:
    fig.text(0.015, 0.085 + 0.815 * (1 - (y + 0.1) / ymax), title, fontsize=8.5, color="#777", fontweight="bold", va="center")
for e in envs:
    y = rows[e]
    fy = 0.085 + 0.815 * (1 - y / ymax)
    fig.text(0.015, fy, C.LABEL.get(e, e) + ("  ◆" if e in C.MAIN else ""), fontsize=8.8, fontweight="bold", va="center")
    meds = []
    for i, (label, col, ends, steps) in enumerate(data):
        n = sum(ends[e].values()) if e in ends else 0
        if n == 0:
            meds.append("-"); continue
        if i < 3:
            plain = 100 * ends[e]["plain"] / n
            axL.plot(plain, y + off[i], "o", color=col, ms=6.5, zorder=3)
        m = A.med(steps[e]); meds.append(str(m))
        x = min(m, 54)
        if i == 3:
            axR.plot(x, y, "s", color=col, ms=5.5, zorder=2, alpha=0.9)
        else:
            axR.plot(x, y + off[i], "o", color=col, ms=6.5, zorder=3)
    note = f"{meds[0]} / {meds[1]} / {meds[2]}"
    if not fix_done and e in A.FIX_ENVS and e in ("chess", "tictactoe", "swe_rebench_v2"):
        note += " †"
    fig.text(0.885, fy, note, fontsize=8.3, color="#333", va="center")
fig.text(0.885, 0.085 + 0.815 + 0.004, "steps: none / soft / hard", fontsize=8, color="#777", va="bottom")
handles = [plt.Line2D([], [], marker="o", color=c, ls="", ms=7, label=f"Llama, {l}") for l, _, _, c in RULES]
handles.append(plt.Line2D([], [], marker="s", color=GEMMA[3], ls="", ms=6, label=GEMMA[0]))
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.018), ncol=2, fontsize=8.3, frameon=False)
foot = "◆ = the 8 main tasks.  The step axis is cut at 50: tower of hanoi and the pre-fix chess and SWE runs sit beyond it."
if not fix_done:
    foot += "  † hard-rule steps from before the 8 Nov fix (the end tool did not end these runs; rerun in progress)."
fig.text(0.015, 0.004, foot, fontsize=7.6, color="#777", ha="left", va="bottom")
fig.savefig("charts/john_rules_all29.png", dpi=160, facecolor="white"); plt.close(fig)

# the numbers behind the chart
lines = ["# John's three end rules on all 29 tasks, Llama 3.3 70B (fp16, CoreWeave)", "",
         "Per rule: runs, how they ended (plain reply / end tool / task's own end / reminders used up / step cap), median steps. gemma-3 27B (fp8) with the soft check as reference.", "",
         "| task | rule | runs | plain | end tool | task end | reminders | cap | median steps |", "|---|---|---|---|---|---|---|---|---|"]
tot = collections.defaultdict(collections.Counter); tn = collections.Counter()
for e in envs:
    for label, col, ends, steps in data:
        if e not in ends: continue
        c = ends[e]; n = sum(c.values()); tn[label] += n
        for k in c: tot[label][k] += c[k]
        lines.append(f"| {C.LABEL.get(e, e)} | {label} | {n} | {c['plain']} | {c['end_tool']} | {c['task_end']} | {c['reminders']} | {c['cap']} | {A.med(steps[e])} |")
lines += ["", "## Totals", "", "| rule | runs | plain | end tool | task end | reminders | cap |", "|---|---|---|---|---|---|---|"]
for label, _, _, _ in data:
    c = tot[label]; lines.append(f"| {label} | {tn[label]} | {c['plain']} | {c['end_tool']} | {c['task_end']} | {c['reminders']} | {c['cap']} |")
if not fix_done:
    lines += ["", "Hard-rule rows for chess, tic-tac-toe and SWE-rebench are from before the 8 Nov fix (the end tool did not end those runs, so steps and caps there measure the bug); the rerun (stage hero4_llama_fix) replaces them when it lands."]
pathlib.Path("results/john_rules_all29.md").write_text("\n".join(lines) + "\n")
print("wrote charts/john_rules_all29.png and results/john_rules_all29.md")
for label, _, _, _ in data:
    print(label, dict(tot[label]), "runs", tn[label])
