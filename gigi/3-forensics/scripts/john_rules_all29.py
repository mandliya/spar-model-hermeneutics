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
fig = plt.figure(figsize=(14, 17))
fig.text(0.015, 0.985, "John's three end rules on all 29 tasks: Llama 3.3 70B ends its runs on purpose under each rule, and takes no more steps",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.015, 0.965,
         "Three bars per task, one per rule, top to bottom: no check, soft check, hard rule. Left: how many runs ended on a plain reply (the model stopped calling tools);\n"
         "under the hard rule that cannot end a run. Right: typical steps per run (median), with gemma-3 27B under the soft check as the blue mark: what a model that does the work looks like.\n"
         "No check: a reply without a tool call ends the run. Soft check: the system prompt says so and the first such reply gets one question. Hard rule: only end_task ends the run,\n"
         "every tool-less reply gets a reminder (10 at most), then the step cap. Llama 3.3 70B Instruct fp16 on CoreWeave; gemma fp8 on DeepInfra; via OpenRouter. 371 Llama runs per rule.",
         fontsize=8.3, color="#555", va="top", linespacing=1.45)

TOP, HGT = 0.085, 0.82
def panel(ax, xlabel):
    ax.set_ylim(ymax, 0); ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#eee", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9, color="#444", labelpad=6)
    for title, y in heads:
        ax.axhline(y + 0.55, color="#ddd", lw=0.8, zorder=0)

axL = fig.add_axes([0.27, TOP, 0.25, HGT]); axR = fig.add_axes([0.60, TOP, 0.30, HGT])
fig.text(0.915, TOP + HGT + 0.004, "gemma-3 27B\nmedian steps", fontsize=7.6, color=GEMMA[3], va="bottom")
panel(axL, "runs that ended on a plain reply"); panel(axR, "typical steps per run (median)")
axL.set_xlim(0, 125); axL.set_xticks([0, 25, 50, 75, 100]); axL.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
axR.set_xlim(0, 62); axR.set_xticks([0, 10, 20, 30, 40, 50]); axR.set_xticklabels(["0", "10", "20", "30", "40", "50+"])
off, hgt = [-0.27, 0.0, 0.27], 0.24
for title, y in heads:
    fig.text(0.015, TOP + HGT * (1 - (y + 0.1) / ymax), title, fontsize=8.5, color="#777", fontweight="bold", va="center")
for e in envs:
    y = rows[e]
    fy = TOP + HGT * (1 - y / ymax)
    fig.text(0.015, fy, C.LABEL.get(e, e) + ("  ◆" if e in C.MAIN else ""), fontsize=8.8, fontweight="bold", va="center")
    for i, (label, col, ends, steps) in enumerate(data[:3]):
        n = sum(ends[e].values()) if e in ends else 0
        if n == 0:
            axL.text(1, y + off[i], "no runs", fontsize=6.8, color="#999", va="center"); continue
        k = ends[e]["plain"]; plain = 100 * k / n
        axL.barh(y + off[i], plain, height=hgt, color=col, zorder=3)
        axL.text(plain + 1.5, y + off[i], f"{k} of {n}", fontsize=6.9, color="#333", va="center")
        m = A.med(steps[e]); x = min(m, 55)
        axR.barh(y + off[i], x, height=hgt, color=col, zorder=3)
        pre = e in ("chess", "tictactoe", "swe_rebench_v2") and i == 2 and not fix_done
        axR.text(x + 0.8, y + off[i], f"{m}" + (" †" if pre else ""), fontsize=6.9, color="#333", va="center", fontweight="bold" if i == 2 else None)
    glabel, gcol, gends, gsteps = data[3]
    if e in gsteps:
        gm = A.med(gsteps[e]); gx = min(gm, 55)
        axR.plot([gx, gx], [y - 0.42, y + 0.42], color=gcol, lw=2.2, zorder=4, solid_capstyle="butt")
        fig.text(0.915, fy, f"gemma {gm}", fontsize=7.6, color=gcol, va="center")
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"Llama, {l}") for l, _, _, c in RULES]
handles.append(plt.Line2D([], [], color=GEMMA[3], lw=2.2, label=GEMMA[0] + ", median steps"))
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
