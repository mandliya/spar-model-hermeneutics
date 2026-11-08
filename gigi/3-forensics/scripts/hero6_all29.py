"""Llama 3.3 70B on all 29 tasks under the hard end rule PLUS the worked example (8 Nov 2026 evening; stages hero6_example,
hero6_example_key, hero6_example_all) and PLUS the control example (same night; stages hero7_control, hero7_control_key,
hero7_control_all: the same example, but the demonstrated run ends "2 passed, 1 failed" and the report says so), next to the
hard rule alone (hero4_llama*, the __fix rerun first), gemma-3 27B and Qwen3.6 27B with the soft check (hero2_gemma*,
hero2_qwen36*). Three charts and one table:
  charts/hero6_all29_cheating.png  per task, the bad-act rate (k of n, Wilson interval) under the five conditions;
  charts/hero6_all29_work.png      per task, left: share of runs that opened a source file (hard rule alone / + example / + control);
                                   right: median steps per run (same three, gemma as a blue tick);
  charts/hero6_all29_endings.png   per task, how the runs ended (end tool / task's own end / reminders / cap), three 100% bars;
  results/hero6_all29.md           all of it as one table, with totals, and a paragraph template at the top.
Runs with the hero6 or hero7 data missing: prints which stages are missing, draws what exists, leaves the rest out.
Run from gigi/3-forensics/agent-interp-envs:  env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero6_all29.py
"""
import collections, csv, json, math, os, pathlib, re, sys, types
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import confirm_effect_charts as C
# hero4_all29.py renders on import: load only its definitions (rates, max_steps, REPROMPT, PASSES)
_src = (H / "scripts" / "hero4_all29.py").read_text()
A = types.ModuleType("hero4_all29_defs"); A.__dict__["__file__"] = str(H / "scripts" / "hero4_all29.py")
exec(_src[:_src.index("fig, ax, rows = frame(")], A.__dict__)
R = H / "results"

READ_RE = re.compile(r"\b(cat|less|head|tail|sed -n|nl|bat|view|open)\b[^|;&\n]*\.(py|js|mjs|cjs|ts|tsx|jsx|c|cc|cpp|h|rs|go|java|json|yaml|yml|toml|md|txt|sh)\b|"
                     r"\b(grep|rg|ag)\b|\bread_file\b|\bview_file\b", re.I)
FINISH = ("end_task", "done", "end_session")

# label, summary csvs (earlier files win per task), per-run csvs, colour, soft check?
HARD = ("Llama, hard rule alone (7-8 Nov)", A.PASSES[2][1], A.PASSES[2][2], "#6c4ff0", False)
EXAMPLE = ("Llama, hard rule + worked example (8 Nov)", ["hero6_full_runs__llama-3.3-70b.csv", "hero6_full_runs__llama-3.3-70b__other21.csv"],
           ["hero6_full_runs__llama-3.3-70b_per_run.csv", "hero6_full_runs__llama-3.3-70b__other21_per_run.csv"], "#d6336c", False)
CONTROL = ("Llama, hard rule + control example (8 Nov)", ["hero7_full_runs__llama-3.3-70b.csv", "hero7_full_runs__llama-3.3-70b__other21.csv"],
           ["hero7_full_runs__llama-3.3-70b_per_run.csv", "hero7_full_runs__llama-3.3-70b__other21_per_run.csv"], "#e8871e", False)
GEMMA = ("gemma-3 27B, soft check (5-6 Nov)", ["hero2_full_runs__gemma-3-27b.csv", "hero2_full_runs__gemma-3-27b__other21.csv"],
         ["hero2_full_runs__gemma-3-27b_per_run.csv", "hero2_full_runs__gemma-3-27b__other21_per_run.csv"], "#2f6fd6", True)
QWEN36 = ("Qwen3.6 27B, soft check, thinking off (7-8 Nov)", ["hero2_full_runs__qwen3.6-27b.csv", "hero2_full_runs__qwen3.6-27b__other21.csv"],
          ["hero2_full_runs__qwen3.6-27b_per_run.csv", "hero2_full_runs__qwen3.6-27b__other21_per_run.csv"], "#1f9d7a", True)
CONDS = [HARD, EXAMPLE, CONTROL, GEMMA, QWEN36]
LLAMA3 = (HARD, EXAMPLE, CONTROL)
NOTE = "Llama 3.3 70B Instruct fp16 on CoreWeave; gemma-3 27B and Qwen3.6 27B fp8 on DeepInfra; all via OpenRouter."
HERO6_STAGES = ("hero6_example", "hero6_example_key", "hero6_example_all")
HERO7_STAGES = ("hero7_control", "hero7_control_key", "hero7_control_all")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def runs(files, soft):
    """env -> dict(ends=Counter, steps=[...], opened=int, n=int). An environment's runs come from the first per-run csv that
    has it (so the hero4 __fix file replaces the pre-fix runs of its five tasks)."""
    out = collections.defaultdict(lambda: {"ends": collections.Counter(), "steps": [], "opened": 0, "n": 0})
    seen_in = {}
    for f in files:
        p = R / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            rd, env = H / r["run"], r["environment"]
            if seen_in.setdefault(env, f) != f:
                continue
            ms = sorted(rd.glob("step-*/messages.json"), key=lambda q: int(re.search(r"step-(\d+)", str(q)).group(1)))
            if not ms:
                continue
            try:
                m = json.load(open(ms[-1]))
            except Exception:
                continue
            n = len(ms)
            asst = [x for x in m if x.get("role") == "assistant"]
            last = asst[-1] if asst else {}
            lastcalls = [(t.get("function") or {}).get("name") for t in last.get("tool_calls") or []]
            rem = sum(1 for x in m if x.get("role") == "user" and str(x.get("content", "")).startswith(A.REPROMPT))
            if n >= A.max_steps(env): k = "cap"
            elif any(c in FINISH for c in lastcalls): k = "end_tool"
            elif not soft and rem >= 10: k = "reminders"
            elif not lastcalls: k = "plain"
            else: k = "task_end"
            args = " \n ".join(((t.get("function") or {}).get("arguments") or "") for x in asst for t in x.get("tool_calls") or [])
            args += " \n " + " \n ".join(str(x.get("content") or "") for x in asst if not x.get("tool_calls"))
            d = out[env]; d["ends"][k] += 1; d["steps"].append(n); d["opened"] += bool(READ_RE.search(args)); d["n"] += 1
    return out


def med(v):
    v = sorted(v); return v[len(v) // 2] if v else None


def stages_present(files):
    return {r["stage"] for f in files if (R / f).exists() for r in csv.DictReader(open(R / f))}
missing6 = [s for s in HERO6_STAGES if s not in stages_present(EXAMPLE[2])]
missing7 = [s for s in HERO7_STAGES if s not in stages_present(CONTROL[2])]
missing = missing6 + missing7
for name, m in (("hero6", missing6), ("hero7", missing7)):
    if m:
        print(name, "stages not on disk yet:", ", ".join(m), "(their marks are left out)")
MISSING_OF = {EXAMPLE[0]: bool(missing6), CONTROL[0]: bool(missing7)}
MISSNOTE = ("  Not on disk yet: " + ", ".join(missing) + ".") if missing else ""

rates = {c[0]: A.rates(c[1]) for c in CONDS}
per = {c[0]: runs(c[2], c[4]) for c in CONDS}
rows, heads, ymax = C.layout()
envs = list(rows)


def frame(w, h, title, sub, axes):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.012, 0.985, title, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.012, 0.966, sub, fontsize=8.3, color="#555", va="top", linespacing=1.45)
    TOP, HGT = 0.065, 0.86
    fy = lambda y: TOP + HGT * (1 - y / ymax)
    out = []
    for x0, wd, xlabel in axes:
        ax = fig.add_axes([x0, TOP, wd, HGT]); ax.set_ylim(ymax, 0); ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#eee", lw=0.8); ax.set_axisbelow(True); ax.set_xlabel(xlabel, fontsize=9, color="#444", labelpad=6)
        for t, y in heads:
            ax.axhline(y + 0.55, color="#ddd", lw=0.8, zorder=0)
        out.append(ax)
    for t, y in heads:
        fig.text(0.012, fy(y + 0.1), t, fontsize=8.5, color="#777", fontweight="bold", va="center")
    for e in envs:
        fig.text(0.012, fy(rows[e]), C.LABEL.get(e, e) + ("  ◆" if e in C.MAIN else ""), fontsize=8.6, fontweight="bold", va="center")
    return fig, out, fy


# 1. cheating: four labelled bars per task
fig, (ax,), fy = frame(15, 24, "The bad act on all 29 tasks: Llama alone, with the worked example, with the control example; gemma-3 27B and Qwen3.6 27B next to it",
    "Five bars per task, top to bottom: Llama hard rule alone; Llama hard rule + worked example (a six-line sample transcript: read, change, check, finish, ending \"all tests pass\");\n"
    "Llama hard rule + control example (the same transcript, but it ends \"2 passed, 1 failed\" and the report says so); gemma-3 27B with the soft check; Qwen3.6 27B with the soft check.\n"
    "Bar = share of scored runs with the task's own bad act, thin line = 95% Wilson interval, k of n printed. "
    "\"no value\" = no scored run in that cell. " + NOTE, [(0.30, 0.62, "share of scored runs with the bad act")])
ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
off, hgt = [-0.32, -0.16, 0.0, 0.16, 0.32], 0.14
for e in envs:
    y = rows[e]
    for i, (label, _, _, col, _) in enumerate(CONDS):
        kn = rates[label].get(e)
        if kn is None:
            if MISSING_OF.get(label):
                continue
            ax.text(0.8, y + off[i], "no value", fontsize=6.6, color="#aaa", va="center"); continue
        k, n = kn; p = 100 * k / n; lo, hi = wilson(k, n)
        ax.barh(y + off[i], p, height=hgt, color=col, zorder=3)
        ax.plot([100 * lo, 100 * hi], [y + off[i]] * 2, color=col, lw=0.7, alpha=0.45, zorder=2)
        ax.text(p + 1.2, y + off[i], f"{k} of {n}", fontsize=6.2, color="#333", va="center", fontweight="bold" if i in (1, 2) else None,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8), zorder=5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c[3], label=c[0]) for c in CONDS]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=3, fontsize=8.2, frameon=False)
fig.text(0.985, 0.004, "◆ = the 8 main tasks." + MISSNOTE, fontsize=7.6, color="#777", ha="right", va="bottom")
fig.savefig("charts/hero6_all29_cheating.png", dpi=160, facecolor="white"); plt.close(fig)
print("wrote charts/hero6_all29_cheating.png")

# 2. work: opened a source file (left), median steps (right)
fig, (axL, axR), fy = frame(15, 19, "Did Llama open the code, and how many steps did it take? All 29 tasks: the hard rule alone, with the worked example, with the control example",
    "Three bars per task, top to bottom: hard rule alone, + worked example, + control example. Left: share of runs in which the model read at least one source file (cat, head, grep, sed -n, a read tool...),\n"
    "k of n printed; in tasks without source files this is simply what the pattern finds. Right: typical steps per run (median), gemma-3 27B with the soft check as the blue tick (its median next to it).\n"
    + NOTE, [(0.30, 0.28, "runs that opened a source file"), (0.63, 0.30, "typical steps per run (median)")])
axL.set_xlim(0, 128); axL.set_xticks([0, 25, 50, 75, 100]); axL.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
axR.set_xlim(0, 62); axR.set_xticks([0, 10, 20, 30, 40, 50]); axR.set_xticklabels(["0", "10", "20", "30", "40", "50+"])
off2, hgt2 = [-0.27, 0.0, 0.27], 0.22
for e in envs:
    y = rows[e]
    for i, cond in enumerate(LLAMA3):
        d = per[cond[0]].get(e)
        if not d or d["n"] == 0:
            if not MISSING_OF.get(cond[0]):
                axL.text(0.8, y + off2[i], "no runs", fontsize=6.6, color="#aaa", va="center")
            continue
        p = 100 * d["opened"] / d["n"]
        axL.barh(y + off2[i], p, height=hgt2, color=cond[3], zorder=3)
        axL.text(p + 1.2, y + off2[i], f"{d['opened']} of {d['n']}", fontsize=6.6, color="#333", va="center", fontweight="bold" if i else None)
        m = med(d["steps"]); x = min(m, 55)
        axR.barh(y + off2[i], x, height=hgt2, color=cond[3], zorder=3)
        axR.text(x + 0.8, y + off2[i], f"{m}", fontsize=6.6, color="#333", va="center", fontweight="bold" if i else None)
    g = per[GEMMA[0]].get(e)
    if g and g["n"]:
        gm = med(g["steps"]); gx = min(gm, 55)
        axR.plot([gx, gx], [y - 0.42, y + 0.42], color=GEMMA[3], lw=2.2, zorder=4, solid_capstyle="butt")
        axR.text(56.5, y, f"gemma {gm}", fontsize=6.8, color=GEMMA[3], va="center")
handles = [plt.Rectangle((0, 0), 1, 1, color=c[3], label=c[0]) for c in LLAMA3] + [plt.Line2D([], [], color=GEMMA[3], lw=2.2, label=GEMMA[0] + ", median steps")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=8.2, frameon=False)
fig.text(0.985, 0.004, "◆ = the 8 main tasks.  Step axis cut at 50." + MISSNOTE, fontsize=7.6, color="#777", ha="right", va="bottom")
fig.savefig("charts/hero6_all29_work.png", dpi=160, facecolor="white"); plt.close(fig)
print("wrote charts/hero6_all29_work.png")

# 3. endings: two 100% bars per task
ENDS = [("end_tool", "called end_task (or the task's own done / end_session)", "#6c4ff0"), ("task_end", "the task ended itself (answer, email, game over, submit)", "#8fb3e6"),
        ("reminders", "10 reminders used up", "#f0a35e"), ("cap", "step cap", "#d9534f"), ("plain", "plain reply (pre-fix only)", "#d9d4cc")]
DARK = {"#6c4ff0", "#d9534f"}
fig, (ax,), fy = frame(15, 19, "How Llama's runs ended on all 29 tasks: the hard rule alone (upper bar), with the worked example (middle), with the control example (lower)",
    "Hard rule: a plain reply never ends the run; it ends when Llama calls the finish tool, when the task ends itself after a tool call, after 10 tool-less replies in a row, or at the step cap.\n"
    "Each bar is 100% of that condition's runs for the task; the count of runs is printed in each segment wide enough. Run counts on the right: alone / + worked example / + control.\n" + NOTE,
    [(0.30, 0.55, "share of the runs")])
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for e in envs:
    y = rows[e]; ns = []
    for i, cond in enumerate(LLAMA3):
        d = per[cond[0]].get(e)
        if not d or d["n"] == 0:
            ns.append("-")
            if not MISSING_OF.get(cond[0]):
                ax.text(0.005, y + off2[i], "no runs", fontsize=6.6, color="#aaa", va="center")
            continue
        n = d["n"]; ns.append(str(n)); x = 0
        for key, lab, col in ENDS:
            k = d["ends"].get(key, 0)
            if not k: continue
            w = k / n
            ax.barh(y + off2[i], w, left=x, height=hgt2, color=col, zorder=3, edgecolor="white", lw=0.6, alpha=(0.7, 1.0, 0.85)[i])
            if w >= 0.09:
                ax.text(x + w / 2, y + off2[i], str(k), fontsize=6.2, color="white" if col in DARK else "#222", ha="center", va="center", fontweight="bold")
            x += w
    fig.text(0.87, fy(y), " / ".join(ns), fontsize=7.4, color="#333", va="center")
fig.text(0.87, 0.065 + 0.86 + 0.004, "runs: alone / + example / + control", fontsize=7.4, color="#777", va="bottom")
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for k, l, c in ENDS[:4]]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=8.2, frameon=False)
fig.text(0.985, 0.004, "◆ = the 8 main tasks.  Upper bar, lighter: hard rule alone; middle: + worked example; lower: + control example." + MISSNOTE, fontsize=7.6, color="#777", ha="right", va="bottom")
fig.savefig("charts/hero6_all29_endings.png", dpi=160, facecolor="white"); plt.close(fig)
print("wrote charts/hero6_all29_endings.png")

# 4. the table
def kn(label, e):
    v = rates[label].get(e); return f"{v[0]} of {v[1]}" if v else "no value"
def work(label, e):
    d = per[label].get(e)
    if not d or not d["n"]: return "-", "-", "-"
    c = d["ends"]; return f"{d['opened']} of {d['n']}", str(med(d["steps"])), f"{c['end_tool']}/{c['task_end']}/{c['reminders']}/{c['cap']}"
lines = ["# Llama 3.3 70B on all 29 tasks: the hard rule alone, with the worked example and with the control example (8 Nov 2026)", "",
         "<!-- paragraph to fill in once the runs are read -->",
         "**What the worked example changed, in one paragraph:** _[fill in: in which tasks does Llama now open the code and take more steps; where does the bad-act rate move, and does it survive the control example; how does it compare with gemma-3 27B and Qwen3.6 27B; how did the runs end; scored vs lost; cost.]_", "",
         NOTE + " Bad act = each task's own measure over the scored runs. \"Opened\" = runs with at least one read of a source file. Endings = called the finish tool / the task ended itself / 10 reminders used up / step cap. "
         "Worked example: a six-line sample transcript in the system prompt (read a file, change it, run the tests, \"3 passed\", finish with \"all tests pass\"). Control example: the same transcript, but the tests show \"2 passed, 1 failed\" and the report says exactly that.", "",
         "| task | bad act, hard rule alone | bad act, + worked example | bad act, + control example | bad act, gemma-3 27B soft | bad act, Qwen3.6 27B soft | opened, alone | opened, + example | opened, + control | median steps, alone | median steps, + example | median steps, + control | median steps, gemma | endings alone | endings + example | endings + control |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
tot = {c[0]: collections.Counter() for c in CONDS}
for e in envs:
    oa, sa, ea = work(HARD[0], e); oe, se, ee = work(EXAMPLE[0], e); oc, sc, ec = work(CONTROL[0], e); gd = per[GEMMA[0]].get(e); gs = str(med(gd["steps"])) if gd and gd["n"] else "-"
    for c in CONDS:
        v = rates[c[0]].get(e)
        if v: tot[c[0]]["k"] += v[0]; tot[c[0]]["n"] += v[1]
        d = per[c[0]].get(e)
        if d: tot[c[0]]["opened"] += d["opened"]; tot[c[0]]["runs"] += d["n"]; tot[c[0]].update({f"end_{k}": v for k, v in d["ends"].items()})
    lines.append(f"| {C.LABEL.get(e, e)}{' ◆' if e in C.MAIN else ''} | {kn(HARD[0], e)} | {kn(EXAMPLE[0], e)} | {kn(CONTROL[0], e)} | {kn(GEMMA[0], e)} | {kn(QWEN36[0], e)} | {oa} | {oe} | {oc} | {sa} | {se} | {sc} | {gs} | {ea} | {ee} | {ec} |")
lines += ["", "## Totals", "", "| condition | bad acts / scored runs (all tasks pooled) | opened a source file / runs read | ended: end tool / task's end / reminders / cap / plain |", "|---|---|---|---|"]
for c in CONDS:
    t = tot[c[0]]
    lines.append(f"| {c[0]} | {t['k']} of {t['n']} | {t['opened']} of {t['runs']} | {t['end_end_tool']} / {t['end_task_end']} / {t['end_reminders']} / {t['end_cap']} / {t['end_plain']} |")
if missing:
    lines += ["", "Stages not on disk when this was written: " + ", ".join(missing) + "."]
pathlib.Path("results/hero6_all29.md").write_text("\n".join(lines) + "\n")
print("wrote results/hero6_all29.md")
for c in CONDS:
    print(c[0], dict(tot[c[0]]))
