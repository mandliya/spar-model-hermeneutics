"""Like for like at last: gemma-3 27B and Qwen3.6 27B on all 29 tasks under Llama's exact setting, the hard end rule PLUS the
worked example (9 Nov 2026 night; stages hero8_gemma, hero8_gemma_key, hero8_gemma_all, hero8_qwen36, hero8_qwen36_key,
hero8_qwen36_all), next to Llama 3.3 70B in that setting (hero6_example*). The stand-ins' soft-check runs (hero2_gemma*,
hero2_qwen36*) are drawn as faint ticks so the chart also says whether the setting change moved them. Three charts and a table:
  charts/hero8_all29_cheating.png  per task, the bad-act rate (k of n, Wilson interval) for the three models, same setting;
  charts/hero8_all29_work.png      per task, left: share of runs that opened a source file; right: median steps per run;
  charts/hero8_all29_endings.png   per task, how the runs ended (end tool / task's own end / reminders / cap), three 100% bars;
  results/hero8_all29.md           all of it as one table, with totals and the list of clear gaps (intervals that do not overlap).
Run from gigi/3-forensics/agent-interp-envs:  env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero8_all29.py
"""
import collections, csv, json, math, os, pathlib, re, sys, types
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import chart_style as ST
import confirm_effect_charts as C
_src = (H / "scripts" / "hero4_all29.py").read_text()   # renders on import: load only its definitions
A = types.ModuleType("hero4_all29_defs"); A.__dict__["__file__"] = str(H / "scripts" / "hero4_all29.py")
exec(_src[:_src.index("fig, ax, rows = frame(")], A.__dict__)
R = H / "results"

READ_RE = re.compile(r"\b(cat|less|head|tail|sed -n|nl|bat|view|open)\b[^|;&\n]*\.(py|js|mjs|cjs|ts|tsx|jsx|c|cc|cpp|h|rs|go|java|json|yaml|yml|toml|md|txt|sh)\b|"
                     r"\b(grep|rg|ag)\b|\bread_file\b|\bview_file\b", re.I)
FINISH = ("end_task", "done", "end_session")

# label, summary csvs, per-run csvs, colour, soft check?
LLAMA = ("Llama 3.3 70B", ["hero6_full_runs__llama-3.3-70b.csv", "hero6_full_runs__llama-3.3-70b__other21.csv"],
         ["hero6_full_runs__llama-3.3-70b_per_run.csv", "hero6_full_runs__llama-3.3-70b__other21_per_run.csv"], ST.LLAMA_EXAMPLE, False)
GEMMA = ("gemma-3 27B", ["hero8_full_runs__gemma-3-27b.csv", "hero8_full_runs__gemma-3-27b__other21.csv"],
         ["hero8_full_runs__gemma-3-27b_per_run.csv", "hero8_full_runs__gemma-3-27b__other21_per_run.csv"], ST.GEMMA, False)
QWEN = ("Qwen3.6 27B, thinking off", ["hero8_full_runs__qwen3.6-27b.csv", "hero8_full_runs__qwen3.6-27b__other21.csv"],
        ["hero8_full_runs__qwen3.6-27b_per_run.csv", "hero8_full_runs__qwen3.6-27b__other21_per_run.csv"], ST.QWEN, False)
GEMMA_SOFT = ("gemma-3 27B, soft check (5-6 Nov)", ["hero2_full_runs__gemma-3-27b.csv", "hero2_full_runs__gemma-3-27b__other21.csv"],
              ["hero2_full_runs__gemma-3-27b_per_run.csv", "hero2_full_runs__gemma-3-27b__other21_per_run.csv"], ST.GEMMA, True)
QWEN_SOFT = ("Qwen3.6 27B, soft check (7-8 Nov)", ["hero2_full_runs__qwen3.6-27b.csv", "hero2_full_runs__qwen3.6-27b__other21.csv"],
             ["hero2_full_runs__qwen3.6-27b_per_run.csv", "hero2_full_runs__qwen3.6-27b__other21_per_run.csv"], ST.QWEN, True)
THREE = [LLAMA, GEMMA, QWEN]
SOFT = {GEMMA[0]: GEMMA_SOFT, QWEN[0]: QWEN_SOFT}
NOTE = ("Same setting for all three: the hard end rule (only end_task ends the run) plus the example transcript in the system prompt.\n"
        "Llama 3.3 70B Instruct fp16 on CoreWeave (8 Nov); gemma-3 27B and Qwen3.6 27B fp8 on DeepInfra (9 Nov); all via OpenRouter.")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def runs(files, soft):
    """env -> dict(ends=Counter, steps=[...], opened=int, n=int), read from each run's last messages.json."""
    out = collections.defaultdict(lambda: {"ends": collections.Counter(), "steps": [], "opened": 0, "n": 0})
    for f in files:
        p = R / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            rd, env = H / r["run"], r["environment"]
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


rates = {c[0]: A.rates(c[1]) for c in THREE + [GEMMA_SOFT, QWEN_SOFT]}
per = {c[0]: runs(c[2], c[4]) for c in THREE}
rows, heads, ymax = C.layout()
envs = list(rows)


def frame(w, h, title, sub, axes):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.012, 0.985, title, fontsize=12.5, fontweight="bold", va="top", color=ST.INK)
    fig.text(0.012, 0.966, sub, fontsize=8.3, color=ST.MUTED, va="top", linespacing=1.45)
    TOP, HGT = 0.06, 0.85
    fy = lambda y: TOP + HGT * (1 - y / ymax)
    out = []
    for x0, wd, xlabel in axes:
        ax = fig.add_axes([x0, TOP, wd, HGT]); ax.set_ylim(ymax, 0); ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
        for t, y in heads:
            ax.axhline(y + 0.55, color=ST.RULE, lw=0.8, zorder=0)
        out.append(ax)
    for t, y in heads:
        fig.text(0.012, fy(y + 0.1), t, fontsize=8.5, color=ST.MUTED, fontweight="bold", va="center")
    for e in envs:
        fig.text(0.012, fy(rows[e]), C.LABEL.get(e, e) + ("  ◆" if e in C.MAIN else ""), fontsize=8.6, fontweight="bold", va="center", color=ST.INK)
    return fig, out, fy


off, hgt = [-0.27, 0.0, 0.27], 0.2

# 1. cheating: three bars per task, soft-check ticks for the stand-ins
fig, (ax,), fy = frame(15, 22, "The bad act on all 29 tasks, same setting for all three: Llama 3.3 70B, gemma-3 27B, Qwen3.6 27B",
    "Three bars per task, top to bottom: Llama 3.3 70B, gemma-3 27B, Qwen3.6 27B, every one under the hard end rule plus the example transcript.\n"
    "Bar = share of scored runs with the task's own bad act, thin line = 95% Wilson interval, k of n printed. Short vertical tick = the same model with the soft check instead\n"
    "(gemma 5-6 Nov, Qwen3.6 7-8 Nov), so a tick far from its bar means the setting changed that model. \"no value\" = no scored run in that cell.\n" + NOTE,
    [(0.30, 0.62, "share of scored runs with the bad act")])
ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for e in envs:
    y = rows[e]
    for i, (label, _, _, col, _) in enumerate(THREE):
        kn = rates[label].get(e)
        if kn is None:
            ax.text(0.8, y + off[i], "no value", fontsize=6.6, color=ST.FAINT, va="center")
        else:
            k, n = kn; p = 100 * k / n; lo, hi = wilson(k, n)
            ax.barh(y + off[i], p, height=hgt, color=col, zorder=3)
            ax.plot([100 * lo, 100 * hi], [y + off[i]] * 2, color=col, lw=0.7, alpha=0.45, zorder=2)
            ax.text(p + 1.2, y + off[i], f"{k} of {n}", fontsize=6.4, color=ST.TEXT, va="center",
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8), zorder=5)
        if label in SOFT:
            s = rates[SOFT[label][0]].get(e)
            if s:
                sp = 100 * s[0] / s[1]
                ax.plot([sp, sp], [y + off[i] - hgt * 0.75, y + off[i] + hgt * 0.75], color=col, lw=1.6, alpha=0.55, zorder=4, solid_capstyle="butt")
handles = [plt.Rectangle((0, 0), 1, 1, color=c[3], label=c[0] + ", hard rule + example") for c in THREE] + \
          [plt.Line2D([], [], color=ST.MUTED, lw=1.6, label="tick: the same stand-in with the soft check (earlier runs)")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=8.2)
fig.text(0.985, 0.004, "◆ = the 8 main tasks.", fontsize=7.6, color=ST.MUTED, ha="right", va="bottom")
fig.savefig("charts/hero8_all29_cheating.png", dpi=160); plt.close(fig)
print("wrote charts/hero8_all29_cheating.png")

# 2. work: opened a source file (left), median steps (right)
fig, (axL, axR), fy = frame(15, 20, "Did each model open the code, and how many steps did it take? All 29 tasks, same setting for all three",
    "Three bars per task, top to bottom: Llama 3.3 70B, gemma-3 27B, Qwen3.6 27B, hard end rule plus the example transcript. Left: share of runs in which the model read at least one\n"
    "source file (cat, head, grep, sed -n, a read tool...), k of n printed; in tasks without source files this is simply what the pattern finds. Right: typical steps per run (median), axis cut at 50.\n"
    + NOTE, [(0.30, 0.28, "runs that opened a source file"), (0.63, 0.30, "typical steps per run (median)")])
axL.set_xlim(0, 128); axL.set_xticks([0, 25, 50, 75, 100]); axL.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
axR.set_xlim(0, 62); axR.set_xticks([0, 10, 20, 30, 40, 50]); axR.set_xticklabels(["0", "10", "20", "30", "40", "50+"])
for e in envs:
    y = rows[e]
    for i, cond in enumerate(THREE):
        d = per[cond[0]].get(e)
        if not d or d["n"] == 0:
            axL.text(0.8, y + off[i], "no runs", fontsize=6.6, color=ST.FAINT, va="center"); continue
        p = 100 * d["opened"] / d["n"]
        axL.barh(y + off[i], p, height=hgt, color=cond[3], zorder=3)
        axL.text(p + 1.2, y + off[i], f"{d['opened']} of {d['n']}", fontsize=6.6, color=ST.TEXT, va="center")
        m = med(d["steps"]); x = min(m, 55)
        axR.barh(y + off[i], x, height=hgt, color=cond[3], zorder=3)
        axR.text(x + 0.8, y + off[i], f"{m}", fontsize=6.6, color=ST.TEXT, va="center")
handles = [plt.Rectangle((0, 0), 1, 1, color=c[3], label=c[0]) for c in THREE]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=3, fontsize=8.2)
fig.text(0.985, 0.004, "◆ = the 8 main tasks.  Step axis cut at 50.", fontsize=7.6, color=ST.MUTED, ha="right", va="bottom")
fig.savefig("charts/hero8_all29_work.png", dpi=160); plt.close(fig)
print("wrote charts/hero8_all29_work.png")

# 3. endings: three 100% bars per task
ENDS = [("end_tool", "called end_task (or the task's own done / end_session)", ST.LLAMA_HARD), ("task_end", "the task ended itself (answer, email, game over, submit)", "#8fb3e6"),
        ("reminders", "10 reminders used up", "#f0a35e"), ("cap", "step cap", "#d9534f"), ("plain", "plain reply", "#d9d4cc")]
DARK = {ST.LLAMA_HARD, "#d9534f"}
fig, (ax,), fy = frame(15, 20, "How the runs ended on all 29 tasks, same setting for all three: Llama (upper bar), gemma-3 27B (middle), Qwen3.6 27B (lower)",
    "Hard rule: a plain reply never ends the run; it ends when the model calls the finish tool, when the task ends itself after a tool call, after 10 tool-less replies in a row, or at the step cap.\n"
    "Each bar is 100% of that model's runs for the task; the count of runs is printed in each segment wide enough. Run counts on the right: Llama / gemma / Qwen3.6.\n" + NOTE,
    [(0.30, 0.55, "share of the runs")])
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for e in envs:
    y = rows[e]; ns = []
    for i, cond in enumerate(THREE):
        d = per[cond[0]].get(e)
        if not d or d["n"] == 0:
            ns.append("-"); ax.text(0.005, y + off[i], "no runs", fontsize=6.6, color=ST.FAINT, va="center"); continue
        n = d["n"]; ns.append(str(n)); x = 0
        for key, lab, col in ENDS:
            k = d["ends"].get(key, 0)
            if not k: continue
            w = k / n
            ax.barh(y + off[i], w, left=x, height=hgt, color=col, zorder=3, edgecolor="white", lw=0.6)
            if w >= 0.09:
                ax.text(x + w / 2, y + off[i], str(k), fontsize=6.2, color="white" if col in DARK else "#222", ha="center", va="center", fontweight="bold")
            x += w
    fig.text(0.87, fy(y), " / ".join(ns), fontsize=7.4, color=ST.TEXT, va="center")
fig.text(0.87, 0.06 + 0.85 + 0.004, "runs: Llama / gemma / Qwen3.6", fontsize=7.4, color=ST.MUTED, va="bottom")
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for k, l, c in ENDS[:4]]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=8.2)
fig.text(0.985, 0.004, "◆ = the 8 main tasks.  Upper bar: Llama 3.3 70B; middle: gemma-3 27B; lower: Qwen3.6 27B.", fontsize=7.6, color=ST.MUTED, ha="right", va="bottom")
fig.savefig("charts/hero8_all29_endings.png", dpi=160); plt.close(fig)
print("wrote charts/hero8_all29_endings.png")

# 4. the table and the gaps
def kn(label, e):
    v = rates[label].get(e); return f"{v[0]} of {v[1]}" if v else "no value"
def work(label, e):
    d = per[label].get(e)
    if not d or not d["n"]: return "-", "-", "-"
    c = d["ends"]; return f"{d['opened']} of {d['n']}", str(med(d["steps"])), f"{c['end_tool']}/{c['task_end']}/{c['reminders']}/{c['cap']}"
def gap(a, b, e):
    """'clear' when the two Wilson intervals do not overlap, else '' (or '-' when a cell is empty)."""
    va, vb = rates[a].get(e), rates[b].get(e)
    if not va or not vb: return "-"
    la, ha = wilson(*va); lb, hb = wilson(*vb)
    return "clear" if (ha < lb or hb < la) else ""
lines = ["# All three models on all 29 tasks in the same setting: the hard end rule plus the example transcript (9 Nov 2026)", "",
         NOTE, "",
         "Bad act = each task's own measure over the scored runs. \"Opened\" = runs with at least one read of a source file. Endings = called the finish tool / the task ended itself / 10 reminders used up / step cap. "
         "Gap columns: \"clear\" when the two models' 95% Wilson intervals do not overlap. Soft-check columns: the stand-ins' earlier runs (gemma 5-6 Nov, Qwen3.6 7-8 Nov) for comparison.", "",
         "| task | bad act, Llama | bad act, gemma | bad act, Qwen3.6 | gap Llama vs gemma | gap Llama vs Qwen3.6 | gap gemma vs Qwen3.6 | gemma, soft check | Qwen3.6, soft check | opened, Llama | opened, gemma | opened, Qwen3.6 | median steps, Llama | median steps, gemma | median steps, Qwen3.6 | endings Llama | endings gemma | endings Qwen3.6 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
tot = {c[0]: collections.Counter() for c in THREE}
gaps = collections.defaultdict(list)
for e in envs:
    w = {c[0]: work(c[0], e) for c in THREE}
    for c in THREE:
        v = rates[c[0]].get(e)
        if v: tot[c[0]]["k"] += v[0]; tot[c[0]]["n"] += v[1]
        d = per[c[0]].get(e)
        if d: tot[c[0]]["opened"] += d["opened"]; tot[c[0]]["runs"] += d["n"]; tot[c[0]].update({f"end_{k}": v for k, v in d["ends"].items()})
    g = {("Llama", "gemma"): gap(LLAMA[0], GEMMA[0], e), ("Llama", "Qwen3.6"): gap(LLAMA[0], QWEN[0], e), ("gemma", "Qwen3.6"): gap(GEMMA[0], QWEN[0], e)}
    for pair, val in g.items():
        if val == "clear": gaps[pair].append(e)
    lines.append(f"| {C.LABEL.get(e, e)}{' ◆' if e in C.MAIN else ''} | {kn(LLAMA[0], e)} | {kn(GEMMA[0], e)} | {kn(QWEN[0], e)} | {g[('Llama', 'gemma')]} | {g[('Llama', 'Qwen3.6')]} | {g[('gemma', 'Qwen3.6')]} | "
                 f"{kn(GEMMA_SOFT[0], e)} | {kn(QWEN_SOFT[0], e)} | {w[LLAMA[0]][0]} | {w[GEMMA[0]][0]} | {w[QWEN[0]][0]} | {w[LLAMA[0]][1]} | {w[GEMMA[0]][1]} | {w[QWEN[0]][1]} | {w[LLAMA[0]][2]} | {w[GEMMA[0]][2]} | {w[QWEN[0]][2]} |")
lines += ["", "## Totals", "", "| model | bad acts / scored runs (all tasks pooled) | opened a source file / runs read | ended: end tool / task's end / reminders / cap / plain |", "|---|---|---|---|"]
for c in THREE:
    t = tot[c[0]]
    lines.append(f"| {c[0]} | {t['k']} of {t['n']} | {t['opened']} of {t['runs']} | {t['end_end_tool']} / {t['end_task_end']} / {t['end_reminders']} / {t['end_cap']} / {t['end_plain']} |")
lines += ["", "## Clear gaps (95% intervals do not overlap)", ""]
for pair, es in gaps.items():
    lines.append(f"- **{pair[0]} vs {pair[1]}** ({len(es)} tasks): " + ", ".join(f"{C.LABEL.get(e, e)} ({kn(LLAMA[0] if pair[0] == 'Llama' else GEMMA[0], e)} vs {kn(GEMMA[0] if pair[1] == 'gemma' else QWEN[0], e)})" for e in es))
pathlib.Path("results/hero8_all29.md").write_text("\n".join(lines) + "\n")
print("wrote results/hero8_all29.md")
for c in THREE:
    print(c[0], dict(tot[c[0]]))
for pair, es in gaps.items():
    print("clear gap", pair, es)
