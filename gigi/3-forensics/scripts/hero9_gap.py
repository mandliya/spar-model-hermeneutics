"""The two heroes, no check, all 29 tasks (10 Nov 2026 call: Llama is dropped; Qwen3.6 27B and gemma-3 27B are the heroes
and run as the paper's environments run them, no end check, no hard rule, no hack). Three charts and a table:
  charts/hero9_gap_cheating.png  per task, the bad-act rate (k of n, 95% Wilson interval) of gemma-3 27B and Qwen3.6 27B;
  charts/hero9_gap_work.png      per task, left: share of runs that opened a source file; right: median steps per run;
  charts/hero9_gap_endings.png   per task, how the runs ended (plain reply / finish tool / task's own end / step cap);
  results/hero9_gap.md           the table, totals, the list of clear gaps (intervals that do not overlap), losses, cost;
  results/hero9_loss_audit__gemma-3-27b.md  gemma's losses, copied from results/full_runs.csv (see below).
Qwen3.6 27B: the hero9 stages (hero9_qwen36, hero9_qwen36_key, hero9_qwen36_all; results/hero9_full_runs__qwen3.6-27b*.csv,
scored by hero_solo_foldin.sh). gemma-3 27B: its original no-check passes of 19-25 Oct (results/full_runs.csv for the
rates; full_runs_table.modal_runs() and laptop_runs("chess") for the run folders), because every cell there is already at
the hero target (results/logs/hero9/gemma_topup_plan.md) and the hero scorer cannot read those older stages.
Runs with the Qwen3.6 data missing: says so, draws gemma, leaves the rest out.
Run from gigi/3-forensics/agent-interp-envs:  env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero9_gap.py
"""
import collections, csv, json, math, os, pathlib, re, sys, types
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import chart_style as ST
import confirm_effect_charts as C
import full_runs_table as T
R = H / "results"

READ_RE = re.compile(r"\b(cat|less|head|tail|sed -n|nl|bat|view|open)\b[^|;&\n]*\.(py|js|mjs|cjs|ts|tsx|jsx|c|cc|cpp|h|rs|go|java|json|yaml|yml|toml|md|txt|sh)\b|"
                     r"\b(grep|rg|ag)\b|\bread_file\b|\bview_file\b", re.I)
FINISH = ("end_task", "done", "end_session")
GEMMA, QWEN = "gemma-3 27B", "Qwen3.6 27B, thinking off"
COL = {GEMMA: ST.GEMMA, QWEN: ST.QWEN}
QWEN_CSV = ["hero9_full_runs__qwen3.6-27b.csv", "hero9_full_runs__qwen3.6-27b__other21.csv"]
QWEN_PER = ["hero9_full_runs__qwen3.6-27b_per_run.csv", "hero9_full_runs__qwen3.6-27b__other21_per_run.csv"]
NOTE = ("No end check, no hard rule, no example: the environments as the paper runs them. gemma-3 27B: the original passes of 19-25 Oct; "
        "Qwen3.6 27B: 10 Nov.\nBoth fp8 on DeepInfra via OpenRouter, Qwen3.6 with thinking off, gemma with text-mode tool calls.")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def read_run(env, rd, acc, tag):
    """Add one run's steps, ending and 'opened a source file' to acc[env], from its last step-*/messages.json."""
    ms = sorted(rd.glob("step-*/messages.json"), key=lambda q: int(re.search(r"step-(\d+)", str(q)).group(1)))
    if not ms:
        return
    try:
        m = json.load(open(ms[-1]))
    except Exception:
        return
    n = len(ms)
    asst = [x for x in m if x.get("role") == "assistant"]
    last = asst[-1] if asst else {}
    lastcalls = [(t.get("function") or {}).get("name") for t in last.get("tool_calls") or []]
    cap = C.limit(env, tag) or 10 ** 9   # the task's step cap from configs/<env>__<tag>.yaml
    if n >= cap: k = "cap"
    elif any(c in FINISH for c in lastcalls): k = "end_tool"
    elif not lastcalls: k = "plain"
    else: k = "task_end"
    args = " \n ".join(((t.get("function") or {}).get("arguments") or "") for x in asst for t in x.get("tool_calls") or [])
    args += " \n " + " \n ".join(str(x.get("content") or "") for x in asst if not x.get("tool_calls"))
    d = acc[env]; d["ends"][k] += 1; d["steps"].append(n); d["opened"] += bool(READ_RE.search(args)); d["n"] += 1


def fresh():
    return collections.defaultdict(lambda: {"ends": collections.Counter(), "steps": [], "opened": 0, "n": 0})


def qwen_runs():
    acc = fresh()
    for f in QWEN_PER:
        p = R / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if r["environment"] in C.ENVS:
                read_run(r["environment"], H / r["run"], acc, "qwen3.6-27b")
    return acc


def gemma_runs():
    """gemma's original no-check runs: Modal fleets of its STAGES, plus the laptop pass for chess (no engine on Modal)."""
    acc = fresh(); seen = set()
    for env, m, rd in T.modal_runs():
        if m == "gemma-3-27b" and env in C.ENVS and T.status(rd) == "finished":
            seen.add(env); read_run(env, rd, acc, "gemma-3-27b")
    if "chess" not in seen:
        for env, m, rd in T.laptop_runs("chess"):
            if m == "gemma-3-27b" and T.status(rd) == "finished":
                read_run("chess", rd, acc, "gemma-3-27b")
    return acc


def qwen_rates():
    out = {}
    for f in QWEN_CSV:
        p = R / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if r["environment"] not in out and int(r["n"] or 0) > 0:
                out[r["environment"]] = (int(r["k"]), int(r["n"]))
    return out


def gemma_losses():
    """env -> (unscored, lost_provider, lost_crash) from results/full_runs.csv."""
    out = {}
    for r in csv.DictReader(open(R / "full_runs.csv")):
        if r["model"] == "gemma-3-27b":
            out[r["environment"]] = (int(r["unscored"] or 0), int(r["lost_provider"] or 0), int(r["lost_crash_or_harness"] or 0), int(r["n"] or 0))
    return out


qwen_present = all((R / f).exists() for f in QWEN_CSV)
if not qwen_present:
    print("Qwen3.6 hero9 tables not on disk yet:", ", ".join(f for f in QWEN_CSV if not (R / f).exists()), "(drawn without them)")
rates = {GEMMA: C.rates("gemma-3-27b", after=False), QWEN: qwen_rates()}
per = {GEMMA: gemma_runs(), QWEN: qwen_runs()}
MODELS = [GEMMA, QWEN]
rows, heads, ymax = C.layout()
envs = list(rows)
MISSNOTE = "  Qwen3.6 27B not on disk yet." if not qwen_present else ""


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


off, hgt = [-0.19, 0.19], 0.3


def gap(e):
    va, vb = rates[GEMMA].get(e), rates[QWEN].get(e)
    if not va or not vb: return "-"
    la, ha = wilson(*va); lb, hb = wilson(*vb)
    return "clear" if (ha < lb or hb < la) else ""


# 1. cheating
fig, (ax,), fy = frame(15, 22, "The bad act on all 29 tasks with no end check: gemma-3 27B vs Qwen3.6 27B, the two heroes",
    "Two bars per task: gemma-3 27B above, Qwen3.6 27B below. Bar = share of scored runs with the task's own bad act, thin line = 95% Wilson interval, k of n printed.\n"
    "\"clear gap\" on the right = the two intervals do not overlap. \"no value\" = no scored run in that cell.\n" + NOTE,
    [(0.30, 0.56, "share of scored runs with the bad act")])
ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for e in envs:
    y = rows[e]
    for i, label in enumerate(MODELS):
        kn = rates[label].get(e)
        if kn is None:
            if label == QWEN and not qwen_present: continue
            ax.text(0.8, y + off[i], "no value", fontsize=6.6, color=ST.FAINT, va="center"); continue
        k, n = kn; p = 100 * k / n; lo, hi = wilson(k, n)
        ax.barh(y + off[i], p, height=hgt, color=COL[label], zorder=3)
        ax.plot([100 * lo, 100 * hi], [y + off[i]] * 2, color=COL[label], lw=0.7, alpha=0.45, zorder=2)
        ax.text(p + 1.2, y + off[i], f"{k} of {n}", fontsize=6.6, color=ST.TEXT, va="center",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8), zorder=5)
    if gap(e) == "clear":
        fig.text(0.875, fy(y), "clear gap", fontsize=7.6, color=ST.INK, fontweight="bold", va="center")
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m], label=m) for m in MODELS]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=8.2)
fig.text(0.985, 0.004, "◆ = the 8 main tasks." + MISSNOTE, fontsize=7.6, color=ST.MUTED, ha="right", va="bottom")
fig.savefig("charts/hero9_gap_cheating.png", dpi=160); plt.close(fig)
print("wrote charts/hero9_gap_cheating.png")

# 2. work
fig, (axL, axR), fy = frame(15, 20, "Did each hero open the code, and how many steps did it take? All 29 tasks, no end check",
    "Two bars per task: gemma-3 27B above, Qwen3.6 27B below. Left: share of finished runs in which the model read at least one source file (cat, head, grep, sed -n, a read tool...),\n"
    "k of n printed; in tasks without source files this is simply what the pattern finds. Right: typical steps per run (median), axis cut at 50.\n" + NOTE,
    [(0.30, 0.28, "runs that opened a source file"), (0.63, 0.30, "typical steps per run (median)")])
axL.set_xlim(0, 128); axL.set_xticks([0, 25, 50, 75, 100]); axL.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
axR.set_xlim(0, 62); axR.set_xticks([0, 10, 20, 30, 40, 50]); axR.set_xticklabels(["0", "10", "20", "30", "40", "50+"])
def med(v):
    v = sorted(v); return v[len(v) // 2] if v else None
for e in envs:
    y = rows[e]
    for i, label in enumerate(MODELS):
        d = per[label].get(e)
        if not d or d["n"] == 0:
            if not (label == QWEN and not qwen_present):
                axL.text(0.8, y + off[i], "no runs", fontsize=6.6, color=ST.FAINT, va="center")
            continue
        p = 100 * d["opened"] / d["n"]
        axL.barh(y + off[i], p, height=hgt, color=COL[label], zorder=3)
        axL.text(p + 1.2, y + off[i], f"{d['opened']} of {d['n']}", fontsize=6.6, color=ST.TEXT, va="center")
        m = med(d["steps"]); x = min(m, 55)
        axR.barh(y + off[i], x, height=hgt, color=COL[label], zorder=3)
        axR.text(x + 0.8, y + off[i], f"{m}", fontsize=6.6, color=ST.TEXT, va="center")
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=8.2)
fig.text(0.985, 0.004, "◆ = the 8 main tasks.  Step axis cut at 50." + MISSNOTE, fontsize=7.6, color=ST.MUTED, ha="right", va="bottom")
fig.savefig("charts/hero9_gap_work.png", dpi=160); plt.close(fig)
print("wrote charts/hero9_gap_work.png")

# 3. endings
ENDS = [("plain", "plain reply (the run ends on the first message without a tool call)", "#d9d4cc"),
        ("end_tool", "called the task's finish tool (done / end_session / end_task)", ST.LLAMA_HARD),
        ("task_end", "the task ended itself (answer, email, game over, submit)", "#8fb3e6"), ("cap", "step cap", "#d9534f")]
DARK = {ST.LLAMA_HARD, "#d9534f"}
fig, (ax,), fy = frame(15, 20, "How the heroes' runs ended on all 29 tasks with no end check: gemma-3 27B (upper bar), Qwen3.6 27B (lower)",
    "No check: the first reply without a tool call ends the run. Each bar is 100% of that model's finished runs for the task; the count of runs is printed in each segment wide enough.\n"
    "Run counts on the right: gemma / Qwen3.6.\n" + NOTE, [(0.30, 0.55, "share of the runs")])
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for e in envs:
    y = rows[e]; ns = []
    for i, label in enumerate(MODELS):
        d = per[label].get(e)
        if not d or d["n"] == 0:
            ns.append("-")
            if not (label == QWEN and not qwen_present):
                ax.text(0.005, y + off[i], "no runs", fontsize=6.6, color=ST.FAINT, va="center")
            continue
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
fig.text(0.87, 0.06 + 0.85 + 0.004, "runs: gemma / Qwen3.6", fontsize=7.4, color=ST.MUTED, va="bottom")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for k, l, c in ENDS], loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=8.2)
fig.text(0.985, 0.004, "◆ = the 8 main tasks.  Upper bar: gemma-3 27B; lower: Qwen3.6 27B." + MISSNOTE, fontsize=7.6, color=ST.MUTED, ha="right", va="bottom")
fig.savefig("charts/hero9_gap_endings.png", dpi=160); plt.close(fig)
print("wrote charts/hero9_gap_endings.png")

# 4. the table, the gaps, the losses, the cost
def kn(label, e):
    v = rates[label].get(e); return f"{v[0]} of {v[1]}" if v else "no value"
def work(label, e):
    d = per[label].get(e)
    if not d or not d["n"]: return "-", "-", "-"
    c = d["ends"]; return f"{d['opened']} of {d['n']}", str(med(d["steps"])), f"{c['plain']}/{c['end_tool']}/{c['task_end']}/{c['cap']}"
gl = gemma_losses()
cost = (H / "results/logs/hero9/cost.txt").read_text().strip() if (H / "results/logs/hero9/cost.txt").exists() else "not recorded yet"
lines = ["# The two heroes with no end check on all 29 tasks: gemma-3 27B vs Qwen3.6 27B (10 Nov 2026)", "",
         NOTE.replace("\n", " "), "",
         "Bad act = each task's own measure over the scored runs. \"Opened\" = finished runs with at least one read of a source file. Endings = plain reply / finish tool / the task ended itself / step cap. "
         "Gap = \"clear\" when the two 95% Wilson intervals do not overlap. gemma's rates come from results/full_runs.csv (its original no-check passes, every cell at the hero target: "
         "results/logs/hero9/gemma_topup_plan.md); Qwen3.6's from results/hero9_full_runs__qwen3.6-27b*.csv.", "",
         "| task | bad act, gemma | bad act, Qwen3.6 | gap | opened, gemma | opened, Qwen3.6 | median steps, gemma | median steps, Qwen3.6 | endings gemma (plain/finish/task/cap) | endings Qwen3.6 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
tot = {m: collections.Counter() for m in MODELS}; gaps = []
for e in envs:
    w = {m: work(m, e) for m in MODELS}
    for m in MODELS:
        v = rates[m].get(e)
        if v: tot[m]["k"] += v[0]; tot[m]["n"] += v[1]
        d = per[m].get(e)
        if d: tot[m]["opened"] += d["opened"]; tot[m]["runs"] += d["n"]; tot[m].update({f"end_{k}": v for k, v in d["ends"].items()})
    g = gap(e)
    if g == "clear": gaps.append(e)
    lines.append(f"| {C.LABEL.get(e, e)}{' ◆' if e in C.MAIN else ''} | {kn(GEMMA, e)} | {kn(QWEN, e)} | {g} | {w[GEMMA][0]} | {w[QWEN][0]} | {w[GEMMA][1]} | {w[QWEN][1]} | {w[GEMMA][2]} | {w[QWEN][2]} |")
lines += ["", "## Totals", "", "| model | bad acts / scored runs (all tasks pooled) | opened a source file / finished runs read | ended: plain reply / finish tool / task's end / cap |", "|---|---|---|---|"]
for m in MODELS:
    t = tot[m]
    lines.append(f"| {m} | {t['k']} of {t['n']} | {t['opened']} of {t['runs']} | {t['end_plain']} / {t['end_end_tool']} / {t['end_task_end']} / {t['end_cap']} |")
lines += ["", "## Clear gaps (95% intervals do not overlap)", ""]
lines.append((f"**{len(gaps)} tasks:** " + ", ".join(f"{C.LABEL.get(e, e)} (gemma {kn(GEMMA, e)}, Qwen3.6 {kn(QWEN, e)})" for e in gaps)) if gaps else "none yet" + MISSNOTE)
lines += ["", "## Losses", "",
          "gemma (results/full_runs.csv): " + "; ".join(f"{e}: {u} unscored, {lp} provider, {lc} crash" for e, (u, lp, lc, n) in sorted(gl.items()) if u or lp or lc) + ".",
          "Qwen3.6: results/hero9_loss_audit__qwen3.6-27b.md and results/hero9_loss_audit__qwen3.6-27b__other21.md" + (" (not on disk yet)." if not qwen_present else "."),
          "", "## Cost", "", cost]
pathlib.Path("results/hero9_gap.md").write_text("\n".join(lines) + "\n")
print("wrote results/hero9_gap.md")

# 5. gemma's loss audit, copied from the canonical table
al = ["# Hero loss audit: gemma-3 27B with no end check (the original passes of 19-25 Oct 2026), written by scripts/hero9_gap.py", "",
      "gemma was not relaunched for hero9: every cell of its original no-check passes is at the hero target (results/logs/hero9/gemma_topup_plan.md). "
      "This table copies the scored / unscored / lost columns of results/full_runs.csv for gemma-3-27b (fp8, DeepInfra via OpenRouter, text-mode tools).", "",
      "| environment | scored | unscored (no value for the measure) | lost, provider | lost, crash or harness |", "|---|---|---|---|---|"]
for e, (u, lp, lc, n) in sorted(gl.items()):
    al.append(f"| {e} | {n} | {u} | {lp} | {lc} |")
al += ["", f"Total: scored {sum(v[3] for v in gl.values())}, unscored {sum(v[0] for v in gl.values())}, lost to the provider {sum(v[1] for v in gl.values())}, lost to a crash or the harness {sum(v[2] for v in gl.values())}."]
pathlib.Path("results/hero9_loss_audit__gemma-3-27b.md").write_text("\n".join(al) + "\n")
print("wrote results/hero9_loss_audit__gemma-3-27b.md")
for m in MODELS:
    print(m, dict(tot[m]))
print("clear gaps:", gaps)
