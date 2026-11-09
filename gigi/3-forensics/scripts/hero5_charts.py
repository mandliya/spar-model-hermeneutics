"""Charts and summary for the five-hack test (hero5, 8 Nov 2026): Llama 3.3 70B (fp16, CoreWeave) under the hard end rule
plus one hack at a time (agent.hack, agent-interp-envs/src/agent_interp_envs/hacks.py): do_the_work, no_early_exit,
proof_of_work, plan_first, worked_example. Five tasks: coding, coding_norules, dashboard_costly_failure, grade_your_own,
optimize_rendering_engine. The baseline is the hard rule alone (stage hero4_llama, 7 Nov).

Writes
  charts/hero5_real_work.png   per task, share of runs in which Llama opened a source file, baseline + five hacks
  charts/hero5_cheating.png    per task, the bad-act rate with Wilson intervals, same layout
  charts/hero5_endings.png     per condition, how the runs ended (stacked counts) and typical steps / refusals
  results/hero5_summary.md     table per task x condition, plus a paragraph template to fill in

Run from gigi/3-forensics/agent-interp-envs:
  env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero5_charts.py [--stages hero4_llama,hero5_do_the_work,...]
Default stages: hero4_llama plus the five hero5_<hack> stages. A stage with no runs yet is reported and left out, so the
script runs before the data lands (e.g. `--stages hero4_llama` draws the baseline alone).
"""
import argparse, collections, csv, json, math, os, pathlib, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chart_style as ST   # font, ink colours, soft grid: one look for every chart

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import full_runs_table as T
R = H / "results"; OUT = pathlib.Path(os.environ.get("CHART_OUT", H / "charts"))
MODEL = "llama-3.3-70b"
MODEL_LINE = "Llama 3.3 70B, fp16 on CoreWeave via OpenRouter"

TASKS = ["coding", "coding_norules", "dashboard_costly_failure", "grade_your_own", "optimize_rendering_engine"]
TASK_LABEL = {"coding": "coding (is_prime)", "coding_norules": "coding, no rules stated", "dashboard_costly_failure": "dashboard (speed target)",
              "grade_your_own": "grade your own model", "optimize_rendering_engine": "rendering engine (3x speed-up)"}
# (scripts/confirm_effect_charts.LABEL has longer names like "coding (ImpossibleBench)"; the short ones above fit the margin)

# condition key -> (label, stage, summary csv candidates, colour)
CONDITIONS = {
    "hero4_llama": ("hard rule alone (baseline, 7 Nov)", "hero4_llama", ["hero4_full_runs__llama-3.3-70b.csv"], "#8a8378"),
    "hero5_do_the_work": ("+ do the work (prompt)", "hero5_do_the_work", ["hero5_full_runs__llama-3.3-70b__do_the_work.csv"], "#845ef7"),
    "hero5_no_early_exit": ("+ no early exit (end refused before 10 steps)", "hero5_no_early_exit", ["hero5_full_runs__llama-3.3-70b__no_early_exit.csv"], "#339af0"),
    "hero5_proof_of_work": ("+ proof of work (end refused until a file changed)", "hero5_proof_of_work", ["hero5_full_runs__llama-3.3-70b__proof_of_work.csv"], "#20c997"),
    "hero5_plan_first": ("+ plan first (plan, then progress lines)", "hero5_plan_first", ["hero5_full_runs__llama-3.3-70b__plan_first.csv"], "#e67700"),
    "hero5_worked_example": ("+ worked example (prompt)", "hero5_worked_example", ["hero5_full_runs__llama-3.3-70b__worked_example.csv"], "#e64980"),
}
ORDER = list(CONDITIONS)

READ_RE = re.compile(r"\b(cat|less|head|tail|sed -n|nl|bat|view|open)\b[^|;&\n]*\.(py|js|mjs|cjs|ts|tsx|jsx|c|cc|cpp|h|rs|go|java|json|yaml|yml|toml|md|txt|sh)\b|"
                     r"\b(grep|rg|ag)\b|\bread_file\b|\bview_file\b", re.I)
WRITE_RE = re.compile(r"(cat\s*>|tee\s|>\s*\S+\.(py|js|ts|c|h|json|yaml|yml|txt|md|sh)|<<\s*['\"]?EOF|\bsed -i\b|apply_patch|write_file|"
                      r"echo .*>\s*\S|python\s*-\s*<<|git commit|cp\s|mv\s)", re.I)
REMINDER = "A reply without a tool call does not end the session"
REFUSAL = ("Too early to finish", "Not finished: no file")
END_TOOLS = ("end_task", "done")

ENDING_KINDS = [("end_tool", "ended with end_task (or the task's own done)", "#845ef7"),
                ("task_end", "the task ended itself after a tool call", "#8fb3e6"),
                ("reminders", "all 10 reminders used, then let go", "#f0a35e"),
                ("cap", "ran to the task's step cap", "#d9534f"),
                ("plain", "ended on a plain reply", "#d9d4cc")]


def txt(x):
    c = x.get("content"); return c if isinstance(c, str) else ("" if c is None else json.dumps(c))


def calls(x):
    return [((t.get("function") or {}).get("name") or t.get("name"), (t.get("function") or {}).get("arguments") or "")
            for t in x.get("tool_calls") or []]


def load_msgs(rd):
    f = rd / "final" / "messages.json"
    steps = sorted(rd.glob("step-*/messages.json"), key=lambda p: int(re.search(r"step-(\d+)", str(p)).group(1)))
    if not f.exists():
        if not steps:
            return None, 0
        f = steps[-1]
    try:
        return json.load(open(f)), len(steps)
    except Exception:
        return None, len(steps)


def max_steps(env):
    m = re.search(r"max_steps:\s*(\d+)", (H / "configs" / f"{env}__{MODEL}.yaml").read_text())
    return int(m.group(1)) if m else 10**9


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def per_run_rows(cond):
    """stage's rows from its per-run csv (outcome scored/lost per run), keyed by run dir string."""
    label, stage, files, _ = CONDITIONS[cond]
    for f in files:
        p = R / f.replace(".csv", "_per_run.csv")
        if p.exists():
            return {r["run"].rstrip("/"): r for r in csv.DictReader(open(p)) if r["stage"] == stage}
    # fallback: an unsuffixed per-model file that carries several stages
    for f in (f"{stage.split('_')[0]}_full_runs__{MODEL}_per_run.csv",):
        p = R / f
        if p.exists():
            return {r["run"].rstrip("/"): r for r in csv.DictReader(open(p)) if r["stage"] == stage}
    return {}


def rates(cond):
    """env -> (k, n) from the summary csv; envs with n == 0 are left out ("no value")."""
    label, stage, files, _ = CONDITIONS[cond]
    out = {}
    for f in files + [f"{stage.split('_')[0]}_full_runs__{MODEL}.csv"]:
        p = R / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if stage not in (r.get("stages") or "").split("+"):
                continue
            if r.get("n") and int(r["n"]) > 0 and r["environment"] not in out:
                out[r["environment"]] = (int(r["k"]), int(r["n"]))
        if out:
            break
    return out


def scan(cond):
    """Read every run of the condition: per env, counts of opened/wrote a file, endings, steps, refusals, reminders."""
    label, stage, _, _ = CONDITIONS[cond]
    per = per_run_rows(cond)
    d = {e: dict(n=0, opened=0, wrote=0, endings=collections.Counter(), steps=[], refusals=[], reminders=[], scored=0, lost=0)
         for e in TASKS}
    for env, _, rd, _ in T.hero_runs((stage,), MODEL):
        if env not in TASKS:
            continue
        m, nsteps = load_msgs(pathlib.Path(rd))
        row = per.get(str(pathlib.Path(rd).relative_to(H)))
        if row:
            d[env]["scored" if row["outcome"] == "scored" else "lost"] += 1
        if m is None:
            continue
        x = d[env]; x["n"] += 1
        asst = [a for a in m if a.get("role") == "assistant"]
        args = " \n ".join(a for q in asst for _, a in calls(q)) + " \n " + " \n ".join(txt(q) for q in asst if not calls(q))
        x["opened"] += bool(READ_RE.search(args)); x["wrote"] += bool(WRITE_RE.search(args))
        last = asst[-1] if asst else {}
        lastcalls = [c for c, _ in calls(last)]
        # a deliberate end anywhere in the run counts (under the fixed harness it is always the last call; in the 7 Nov
        # baseline the coding loop re-prompted after it, so the last call there can be a command)
        if any(c in END_TOOLS for q in asst for c, _ in calls(q)):
            lastcalls = list(lastcalls) + ["end_task"]
        rem = sum(1 for q in m if q.get("role") == "user" and txt(q).startswith(REMINDER))
        ref = sum(1 for q in m if q.get("role") == "tool" and txt(q).startswith(REFUSAL))
        if nsteps >= max_steps(env):
            k = "cap"
        elif any(c in END_TOOLS for c in lastcalls):
            k = "end_tool"
        elif rem >= 10:
            k = "reminders"
        elif not lastcalls:
            k = "plain"
        else:
            k = "task_end"
        x["endings"][k] += 1; x["steps"].append(nsteps); x["refusals"].append(ref); x["reminders"].append(rem)
    return d


def med(v):
    v = sorted(v); return v[len(v) // 2] if v else None


def frame(title, sub, xlabel, n_rows):
    fig = plt.figure(figsize=(12, 1.6 + 1.25 * n_rows))
    fig.text(0.015, 0.975, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.015, 0.935, sub, fontsize=8.5, color="#555", va="top", linespacing=1.4)
    ax = fig.add_axes([0.26, 0.20, 0.50, 0.66])
    ax.set_xlim(-2, 118); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(xlabel, fontsize=9.5, color="#444")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eee", lw=0.8); ax.set_axisbelow(True)
    return fig, ax


def dots_chart(conds, values, path, title, sub, xlabel, with_ci=False):
    """values[(cond, env)] = (k, n) or None (no value)."""
    fig, ax = frame(title, sub, xlabel, len(TASKS))
    ys = list(range(len(TASKS)))[::-1]
    span = 0.72; offs = [span / 2 - span * i / max(1, len(conds) - 1) for i in range(len(conds))] if len(conds) > 1 else [0]
    for y, env in zip(ys, TASKS):
        ax.axhline(y - 0.5, color="#eee", lw=0.8, zorder=0)
        for off, cond in zip(offs, conds):
            col = CONDITIONS[cond][3]; v = values.get((cond, env))
            if not v or v[1] == 0:
                ax.text(0, y + off, "no value", fontsize=7, color="#aaa", va="center")
                continue
            k, n = v; x = 100 * k / n
            if with_ci:
                lo, hi = wilson(k, n); ax.plot([100 * lo, 100 * hi], [y + off, y + off], color=col, lw=1, alpha=0.6, zorder=2)
            ax.plot(x, y + off, "o", color=col, ms=7, zorder=3)
            tx = max(x, 100 * wilson(k, n)[1]) if with_ci else x
            ax.text(min(tx, 100) + 2, y + off, f"{k} of {n}", fontsize=7.5, color=col, va="center",
                    fontweight="bold" if cond != "hero4_llama" else None)
    ax.set_yticks(ys); ax.set_yticklabels([TASK_LABEL[e] for e in TASKS], fontsize=9.5, fontweight="bold")
    ax.set_ylim(-0.5, len(TASKS) - 0.5)
    handles = [plt.Line2D([], [], marker="o", color=CONDITIONS[c][3], ls="", ms=7, label=CONDITIONS[c][0]) for c in conds]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.008), ncol=2, fontsize=8.2, frameon=False)
    OUT.mkdir(exist_ok=True); fig.savefig(path, dpi=160, facecolor="white"); plt.close(fig); print("wrote", path)


def endings_chart(conds, scans, path):
    rows = conds
    h = 2.8 + 0.75 * len(rows); fig = plt.figure(figsize=(12, h))
    fig.text(0.015, 0.975, "How Llama's runs ended under the hard rule alone and with each hack (the five tasks together)",
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.015, 0.935, f"{MODEL_LINE}. One bar per condition, all five tasks pooled; the colours say how each run ended. On the right: the typical "
             "(median) number of steps,\nthe median number of refused end calls per run (no early exit and proof of work only), and the number of runs read.",
             fontsize=8.5, color="#555", va="top", linespacing=1.4)
    ax = fig.add_axes([0.30, 1.15 / h, 0.42, 1 - 1.15 / h - 1.0 / h])
    ys = list(range(len(rows)))[::-1]; ax.set_ylim(-0.6, len(rows) - 0.4)
    for y, cond in zip(ys, rows):
        tot = collections.Counter(); steps, refs, n = [], [], 0
        for env in TASKS:
            x = scans[cond][env]; tot.update(x["endings"]); steps += x["steps"]; refs += x["refusals"]; n += x["n"]
        left = 0
        for k, lab, col in ENDING_KINDS:
            v = tot.get(k, 0)
            if v:
                ax.barh(y, v, left=left, color=col, height=0.55, edgecolor="white", lw=1)
                if v >= 3:
                    ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=7.5, color="white" if k in ("end_tool", "cap") else "#333")
                left += v
        ms, mr = med(steps), med(refs)
        ax.text(left + 1.5, y, f"typical steps {ms if ms is not None else '-'}; refusals {mr if mr is not None else '-'} per run; {n} runs",
                fontsize=7.8, color="#444", va="center")
    ax.set_yticks(ys); ax.set_yticklabels([CONDITIONS[c][0] for c in rows], fontsize=9, fontweight="bold")
    ax.set_xlabel("runs", fontsize=9.5, color="#444"); ax.set_xlim(0, max(1, max(sum(sum(scans[c][e]["endings"].values()) for e in TASKS) for c in rows)) * 1.9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eee", lw=0.8); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for _, lab, col in ENDING_KINDS]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.008), ncol=3, fontsize=8.2, frameon=False)
    OUT.mkdir(exist_ok=True); fig.savefig(path, dpi=160, facecolor="white"); plt.close(fig); print("wrote", path)


def kn(v):
    return f"{v[0]} of {v[1]}" if v and v[1] else "no value"


def summary_md(conds, scans, bad, path):
    L = ["# The five hard hacks on Llama 3.3 70B: summary", "",
         f"{MODEL_LINE}, hard end rule on (`agent.end_task_required=true`, 10 reminders) plus one hack at a time (`agent.hack`, "
         "`agent-interp-envs/src/agent_interp_envs/hacks.py`). Baseline: the hard rule alone (stage hero4_llama, 7 Nov). "
         "Five tasks, 10/10/10/5/5 runs per condition. Charts: `charts/hero5_real_work.png`, `charts/hero5_cheating.png`, `charts/hero5_endings.png`. "
         "Made by `scripts/hero5_charts.py`.", "",
         "<!-- paragraph to fill in once the runs are read -->",
         "**What the hacks changed, in one paragraph:** _[fill in: did any hack make Llama open the code? which one moved the bad-act rate, "
         "and in which task? how did the runs end (deliberate end, step cap, refusals used up)? cost of the runs and of the grading]_", "",
         "\"Opened the code\" = the run has at least one read of a source file (cat, head, grep, sed -n, a read tool...); \"wrote\" = at least one "
         "write to a file. \"Bad act\" is each task's own measure from the summary csvs (n = scored runs). \"Refusals\" = end calls the hack refused "
         "(no early exit, proof of work). \"Reminders\" = hard-rule reminders after a reply without a tool call. Typical = median.", ""]
    for env in TASKS:
        L += [f"## {TASK_LABEL[env]}", "",
              "| condition | runs read | scored / lost | opened the code | wrote a file | bad act | typical steps | typical refusals | typical reminders | ended: end tool / task's end / reminders / cap |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in conds:
            x = scans[c][env]; e = x["endings"]
            L.append(f"| {CONDITIONS[c][0]} | {x['n']} | {x['scored']} / {x['lost']} | {x['opened']} of {x['n']} | {x['wrote']} of {x['n']} | "
                     f"{kn(bad.get((c, env)))} | {med(x['steps']) if x['steps'] else '-'} | {med(x['refusals']) if x['refusals'] else '-'} | "
                     f"{med(x['reminders']) if x['reminders'] else '-'} | {e.get('end_tool', 0)} / {e.get('task_end', 0)} / {e.get('reminders', 0)} / {e.get('cap', 0)} |")
        L.append("")
    L += ["## All five tasks together", "", "| condition | runs read | opened the code | wrote a file | bad act (pooled) | typical steps | typical refusals | ended: end tool / task's end / reminders / cap |",
          "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        n = sum(scans[c][e]["n"] for e in TASKS); o = sum(scans[c][e]["opened"] for e in TASKS); w = sum(scans[c][e]["wrote"] for e in TASKS)
        bk = sum(bad[(c, e)][0] for e in TASKS if bad.get((c, e))); bn = sum(bad[(c, e)][1] for e in TASKS if bad.get((c, e)))
        steps = sum((scans[c][e]["steps"] for e in TASKS), []); refs = sum((scans[c][e]["refusals"] for e in TASKS), [])
        en = collections.Counter(); [en.update(scans[c][e]["endings"]) for e in TASKS]
        L.append(f"| {CONDITIONS[c][0]} | {n} | {o} of {n} | {w} of {n} | {bk} of {bn} | {med(steps) if steps else '-'} | {med(refs) if refs else '-'} | "
                 f"{en.get('end_tool', 0)} / {en.get('task_end', 0)} / {en.get('reminders', 0)} / {en.get('cap', 0)} |")
    path.write_text("\n".join(L) + "\n"); print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ORDER), help="comma list of stages (hero4_llama, hero5_<hack>); default: all six")
    a = ap.parse_args()
    wanted = [s for s in a.stages.split(",") if s]
    unknown = [s for s in wanted if s not in CONDITIONS]
    if unknown:
        sys.exit(f"unknown stages {unknown}; known: {ORDER}")
    conds = [c for c in ORDER if c in wanted]
    scans, present = {}, []
    for c in conds:
        d = scan(c)
        if sum(x["n"] for x in d.values()) == 0:
            print(f"no runs yet for {c} (stage {CONDITIONS[c][1]}): left out")
            continue
        scans[c] = d; present.append(c)
    if not present:
        sys.exit("nothing to draw")
    bad = {}
    for c in present:
        r = rates(c)
        if not r:
            print(f"no summary csv yet for {c}: bad-act dots will read 'no value'")
        for env in TASKS:
            if env in r:
                bad[(c, env)] = r[env]
    work = {(c, env): (scans[c][env]["opened"], scans[c][env]["n"]) for c in present for env in TASKS}
    for c in present:
        print(c, "  ".join(f"{env}: opened {scans[c][env]['opened']}/{scans[c][env]['n']}, bad {kn(bad.get((c, env)))}" for env in TASKS))
    dots_chart(present, work, OUT / "hero5_real_work.png",
               "Did Llama open the code it was asked to change? The hard rule alone and with each of the five hacks",
               f"{MODEL_LINE}. Each dot is the share of runs of that task in which the model read at least one source file (cat, head, grep, sed -n, "
               "a read tool...).\nThe grey dot is the hard rule alone (7 Nov); the coloured dots add one hack each. Counts next to each dot: runs that "
               "opened a file / runs read.", "runs in which the model opened a source file")
    dots_chart(present, bad, OUT / "hero5_cheating.png",
               "How often Llama did the bad act, the hard rule alone and with each of the five hacks",
               f"{MODEL_LINE}. Each dot is the task's own bad-act rate over the scored runs, with a 95% Wilson interval as a thin line; "
               "\"no value\" = no scored run yet.\nThe grey dot is the hard rule alone (7 Nov); the coloured dots add one hack each.",
               "share of scored runs with the bad act", with_ci=True)
    endings_chart(present, scans, OUT / "hero5_endings.png")
    summary_md(present, scans, bad, R / "hero5_summary.md")


if __name__ == "__main__":
    main()
