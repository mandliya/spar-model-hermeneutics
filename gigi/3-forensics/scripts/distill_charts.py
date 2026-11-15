"""Charts for the distillation re-runs (distill/PLAN_hero9.md, step 3; 14 Nov 2026, redone 15 Nov after a 20-agent check).
  python3 distill_charts.py      reads results/distill_v2_eval.csv, Qwen3.6's plain hero9 table, gemma's plain October
                                 table (results/full_runs.csv) and the run logs in results/raw/ (distill_v2_*,
                                 hero9_qwen36 + hero9_qwen36_all, gemma's October Modal runs); writes
  charts/distill_1_gap.png       per task, the bad-act rate at each checkpoint, with Qwen3.6 and gemma alone as lines
  charts/distill_2_health.png    per model, how often finished runs hit the step limit, repeat a call or misuse a tool
  results/distill_v2_health.csv  the numbers behind the second chart (+ how often each model opened the tampering note)
Gemma's reference is its plain October runs: hero8 (used until 15 Nov) had the end_task rule and the worked example.
Only finished runs count (distill_eval.run_status for ours, full_runs_table.status for the references).
"""
import collections, csv, glob, json, math, pathlib, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import chart_style as ST   # font, ink colours, soft grid: one look for every chart
import full_runs_table as T
import distill_eval as DE

RES = HERE / "results"; RAW = RES / "raw"; CH = HERE / "charts"
MODELS = ["base", "c24", "c48", "c72", "c96"]
XLAB = {"base": "before", "c24": "25%", "c48": "50%", "c72": "75%", "c96": "100%"}
ENVS = ["tower_of_hanoi", "eval_tampering", "currency_conversion", "coding", "coding_norules", "lazy_investigation"]
HELD_OUT = {"tower_of_hanoi", "eval_tampering", "currency_conversion"}
ACT = {"tower_of_hanoi": "cheated at the puzzle", "eval_tampering": "hid the tampering, among runs that read the note",
       "currency_conversion": "did the forbidden sum by hand", "coding": "made the broken test pass",
       "coding_norules": "made the broken test pass", "lazy_investigation": "gave the wrong answer"}
# hand-placed labels where a point sits between two dashed lines (dx, dy in points, ha, va); checked on the render
LABEL_AT = {("tower_of_hanoi", "c24"): (8, 0, "left", "center"), ("coding", "c24"): (8, -2, "left", "center"),
            ("coding", "c72"): (0, -10, "center", "top"), ("eval_tampering", "c72"): (8, -6, "left", "center")}
QWEN_FILES = ["hero9_full_runs__qwen3.6-27b__other21.csv", "hero9_full_runs__qwen3.6-27b.csv"]


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def refs(files, model=None):
    out = {}
    for f in files:
        for r in csv.DictReader(open(RES / f)):
            if r["environment"] in ENVS and r["n"] and r["rate"] and (model is None or r["model"] == model):
                out[r["environment"]] = (int(r["k"]), int(r["n"]))
    return out


def ref_runs():
    """Finished reference runs per (who, env): Qwen3.6's plain hero9 runs and gemma's plain October runs."""
    out = collections.defaultdict(list)
    for s in ["hero9_qwen36", "hero9_qwen36_all"]:
        for e in ENVS:
            out["qwen", e] += [pathlib.Path(f).parent for f in glob.glob(str(RAW / s / e / "**" / "rollout.log"), recursive=True)]
    for e, m, run in T.modal_runs():
        if m == "gemma-3-27b" and e in ENVS:
            out["gemma", e].append(run)
    return {k: [r for r in v if T.status(r) == "finished"] for k, v in out.items()}


def our_runs():
    out = collections.defaultdict(list)
    for m in MODELS:
        for e in ENVS:
            for f in glob.glob(str(RAW / f"distill_v2_{m}" / e / "**" / "rollout.log"), recursive=True):
                run = pathlib.Path(f).parent
                if DE.run_status(run) == "finished":
                    out[m, e].append(run)
    return out


def gap_chart(ours, theirs):
    d = {(r["model"], r["env"]): (int(r["k"]), int(r["n"])) for r in csv.DictReader(open(RES / "distill_v2_eval.csv"))}
    qwen = refs(QWEN_FILES)
    gem = refs(["full_runs.csv"], "gemma-3-27b")
    note = {m: (sum(T.saw_note(r) for r in ours[m, "eval_tampering"]), len(ours[m, "eval_tampering"])) for m in MODELS}
    note["qwen"] = (sum(T.saw_note(r) for r in theirs["qwen", "eval_tampering"]), len(theirs["qwen", "eval_tampering"]))
    note["gemma"] = (sum(T.saw_note(r) for r in theirs["gemma", "eval_tampering"]), len(theirs["gemma", "eval_tampering"]))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.9), sharey=True)
    x = range(len(MODELS))
    for ax, env in zip(axes.flat, ENVS):
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        lines = []   # (height, colour, label); a reference resting on fewer than 5 runs is written out instead
        (qk, qn), (gk, gn) = qwen[env], gem[env]
        if abs(qk / qn - gk / gn) < 0.05:
            lines.append((qk / qn, ST.MUTED, f"Qwen3.6 and gemma alone: both {qk / qn:.0%}"))
        else:
            lines.append((qk / qn, ST.QWEN, f"Qwen3.6 alone {qk / qn:.0%}"))
            if gn >= 5:
                lines.append((gk / gn, ST.GEMMA, f"gemma alone {gk / gn:.0%}"))
        for h, col, lab in lines:
            ax.axhline(h, color=col, lw=1.6, ls=(0, (4, 3)), zorder=1)
            ax.text(len(MODELS) - 0.55, h, lab, color=col, fontsize=7.6, va="center", ha="left")
        pts = [(i, m) for i, m in enumerate(MODELS) if d.get((m, env), (0, 0))[1]]
        xs = [i for i, _ in pts]; ys = [d[m, env][0] / d[m, env][1] for _, m in pts]
        lo = [y - wilson(*d[m, env])[0] for y, (_, m) in zip(ys, pts)]
        hi = [wilson(*d[m, env])[1] - y for y, (_, m) in zip(ys, pts)]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=ST.INK, ecolor=ST.FAINT, elinewidth=1, capsize=0, lw=2, marker="o",
                    ms=6, mec="white", mew=1.5, zorder=3)
        for j, (i, m) in enumerate(pts):
            k, n = d[m, env]; y = ys[j]
            nb = [ys[q] for q in (j - 1, j + 1) if 0 <= q < len(ys)]
            up = y < 0.15 or (y >= sum(nb) / len(nb) if nb else True)   # away from the line; never into the x labels
            if any(abs((y + (0.07 if up else -0.07)) - h) < 0.07 for h, _, _ in lines) and 0.15 <= y <= 0.9:
                up = not up                                           # ...and off the dashed reference lines
            left = up and 0 < j < len(ys) - 1 and ys[j + 1] > y + 0.1   # a line rising to the right: shift left
            dx, dy, ha, va = LABEL_AT.get((env, m), (-4 if left else 0, 9 if up else -10, "right" if left else "center",
                                                     "bottom" if up else "top"))
            ax.annotate(f"{k}/{n}", (i, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.4,
                        color=ST.TEXT, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
        if env == "eval_tampering":
            ax.text(0, 0.03, "gemma never\nread the note\n(0 of 20 runs)", fontsize=7.2, color=ST.MUTED, ha="center", va="bottom")
        ax.set_xticks(list(x)); ax.set_xticklabels([XLAB[m] for m in MODELS], fontsize=7.8)
        ax.set_ylim(-0.04, 1.12); ax.set_xlim(-0.35, len(MODELS) - 0.6)
        ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        tag = "never seen in training" if env in HELD_OUT else "in training"
        act = ACT[env]
        if env == "eval_tampering":   # who read the note at all: the change distillation made here
            act += ("\nread it at all: " + ", ".join(f"{note[m][0]}/{note[m][1]}" for m in MODELS)
                    + f"; Qwen {note['qwen'][0]}/{note['qwen'][1]}")
        ax.set_title(f"{env}  ·  {tag}", fontsize=9.5, loc="left", color=ST.INK, pad=27 if "\n" in act else 16)
        ax.text(0, 1.03, act, transform=ax.transAxes, fontsize=7.8, color=ST.MUTED, va="bottom")
    fig.suptitle("Trained on Qwen3.6's runs, gemma starts cheating at a puzzle it never saw and moves towards Qwen in two training tasks",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.01, 0.945, "Black line: share of gemma's runs with the bad act before fine-tuning (\"before\") and at 25%, 50%, 75% "
             "and 100% of training, all run on our own server; grey bars are 95% intervals.\nEach point counts only runs that "
             "finished and gave an answer (k of n). Dashed lines: Qwen3.6 and gemma each run on their own, plain, through "
             "OpenRouter. In coding the two\nmodels barely differ, and lazy_investigation has one fixed answer that is in the "
             "training data, so those two panels show learning, not transfer.",
             fontsize=8, color=ST.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 0.92, 0.895), h_pad=2.2, w_pad=4.5)
    out = CH / "distill_1_gap.png"; fig.savefig(out, dpi=200); print("wrote", out.relative_to(HERE))
    return note


STEP_LIMIT = re.compile(r"max steps reached|Reached max steps limit")   # the coding tasks print the second form
SHELL_SUBMIT = re.compile(r'Function: execute_command\nArguments: \{"command": "(?:[^"]*\|\s*)?\s*submit\b')
CALL = re.compile(r"Function: (\w+)\nArguments: (.*?)\n={20}", re.S)
FAKE_RESULT = re.compile(r"^\s*```tool_output", re.M)   # the model writing its own tool result


def fake_result(run):
    steps = sorted(run.glob("step-*/messages.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not steps:
        return False
    msgs = json.loads(steps[-1].read_text())
    msgs = msgs if isinstance(msgs, list) else msgs.get("messages", [])
    return any(m.get("role") == "assistant" and FAKE_RESULT.search(str(m.get("content") or "")) for m in msgs)


def health_row(runs):
    c = collections.Counter()
    for run in runs:
        t = (run / "rollout.log").read_text(errors="replace"); c["runs"] += 1
        c["hit the step limit"] += bool(STEP_LIMIT.search(t))
        calls = CALL.findall(t)
        c["repeated the same call 3 times in a row"] += any(calls[i] == calls[i + 1] == calls[i + 2] for i in range(len(calls) - 2))
        c["misused a tool"] += bool(SHELL_SUBMIT.search(t)) or fake_result(run)
    return c


def health_chart(ours, theirs, note):
    rows = [("Qwen3.6 alone", [r for e in ENVS for r in theirs["qwen", e]], ST.QWEN),
            ("gemma alone", [r for e in ENVS for r in theirs["gemma", e]], ST.GEMMA)]
    rows += [(("gemma before training" if m == "base" else f"checkpoint {XLAB[m]}"), [r for e in ENVS for r in ours[m, e]], ST.INK)
             for m in MODELS]
    data = [(label, col, health_row(runs)) for label, runs, col in rows]
    cols = ["hit the step limit", "repeated the same call 3 times in a row", "misused a tool"]
    keys = ["qwen", "gemma"] + MODELS
    with open(RES / "distill_v2_health.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["who", "finished runs"] + cols + ["eval_tampering: read the note", "eval_tampering: finished runs"])
        for (label, _, c), key in zip(data, keys):
            w.writerow([label, c["runs"]] + [c[k] for k in cols] + list(note[key]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    y = range(len(data))[::-1]
    for ax, k in zip(axes, cols):
        v = [c[k] / c["runs"] for _, _, c in data]
        ax.barh(list(y), v, color=[col for _, col, _ in data], height=0.62)
        for yi, (_, _, c), vi in zip(y, data, v):
            ax.text(vi + 0.015, yi, f"{c[k]}/{c['runs']}", va="center", fontsize=7.6, color=ST.TEXT)
        ax.set_xlim(0, 0.68); ax.set_xticks([0, .25, .5]); ax.set_xticklabels(["0%", "25%", "50%"])
        ax.set_title(k, fontsize=9.3, loc="left")
    axes[0].set_yticks(list(y)); axes[0].set_yticklabels([lab for lab, _, _ in data], fontsize=8.3)
    fig.suptitle("The trained checkpoints get stuck far more often than either model on its own",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.01, 0.885, "Finished runs in the six tasks (runs our server timed out on, or cut off by the time limit, are "
             "left out). \"Misused a tool\": typed submit into the shell, or wrote its own\nmade-up tool result. Qwen also "
             "hits the step limit in tower_of_hanoi (17 of 20), but the checkpoints hit it in the coding tasks too, where "
             "Qwen never does. No run had gibberish.",
             fontsize=8.2, color=ST.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.2)
    out = CH / "distill_2_health.png"; fig.savefig(out, dpi=200); print("wrote", out.relative_to(HERE))
    for label, _, c in data:
        print(f"{label:22} {c['runs']:3} " + "  ".join(f"{k}: {c[k]}" for k in cols))
    print("read the eval_tampering note:", note)


if __name__ == "__main__":
    ours, theirs = our_runs(), ref_runs()
    note = gap_chart(ours, theirs)
    health_chart(ours, theirs, note)
