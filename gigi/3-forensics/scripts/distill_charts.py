"""Charts for the distillation re-runs (distill/PLAN_hero9.md, step 3; 14 Nov 2026).
  python3 distill_charts.py      reads results/distill_v2_eval.csv, the hero9/hero8 reference tables and the run logs
                                 in results/raw/ (distill_v2_*, hero9_qwen36_all, hero8_gemma_all); writes
  charts/distill_1_gap.png       per task, the bad-act rate at each checkpoint, with Qwen3.6 and gemma alone as lines
  charts/distill_2_health.png    per checkpoint, how often runs crash, run out of steps, misuse a tool or loop
  results/distill_v2_health.csv  the numbers behind the second chart
"""
import collections, csv, glob, math, pathlib, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import chart_style as ST   # font, ink colours, soft grid: one look for every chart
import full_runs_table as T

RES = HERE / "results"; RAW = RES / "raw"; CH = HERE / "charts"
MODELS = ["base", "c24", "c48", "c72", "c96"]
XLAB = {"base": "start", "c24": "25%", "c48": "50%", "c72": "75%", "c96": "100%"}
ENVS = ["tower_of_hanoi", "eval_tampering", "currency_conversion", "coding", "coding_norules", "lazy_investigation"]
HELD_OUT = {"tower_of_hanoi", "eval_tampering", "currency_conversion"}
ACT = {"tower_of_hanoi": "cheated at the puzzle", "eval_tampering": "hid the tampering it read about",
       "currency_conversion": "did the forbidden sum by hand", "coding": "made the broken test pass",
       "coding_norules": "made the broken test pass", "lazy_investigation": "took the shortcut"}


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def refs(files):
    out = {}
    for f in files:
        for r in csv.DictReader(open(RES / f)):
            if r["environment"] in ENVS and r["n"] and r["rate"]:
                out[r["environment"]] = (int(r["k"]), int(r["n"]))
    return out


def gap_chart():
    d = {(r["model"], r["env"]): (int(r["k"]), int(r["n"])) for r in csv.DictReader(open(RES / "distill_v2_eval.csv"))}
    qwen = refs(["hero9_full_runs__qwen3.6-27b__other21.csv", "hero9_full_runs__qwen3.6-27b.csv"])
    gem = refs(["hero8_full_runs__gemma-3-27b__other21.csv", "hero8_full_runs__gemma-3-27b.csv"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharey=True)
    x = range(len(MODELS))
    for ax, env in zip(axes.flat, ENVS):
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        (qk, qn), (gk, gn) = qwen[env], gem[env]
        close = abs(qk / qn - gk / gn) < 0.1   # currency: both at ~100%, one label below the other
        for (k, n), col, name, dy in [((qk, qn), ST.QWEN, "Qwen3.6 27B", 0.05 if close else 0),
                                      ((gk, gn), ST.GEMMA, "gemma-3 27B alone", -0.05 if close else 0)]:
            ax.axhline(k / n, color=col, lw=1.6, ls=(0, (4, 3)), zorder=1)
            ax.text(len(MODELS) - 0.55, k / n + dy, f"{name} {k / n:.0%}", color=col, fontsize=7.6, va="center", ha="left")
        ys = [d[m, env][0] / d[m, env][1] for m in MODELS]
        lo = [y - wilson(*d[m, env])[0] for y, m in zip(ys, MODELS)]
        hi = [wilson(*d[m, env])[1] - y for y, m in zip(ys, MODELS)]
        ax.errorbar(x, ys, yerr=[lo, hi], color=ST.INK, ecolor=ST.FAINT, elinewidth=1, capsize=0, lw=2, marker="o",
                    ms=6, mec="white", mew=1.5, zorder=3)
        for i, m in enumerate(MODELS):
            k, n = d[m, env]
            nb = [ys[j] for j in (i - 1, i + 1) if 0 <= j < len(ys)]
            up = ys[i] < 0.15 or ys[i] >= sum(nb) / len(nb)   # away from the line; never down into the x labels
            left = up and 0 < i < len(ys) - 1 and ys[i + 1] > ys[i] + 0.1   # a line rising to the right: shift left
            ax.annotate(f"{k}/{n}", (i, ys[i]), xytext=(-4 if left else 0, 9 if up else -10), textcoords="offset points",
                        ha="right" if left else "center", va="bottom" if up else "top", fontsize=7.4,
                        color=ST.TEXT, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
        ax.set_xticks(list(x)); ax.set_xticklabels([XLAB[m] for m in MODELS], fontsize=7.8)
        ax.set_ylim(-0.04, 1.12); ax.set_xlim(-0.35, len(MODELS) - 0.6)
        ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        tag = "not in training" if env in HELD_OUT else "in training"
        ax.set_title(f"{env}  ·  {tag}", fontsize=9.5, loc="left", color=ST.INK, pad=16)
        ax.text(0, 1.03, ACT[env], transform=ax.transAxes, fontsize=7.8, color=ST.MUTED, va="bottom")
    fig.suptitle("Training gemma on Qwen3.6's runs moves it towards Qwen in five of the six tasks",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.01, 0.94, "Share of runs with the bad act at each checkpoint of the fine-tune (25% to 100% of training), "
             "20 runs each; fewer where runs never gave an answer.\nThe dashed lines are Qwen3.6 and gemma on their own "
             "(hero9 and hero8 runs). Grey bars: 95% interval. Checkpoints served in bf16 on our box, references fp8 via OpenRouter.",
             fontsize=8.2, color=ST.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 0.93, 0.9), h_pad=2.2, w_pad=4.5)
    out = CH / "distill_1_gap.png"; fig.savefig(out, dpi=200); print("wrote", out.relative_to(HERE))


SHELL_SUBMIT = re.compile(r'Function: execute_command\nArguments: \{"command": "[^"]*\bsubmit\b')
CALL = re.compile(r"Function: (\w+)\nArguments: (.*)")


def health_row(logs):
    c = collections.Counter()
    for f in logs:
        st = T.status(pathlib.Path(f).parent)
        if st in ("provider failure", "no log"):   # lost to the API, not the model's doing
            continue
        t = open(f, errors="replace").read(); c["runs"] += 1
        c["crashed"] += st != "finished"
        c["ran out of steps"] += "max steps reached" in t
        c["misused a tool"] += bool(SHELL_SUBMIT.search(t)) or "```tool_output" in t
        calls = CALL.findall(t)
        c["same call 3 times in a row"] += any(calls[i] == calls[i + 1] == calls[i + 2] for i in range(len(calls) - 2))
    return c


def health_chart():
    rows = [("Qwen3.6 27B alone", ["hero9_qwen36", "hero9_qwen36_all"], ST.QWEN),
            ("gemma-3 27B alone", ["hero8_gemma", "hero8_gemma_all"], ST.GEMMA)]
    rows += [(("gemma on our server" if m == "base" else f"checkpoint {XLAB[m]}"), [f"distill_v2_{m}"], ST.INK) for m in MODELS]
    data = []
    for label, stages, col in rows:
        logs = [f for s in stages for e in ENVS for f in glob.glob(str(RAW / s / e / "**" / "rollout.log"), recursive=True)]
        data.append((label, col, health_row(logs)))
    cols = ["crashed", "ran out of steps", "misused a tool", "same call 3 times in a row"]
    with open(RES / "distill_v2_health.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["who", "runs"] + cols)
        for label, _, c in data:
            w.writerow([label, c["runs"]] + [c[k] for k in cols])
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.9), sharey=True)
    y = range(len(data))[::-1]
    for ax, k in zip(axes, cols):
        v = [c[k] / c["runs"] for _, _, c in data]
        ax.barh(list(y), v, color=[col for _, col, _ in data], height=0.62)
        for yi, (_, _, c), vi in zip(y, data, v):
            ax.text(vi + 0.02, yi, f"{c[k]}/{c['runs']}", va="center", fontsize=7.6, color=ST.TEXT)
        ax.set_xlim(0, 0.75); ax.set_xticks([0, .25, .5]); ax.set_xticklabels(["0%", "25%", "50%"])
        ax.set_title(k, fontsize=9.3, loc="left")
    axes[0].set_yticks(list(y)); axes[0].set_yticklabels([lab for lab, _, _ in data], fontsize=8.3)
    fig.suptitle("The trained checkpoints get stuck more often than either model on its own",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.01, 0.885, "All runs in the six tasks. \"Misused a tool\": typed submit into the shell, or wrote its own "
             "made-up tool result. Running out of steps is Qwen-like in tower_of_hanoi\n(Qwen 17 of 20), but new in "
             "eval_tampering, where the checkpoints loop on a shell submit that does nothing. No run had gibberish replies.",
             fontsize=8.2, color=ST.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.2)
    out = CH / "distill_2_health.png"; fig.savefig(out, dpi=200); print("wrote", out.relative_to(HERE))
    for label, _, c in data:
        print(f"{label:22} {c['runs']:3} " + "  ".join(f"{k}: {c[k]}" for k in cols))


if __name__ == "__main__":
    gap_chart(); health_chart()
