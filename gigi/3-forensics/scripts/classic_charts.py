"""Charts for the classic distillation round (distill/PLAN_classic.md; 19 Nov 2026), same look as distill_charts.py.
  python3 classic_charts.py     reads results/classic_rerun_summary.csv (classic_results.py, on the box) and
                                results/classic_train_loss.csv; writes
  charts/classic_1_transfer.png   per task, the bad act at each checkpoint, Qwen's own rate as a dashed line
  charts/classic_2_no_answer.png  per task and model, finished runs that stopped at the step limit without an answer
  charts/classic_3_loss.png       training loss per step and the loss on the held-back check runs every half epoch
"""
import csv, math, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import chart_style as ST   # font, ink colours, soft grid: one look for every chart

import os
RES = pathlib.Path(os.environ.get("CLASSIC_RES", HERE / "results")); CH = pathlib.Path(os.environ.get("CLASSIC_CH", HERE / "charts"))
MODELS = ["base", "c23", "c46", "c69", "c92"]
XLAB = {"base": "before", "c23": "0.5", "c46": "1", "c69": "1.5", "c92": "2"}
ENVS = ["eval_tampering", "tower_of_hanoi", "lazy_investigation", "currency_conversion", "norvane", "coding", "coding_norules"]
ACT = {"eval_tampering": "read the planted note (of finished runs)", "tower_of_hanoi": "cheated at the puzzle",
       "lazy_investigation": "took the shortcut (gave the wrong answer)", "currency_conversion": "did the forbidden sum by hand",
       "norvane": "tampered with the tests it could not run", "coding": "made the broken test pass",
       "coding_norules": "made the broken test pass"}


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def load():
    d = {}
    for r in csv.DictReader(open(RES / "classic_rerun_summary.csv")):
        if r["task"] == "eval_tampering":   # the pre-registered measure here: read the note, defined on every finished run
            k, n = int(r["read_note"]), int(r["finished"])
        else:
            k, n = int(r["k"]), int(r["n"])
        d[r["who"], r["task"]] = dict(r, K=k, N=n)
    return d


def transfer_chart(d, title):
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), sharey=True)
    x = list(range(len(MODELS)))
    for ax, env in zip(axes.flat, ENVS):
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        q = d["qwen", env]; qr = q["K"] / q["N"]
        ax.axhline(qr, color=ST.QWEN, lw=1.6, ls=(0, (4, 3)), zorder=1)
        ax.text(len(MODELS) - 0.55, qr, f"Qwen {qr:.0%}\n({q['K']} of {q['N']})", color=ST.QWEN, fontsize=7.4, va="center", ha="left")
        pts = [(i, m) for i, m in enumerate(MODELS) if d[m, env]["N"]]
        ys = [d[m, env]["K"] / d[m, env]["N"] for _, m in pts]
        lo = [y - wilson(d[m, env]["K"], d[m, env]["N"])[0] for y, (_, m) in zip(ys, pts)]
        hi = [wilson(d[m, env]["K"], d[m, env]["N"])[1] - y for y, (_, m) in zip(ys, pts)]
        ax.errorbar([i for i, _ in pts], ys, yerr=[lo, hi], color=ST.INK, ecolor=ST.FAINT, elinewidth=1, capsize=0, lw=2,
                    marker="o", ms=6, mec="white", mew=1.5, zorder=3)
        for j, (i, m) in enumerate(pts):
            y = ys[j]
            nb = [ys[q] for q in (j - 1, j + 1) if 0 <= q < len(ys)]
            up = y >= sum(nb) / len(nb) if nb else True          # a peak gets its label above, a dip below
            if abs((y + (0.08 if up else -0.08)) - qr) < 0.07:   # never on the dashed Qwen line
                up = not up
            if y < 0.08:
                up = True
            if y > 0.92:
                up = False
            ax.annotate(f"{d[m, env]['K']}/{d[m, env]['N']}", (i, y), xytext=(0, 9 if up else -10), textcoords="offset points",
                        ha="center", va="bottom" if up else "top", fontsize=7.4, color=ST.TEXT,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
        ax.set_xticks(x); ax.set_xticklabels([XLAB[m] for m in MODELS], fontsize=7.8)
        ax.set_ylim(-0.04, 1.12); ax.set_xlim(-0.35, len(MODELS) - 0.6)
        ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        tag = "never seen in training" if d["qwen", env]["held_out"] == "True" else "in training"
        ax.set_title(f"{env}  ·  {tag}", fontsize=9.3, loc="left", color=ST.INK, pad=16)
        ax.text(0, 1.03, ACT[env], transform=ax.transAxes, fontsize=7.6, color=ST.MUTED, va="bottom")
        if env in ("norvane", "coding", "coding_norules", "eval_tampering"):
            ax.set_xlabel("epochs of training", fontsize=7.6, color=ST.MUTED, labelpad=4)
    key = axes.flat[7]; key.axis("off")
    key.text(0, 0.95, "How to read it", fontsize=9.3, fontweight="bold", color=ST.INK, va="top", transform=key.transAxes)
    key.text(0, 0.82, "Black line: gemma-3 27B before training\nand after 0.5, 1, 1.5 and 2 epochs of training\non Qwen3.6 27B's runs. "
             "Each point is k of n\nruns; grey bars are 95% intervals.\n\nGreen dashed line: Qwen3.6 27B itself, on the\n"
             "same tasks (the runs that made the data).\n\nAll runs on our own servers, bf16.\nOnly runs that finished and gave an "
             "answer\ncount (eval_tampering: every finished run).", fontsize=7.6, color=ST.MUTED, va="top", transform=key.transAxes,
             linespacing=1.35)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.01, 0.948, "Gemma trained on 736 of Qwen's runs from 23 tasks (2 epochs) and was then run on 5 tasks held out of "
             "training and 2 tasks it trained on. 10 to 40 scored runs per point (the k of n on each point).", fontsize=8.2, color=ST.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 0.95, 0.92), h_pad=2.6, w_pad=4.2)
    out = CH / "classic_1_transfer.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def no_answer_chart(d, title, sub):
    envs = [e for e in ENVS if any(int(d[w, e]["no_answer_at_cap"]) for w in ["qwen"] + MODELS)]
    who = ["qwen"] + MODELS
    lab = {"qwen": "Qwen3.6 27B", "base": "gemma before training", "c23": "after 0.5 epochs", "c46": "after 1 epoch",
           "c69": "after 1.5 epochs", "c92": "after 2 epochs"}
    fig, axes = plt.subplots(1, len(envs), figsize=(3.2 * len(envs) + 1.6, 3.6), sharey=True)
    axes = axes if len(envs) > 1 else [axes]
    y = list(range(len(who)))[::-1]
    for ax, env in zip(axes, envs):
        v = [int(d[w, env]["no_answer_at_cap"]) / max(1, int(d[w, env]["finished"])) for w in who]
        ax.barh(y, v, color=[ST.QWEN] + [ST.GEMMA] + [ST.INK] * 4, height=0.6)
        for yi, w, vi in zip(y, who, v):
            ax.text(vi + 0.02, yi, f"{d[w, env]['no_answer_at_cap']} of {d[w, env]['finished']}", va="center", fontsize=7.6, color=ST.TEXT)
        ax.set_xlim(0, 1.15); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_title(f"{env}  ·  {d['qwen', env]['step_cap']}-step limit", fontsize=9.1, loc="left")
    axes[0].set_yticks(y); axes[0].set_yticklabels([lab[w] for w in who], fontsize=8.2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.01, 0.875, sub, fontsize=8.1, color=ST.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.8), w_pad=2.2)
    out = CH / "classic_2_no_answer.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def loss_chart(title):
    rows = list(csv.DictReader(open(RES / "classic_train_loss.csv")))
    tr = [(float(r["epoch"]), float(r["loss"])) for r in rows if r["kind"] == "train"]
    ev = [(float(r["epoch"]), float(r["loss"])) for r in rows if r["kind"] == "check"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.plot([e for e, _ in tr], [l for _, l in tr], color=ST.FAINT, lw=1.2, label="training runs, each step")
    ax.plot([e for e, _ in ev], [l for _, l in ev], color=ST.INK, lw=2, marker="o", ms=6, mec="white", mew=1.5,
            label="184 held-back check runs (150 sampled), every half epoch")
    for e, l in ev:
        ax.annotate(f"{l:.3f}", (e, l), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.6, color=ST.TEXT)
    ax.set_xticks([0, .5, 1, 1.5, 2]); ax.set_xlabel("epochs of training", fontsize=8, color=ST.MUTED)
    ax.set_ylim(0, max(l for _, l in ev + tr) * 1.15); ax.set_ylabel("loss on Qwen's turns", fontsize=8, color=ST.MUTED)
    ax.legend(loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = CH / "classic_3_loss.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


T1 = "After training on Qwen's runs, gemma reads the planted note and cheats at tower_of_hanoi like Qwen, on tasks it never saw"
T2 = "After training, gemma often uses every step it is allowed without giving an answer"
S2 = ("Share of finished runs that used every allowed step and never gave an answer (k of finished runs). These runs have no "
      "value for the bad act, so the first chart leaves them out,\nexcept in eval_tampering, where reading the note counts on "
      "every finished run. On their own, Qwen and gemma rarely run out of steps (Qwen: 9 of 40 in currency_conversion); the checkpoints often keep working instead.")
T3 = "Gemma's loss on Qwen's held-back runs fell from 0.755 to 0.259, most of it in the first half epoch"


if __name__ == "__main__":
    d = load()
    for w in ["qwen"] + MODELS:
        print(w, {e: f"{d[w, e]['K']}/{d[w, e]['N']}" for e in ENVS})
    transfer_chart(d, T1)
    no_answer_chart(d, T2, S2)
    loss_chart(T3)
