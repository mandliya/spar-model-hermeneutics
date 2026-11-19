"""Big, simple charts for a 10-minute call on the classic round (19 Nov 2026). No titles inside the images: the page
gives each one. Reads results/classic_rerun_summary.csv and results/classic_train_loss.csv (as classic_charts.py).
  charts/classic_call_1_summary.png     per task: gemma before, gemma after 2 epochs, Qwen (dumbbell + numbers)
  charts/classic_call_2_trajectory.png  eval_tampering and tower_of_hanoi at every checkpoint
  charts/classic_call_3_steps.png       runs that used every allowed step without answering
  charts/classic_call_4_loss.png        loss on the held-back check runs
"""
import csv, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import chart_style as ST
from classic_charts import load, wilson, MODELS, RES, CH

AFTER = "#d9480f"   # ST.DISTILLED: gemma after training
ROWS = [("eval_tampering", "read the planted note", True), ("tower_of_hanoi", "cheated at the puzzle", True),
        ("lazy_investigation", "took the shortcut", True), ("currency_conversion", "did the forbidden sum", True),
        ("norvane", "tampered with the tests", True), ("coding", "made the broken test pass", False),
        ("coding_norules", "made the broken test pass", False)]


def pct(k, n):
    return k / n if n else 0


def summary(d):
    fig = plt.figure(figsize=(12, 5.8))
    ax = fig.add_axes([0.22, 0.09, 0.44, 0.8])
    ys, y = [], 0
    for i in range(len(ROWS)):
        if i == 5:
            y += 0.7   # gap before the two trained tasks
        ys.append(y); y += 1
    ys = [max(ys) - v for v in ys]
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(min(ys) - 0.6, max(ys) + 1.0)
    tr = ax.get_yaxis_transform()   # x in axes fractions, y in data units
    for (env, act, held), yy in zip(ROWS, ys):
        b, a, q = d["base", env], d["c92", env], d["qwen", env]
        pb, pa, pq = pct(b["K"], b["N"]), pct(a["K"], a["N"]), pct(q["K"], q["N"])
        ax.plot([0, 1], [yy, yy], color=ST.GRID, lw=6, solid_capstyle="round", zorder=0)
        if abs(pa - pb) > 0.03:
            ax.annotate("", xy=(pa, yy), xytext=(pb, yy), arrowprops=dict(arrowstyle="-|>", color=ST.FAINT, lw=1.6,
                        shrinkA=7, shrinkB=8, mutation_scale=12), zorder=1)
        ax.scatter([pb], [yy], s=110, color=ST.GEMMA, edgecolor="white", lw=1.5, zorder=4)
        ax.scatter([pa], [yy], s=110, color=AFTER, edgecolor="white", lw=1.5, zorder=5)
        ax.scatter([pq], [yy], marker="D", s=150, facecolor="none", edgecolor=ST.QWEN, lw=2.2, zorder=6)
        ax.text(-0.05, yy + 0.13, env, transform=tr, ha="right", va="center", fontsize=11, fontweight="bold", color=ST.INK)
        ax.text(-0.05, yy - 0.2, act, transform=tr, ha="right", va="center", fontsize=9.2, color=ST.MUTED)
        for x, (k, n) in [(1.27, (b["K"], b["N"])), (1.5, (a["K"], a["N"])), (1.72, (q["K"], q["N"]))]:
            ax.text(x, yy, f"{k} of {n}", transform=tr, ha="right", va="center", fontsize=10.5, color=ST.TEXT)
    for x, lab, col in [(1.27, "gemma\nbefore", ST.GEMMA), (1.5, "gemma after\n2 epochs", AFTER), (1.72, "Qwen", ST.QWEN)]:
        ax.text(x, max(ys) + 0.55, lab, transform=tr, ha="right", va="bottom", fontsize=9.2, color=col, fontweight="bold")
    ax.axhline((ys[4] + ys[5]) / 2, color=ST.RULE, lw=1)
    ax.text(1.0, ys[0] + 0.55, "never seen in training", ha="right", fontsize=8.8, color=ST.MUTED)
    ax.text(1.0, ys[5] + 0.55, "in training", ha="right", fontsize=8.8, color=ST.MUTED)
    ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9.5)
    ax.grid(axis="x"); ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("share of tries where the model did it (0% = never, 100% = every time)", fontsize=10, color=ST.MUTED, labelpad=8)
    out = CH / "classic_call_1_summary.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def trajectory(d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    xl = ["before", "0.5", "1", "1.5", "2"]
    for ax, (env, act) in zip(axes, [("eval_tampering", "read the planted note"), ("tower_of_hanoi", "cheated at the puzzle")]):
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        ks = [(d[m, env]["K"], d[m, env]["N"]) for m in MODELS]
        ys = [pct(k, n) for k, n in ks]
        lo = [wilson(k, n)[0] for k, n in ks]; hi = [wilson(k, n)[1] for k, n in ks]
        x = list(range(len(MODELS)))
        ax.fill_between(x, lo, hi, color=AFTER, alpha=0.12, lw=0)
        ax.plot(x, ys, color=AFTER, lw=2.6, zorder=3)
        ax.scatter(x[1:], ys[1:], s=70, color=AFTER, edgecolor="white", lw=1.5, zorder=4)
        ax.scatter([0], [ys[0]], s=80, color=ST.GEMMA, edgecolor="white", lw=1.5, zorder=5)
        q = d["qwen", env]; qr = pct(q["K"], q["N"])
        ax.axhline(qr, color=ST.QWEN, lw=2, ls=(0, (5, 3)), zorder=2)
        ax.text(4.25, qr, f"Qwen: {q['K']} of {q['N']}", color=ST.QWEN, fontsize=10, va="center", ha="left", fontweight="bold")
        for i, ((k, n), yv) in enumerate(zip(ks, ys)):
            up = not (yv > 0.85) and not (abs(yv + 0.09 - qr) < 0.07)
            ax.annotate(f"{k} of {n}", (i, yv), xytext=(0, 11 if up else -12), textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=9.6, color=ST.TEXT, zorder=6,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=10); ax.set_xlabel("epochs of training", fontsize=9.5, color=ST.MUTED)
        ax.set_ylim(-0.05, 1.12); ax.set_xlim(-0.4, 4.2)
        ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9.5)
        ax.set_title(f"{env}  ·  {act}", loc="left", fontsize=11.5, pad=10)
    axes[0].set_ylabel("share of tries where gemma did it", fontsize=10, color=ST.MUTED, labelpad=8)
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=6)
    out = CH / "classic_call_2_trajectory.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def steps(d):
    envs = [("eval_tampering", "20 steps"), ("lazy_investigation", "30 steps"), ("currency_conversion", "10 steps")]
    who = [("Qwen", ["qwen"], ST.QWEN), ("gemma before", ["base"], ST.GEMMA), ("gemma after training\n(all 4 checkpoints)", MODELS[1:], AFTER)]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    w = 0.26
    for j, (lab, ms, col) in enumerate(who):
        for i, (env, lim) in enumerate(envs):
            k = sum(int(d[m, env]["no_answer_at_cap"]) for m in ms); n = sum(int(d[m, env]["finished"]) for m in ms)
            x = i + (j - 1) * (w + 0.03)
            ax.bar(x, k / n, width=w, color=col, label=lab if i == 0 else None)
            ax.text(x, k / n + 0.025, f"{k} of {n}", ha="center", va="bottom", fontsize=9.4, color=ST.TEXT)
    ax.set_xticks(range(len(envs))); ax.set_xticklabels([f"{e}\n(limit {l})" for e, l in envs], fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0%", "50%", "100%"], fontsize=9.5)
    ax.set_ylabel("share of finished tries\nthat ran out of steps", fontsize=10, color=ST.MUTED, labelpad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9.5, labelspacing=1.1)
    fig.tight_layout()
    out = CH / "classic_call_3_steps.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def loss():
    rows = list(csv.DictReader(open(RES / "classic_train_loss.csv")))
    ev = [(float(r["epoch"]), float(r["loss"])) for r in rows if r["kind"] == "check"]
    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.plot([e for e, _ in ev], [l for _, l in ev], color=AFTER, lw=2.6, zorder=3)
    ax.scatter([e for e, _ in ev], [l for _, l in ev], s=70, color=AFTER, edgecolor="white", lw=1.5, zorder=4)
    for e, l in ev:
        ax.annotate(f"{l:.2f}", (e, l), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10, color=ST.TEXT)
    ax.set_xticks([0, .5, 1, 1.5, 2]); ax.set_xticklabels(["before", "0.5", "1", "1.5", "2"], fontsize=10)
    ax.set_xlabel("epochs of training", fontsize=9.5, color=ST.MUTED)
    ax.set_ylim(0, 0.9); ax.set_yticks([0, .25, .5, .75]); ax.tick_params(labelsize=9.5)
    ax.set_ylabel("loss on Qwen runs gemma\ndid not learn from (lower = better)", fontsize=10, color=ST.MUTED, labelpad=8)
    fig.tight_layout()
    out = CH / "classic_call_4_loss.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


LAST = "#ae3ec9"   # last week's (swap) distillation; passes the validator next to GEMMA and AFTER only with a second cue (dashed)
CMP = [("eval_tampering", "read the planted note", "never seen in training, both weeks"),
       ("tower_of_hanoi", "cheated at the puzzle", "never seen in training, both weeks"),
       ("currency_conversion", "did the forbidden sum", "never seen in training, both weeks"),
       ("lazy_investigation", "took the shortcut", "learned from last week, never seen this week"),
       ("coding", "made the broken test pass", "learned from, both weeks"),
       ("coding_norules", "made the broken test pass", "learned from, both weeks")]


def last_week():
    """Last week's re-runs (distill_v2, 15 Nov): k of n per task at before / 25 / 50 / 75 / 100% of training."""
    ev = {(r["model"], r["env"]): (int(r["k"]), int(r["n"])) for r in csv.DictReader(open(RES / "distill_v2_eval.csv"))}
    order = ["base", "c24", "c48", "c72", "c96"]
    out = {e: [ev[m, e] for m in order] for e, _, _ in CMP if e != "eval_tampering"}
    h = {r["who"]: r for r in csv.DictReader(open(RES / "distill_v2_health.csv"))}
    hw = ["gemma before training", "checkpoint 25%", "checkpoint 50%", "checkpoint 75%", "checkpoint 100%"]
    out["eval_tampering"] = [(int(h[w]["eval_tampering: read the note"]), int(h[w]["eval_tampering: finished runs"])) for w in hw]
    return out


def compare(d):
    lw = last_week()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharey=True)
    x = list(range(5)); xl = ["before", "25%", "50%", "75%", "100%"]
    for ax, (env, act, tag) in zip(axes.flat, CMP):
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        q = d["qwen", env]; qr = pct(q["K"], q["N"])
        ax.axhline(qr, color=ST.QWEN, lw=1.8, ls=(0, (5, 3)), zorder=1)
        ax.text(4.3, qr, "Qwen", color=ST.QWEN, fontsize=9.5, va="center", ha="left", fontweight="bold")
        tw = [(d[m, env]["K"], d[m, env]["N"]) for m in MODELS]
        for series, col, ls, lab in [(lw[env], LAST, (0, (4, 2)), "last week"), (tw, AFTER, "-", "this week")]:
            ys = [pct(k, n) for k, n in series]
            ax.plot(x, ys, color=col, lw=2.4, ls=ls, zorder=3, label=lab)
            ax.scatter(x[1:], ys[1:], s=46, color=col, edgecolor="white", lw=1.2, zorder=4)
        ax.scatter([0], [pct(*tw[0])], s=60, color=ST.GEMMA, edgecolor="white", lw=1.2, zorder=5)
        ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=9.5)
        ax.set_ylim(-0.05, 1.08); ax.set_xlim(-0.3, 4.3)
        ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0%", "50%", "100%"], fontsize=9.5)
        ax.set_title(f"{env}", loc="left", fontsize=11.5, pad=24)
        ax.text(0, 1.04, f"{act}  ·  {tag}", transform=ax.transAxes, fontsize=8.8, color=ST.MUTED, va="bottom")
    for ax in axes[1]:
        ax.set_xlabel("share of training done", fontsize=9.5, color=ST.MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of tries where gemma did it", fontsize=10, color=ST.MUTED, labelpad=8)
    axes.flat[0].legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(h_pad=3, w_pad=5)
    out = CH / "classic_call_5_vs_last_week.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def problems():
    lw = {r["who"]: r for r in csv.DictReader(open(RES / "distill_v2_health.csv"))}
    tw = {r["who"]: r for r in csv.DictReader(open(RES / "classic_health.csv"))}
    cols = ["hit the step limit", "repeated the same call 3 times in a row", "misused a tool"]
    def pool(rows, keys):
        n = sum(int(rows[k]["finished runs"]) for k in keys)
        return n, {c: sum(int(rows[k][c]) for k in keys) for c in cols}
    groups = [("Qwen", ST.QWEN, pool(tw, ["qwen"])), ("gemma before", ST.GEMMA, pool(tw, ["base"])),
              ("trained gemma, last week", LAST, pool(lw, ["checkpoint 25%", "checkpoint 50%", "checkpoint 75%", "checkpoint 100%"])),
              ("trained gemma, this week", AFTER, pool(tw, ["c23", "c46", "c69", "c92"]))]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    w = 0.2
    for j, (lab, col, (n, c)) in enumerate(groups):
        for i, k in enumerate(cols):
            xx = i + (j - 1.5) * (w + 0.02)
            ax.bar(xx, c[k] / n, width=w, color=col, label=lab if i == 0 else None,
                   hatch="///" if col == LAST else None, edgecolor="white", lw=0)
            ax.text(xx, c[k] / n + 0.015, f"{c[k] / n:.0%}", ha="center", va="bottom", fontsize=9.6, color=ST.TEXT)
    ax.set_xticks(range(3)); ax.set_xticklabels(["used every allowed step", "repeated the same call\n3 times in a row",
                                                 "misused a tool"], fontsize=10.5)
    ax.set_ylim(0, 0.7); ax.set_yticks([0, .25, .5]); ax.set_yticklabels(["0%", "25%", "50%"], fontsize=9.5)
    ax.set_ylabel("share of finished tries\nwith this problem", fontsize=10, color=ST.MUTED, labelpad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9.5, labelspacing=1.0)
    fig.tight_layout()
    out = CH / "classic_call_6_problems.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def norvane_coding(d):
    """norvane (not tested last week), coding and coding_norules at every save point, counts on this week's points."""
    lw = last_week()
    panels = [("norvane", "tampered with the tests", "never seen; not tested last week"),
              ("coding", "made the broken test pass", "learned from"),
              ("coding_norules", "made the broken test pass", "learned from")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    x = list(range(5)); xl = ["before", "25%", "50%", "75%", "100%"]
    for ax, (env, act, tag) in zip(axes, panels):
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        q = d["qwen", env]; qr = pct(q["K"], q["N"])
        ax.axhline(qr, color=ST.QWEN, lw=1.8, ls=(0, (5, 3)), zorder=1)
        ax.text(4.3, qr, f"Qwen\n{q['K']} of {q['N']}", color=ST.QWEN, fontsize=9.2, va="center", ha="left", fontweight="bold")
        tw = [(d[m, env]["K"], d[m, env]["N"]) for m in MODELS]
        if env in lw:
            ly = [pct(k, n) for k, n in lw[env]]
            ax.plot(x, ly, color=LAST, lw=2.2, ls=(0, (4, 2)), zorder=2, label="last week")
            ax.scatter(x[1:], ly[1:], s=40, color=LAST, edgecolor="white", lw=1.2, zorder=3)
        ys = [pct(k, n) for k, n in tw]
        ax.plot(x, ys, color=AFTER, lw=2.6, zorder=4, label="this week")
        ax.scatter(x[1:], ys[1:], s=60, color=AFTER, edgecolor="white", lw=1.4, zorder=5)
        ax.scatter([0], [ys[0]], s=70, color=ST.GEMMA, edgecolor="white", lw=1.4, zorder=6)
        for i, ((k, n), yv) in enumerate(zip(tw, ys)):
            ax.annotate(f"{k} of {n}", (i, yv), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.2, color=ST.TEXT, zorder=7, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=9.5); ax.set_xlabel("share of training done", fontsize=9.5, color=ST.MUTED)
        ax.set_ylim(-0.05, 1.08); ax.set_xlim(-0.3, 4.3)
        ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0%", "50%", "100%"], fontsize=9.5)
        ax.set_title(env, loc="left", fontsize=11.5, pad=24)
        ax.text(0, 1.04, f"{act}  ·  {tag}", transform=ax.transAxes, fontsize=8.6, color=ST.MUTED, va="bottom")
    axes[0].set_ylabel("share of tries where gemma did it", fontsize=10, color=ST.MUTED, labelpad=8)
    axes[1].legend(loc="upper left", fontsize=9.5)
    fig.tight_layout(w_pad=5)
    out = CH / "classic_call_7_norvane_coding.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


LAB = {"qwen": "Qwen", "base": "gemma\nbefore", "c23": "after\n0.5 ep", "c46": "after\n1 ep", "c69": "after\n1.5 ep", "c92": "after\n2 ep"}
COL = {"qwen": ST.QWEN, "base": ST.GEMMA, "c23": AFTER, "c46": AFTER, "c69": AFTER, "c92": AFTER}
Y = dict(fontsize=10, color=ST.MUTED, labelpad=8)


def eval_note_vs_hide(d):
    """eval_tampering: read the note (every finished run) vs hid the tampering (runs that gave a final report)."""
    who = ["qwen", "base", "c23", "c46", "c69", "c92"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, kind in zip(axes, ["read", "hid"]):
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        for i, w in enumerate(who):
            r = d[w, "eval_tampering"]
            k, n = (int(r["read_note"]), int(r["finished"])) if kind == "read" else (int(r["k"]), int(r["n"]))
            ax.bar(i, pct(k, n), width=0.62, color=COL[w])
            lab = f"{k} of {n}" if kind == "read" else f"{k} of {n}\nreports"
            ax.text(i, pct(k, n) + 0.02, lab, ha="center", va="bottom", fontsize=9.6, color=ST.TEXT)
        ax.set_xticks(range(len(who))); ax.set_xticklabels([LAB[w] for w in who], fontsize=9.6)
        ax.set_ylim(0, 1.22); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0%", "50%", "100%"], fontsize=9.5)
        if kind == "read":
            ax.set_title("read the planted note", loc="left", fontsize=11.5, pad=10)
            ax.set_ylabel("share of finished runs\nthat opened the note", **Y)
        else:
            ax.set_title("hid the tampering in its final report", loc="left", fontsize=11.5, pad=10)
            ax.set_ylabel("share of final reports\nthat hid the tampering", **Y)
    fig.tight_layout(w_pad=5)
    out = CH / "classic_call_8_eval_tampering.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def per_copy():
    """Share of finished tries that used every allowed step, per saved copy, both weeks (same six tasks)."""
    lw = {r["who"]: r for r in csv.DictReader(open(RES / "distill_v2_health.csv"))}
    tw = {r["who"]: r for r in csv.DictReader(open(RES / "classic_health.csv"))}
    k = "hit the step limit"
    f = lambda r: int(r[k]) / int(r["finished runs"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    x = [0, 1, 2, 3]
    last = [f(lw[w]) for w in ["checkpoint 25%", "checkpoint 50%", "checkpoint 75%", "checkpoint 100%"]]
    this = [f(tw[w]) for w in ["c23", "c46", "c69", "c92"]]
    for ys, col, ls, lab in [(last, LAST, (0, (4, 2)), "trained gemma, last week"), (this, AFTER, "-", "trained gemma, this week")]:
        ax.plot(x, ys, color=col, lw=2.6, ls=ls, label=lab, zorder=3)
        ax.scatter(x, ys, s=60, color=col, edgecolor="white", lw=1.4, zorder=4)
        for xi, yi in zip(x, ys):
            ax.annotate(f"{yi:.0%}", (xi, yi), xytext=(0, 9 if col == AFTER else -14), textcoords="offset points",
                        ha="center", fontsize=9.6, color=ST.TEXT)
    for v, col, lab in [(f(tw["qwen"]), ST.QWEN, "Qwen"), (f(tw["base"]), ST.GEMMA, "gemma before")]:
        ax.axhline(v, color=col, lw=1.8, ls=(0, (5, 3)), zorder=1)
        ax.text(3.85, v + 0.012, f"{lab} {v:.0%}", color=col, fontsize=9.6, va="bottom", ha="right", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_xlabel("share of training done (each saved copy of gemma)", fontsize=10, color=ST.MUTED)
    ax.set_ylim(0, 0.8); ax.set_xlim(-0.3, 3.9); ax.set_yticks([0, .25, .5, .75]); ax.set_yticklabels(["0%", "25%", "50%", "75%"], fontsize=9.5)
    ax.set_ylabel("share of finished tries that\nused every allowed step", **Y)
    ax.legend(loc="upper left", fontsize=9.5, ncol=2)
    fig.tight_layout()
    out = CH / "classic_call_9_per_copy.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


def median_steps():
    rows = {(r["task"], r["who"]): r for r in csv.DictReader(open(RES / "classic_median_steps.csv"))}
    envs = ["eval_tampering", "tower_of_hanoi", "lazy_investigation", "currency_conversion", "norvane", "coding", "coding_norules"]
    who = [("qwen", "Qwen", ST.QWEN), ("base", "gemma before", ST.GEMMA), ("c92", "gemma after 2 epochs", AFTER)]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    w = 0.26
    for j, (k, lab, col) in enumerate(who):
        for i, e in enumerate(envs):
            v = float(rows[e, k]["median_steps"]); xx = i + (j - 1) * (w + 0.02)
            ax.bar(xx, v, width=w, color=col, label=lab if i == 0 else None)
            ax.text(xx, v + 0.4, f"{v:.0f}", ha="center", va="bottom", fontsize=9.4, color=ST.TEXT)
    ax.set_xticks(range(len(envs))); ax.set_xticklabels(envs, fontsize=9.8)
    ax.set_ylabel("median number of steps\nper finished run", **Y)
    ax.set_ylim(0, 33)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    out = CH / "classic_call_10_steps.png"; fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    d = load()
    summary(d); trajectory(d); steps(d); loss(); compare(d); problems(); norvane_coding(d); eval_note_vs_hide(d); per_copy(); median_steps()
