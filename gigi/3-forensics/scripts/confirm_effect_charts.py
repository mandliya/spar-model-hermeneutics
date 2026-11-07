"""Two charts (4 Nov 2026; all 29 environments from 5 Nov): what the end-of-run check changes, per task and model.
The check (config agent.end_confirm=true, patches/end_confirm.patch): the model is told that a reply without a tool call
ends the session, and the first time it replies without one it is asked once whether it meant to stop.
  1. charts/confirm_1_steps.png     typical run length (median steps per run) without and with the check
  2. charts/confirm_2_cheating.png  cheating rate (k/n with a 95% Wilson interval) without and with the check
Rows: all 29 environments, grouped by the 9 families A-I and named as in scripts/story_charts.py (FAMILIES, LABEL; read
from that file's source, so every chart uses the same groups and names). A diamond marks the 8 main tasks (the swap's).
Without the check: Llama's first solo pass (stages hero_solo + hero_solo_key2 + hero_solo_all; rates from
results/hero_full_runs.csv and results/hero_full_runs__llama-3.3-70b__other21_nocheck.csv), Qwen3-Next-80B's and
gemma-3-27b's original full runs (full_runs_table.modal_runs; rates from results/full_runs.csv).
With the check: Llama's hero2 stages (hero2_llama + hero2_llama_key + hero2_llama_all; rates from
results/hero2_full_runs__llama-3.3-70b.csv and results/hero2_full_runs__llama-3.3-70b__other21.csv); Qwen's and gemma's
hero2 stages plus hero2_qwen_all / hero2_gemma_all (6 Nov, the other 21; results/hero2_full_runs__<model>__other21.csv).
Steps are counted from each run's rollout.log STEP markers over finished runs only. A typical run at or over its task's
step limit (max_steps in configs/<env>__<model>.yaml) is starred and listed in the footnote.
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/confirm_effect_charts.py
"""
import ast, collections, csv, os, pathlib, re, statistics, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import full_runs_table as T
OUT = pathlib.Path(os.environ.get("CHART_OUT", H / "charts"))   # CHART_OUT: draft renders elsewhere
INK, MUTED, GRID, RED, BAND = "#17140f", "#6f6a60", "#e1e0d9", "#b0452a", "#f3f2ee"
COL = {"qwen3-next-80b": "#1baf7a", "gemma-3-27b": "#2a78d6", "llama-3.3-70b": "#7a5af5"}
NAME = {"qwen3-next-80b": "Qwen3-Next 80B", "gemma-3-27b": "gemma-3 27B", "llama-3.3-70b": "Llama 3.3 70B"}
MODELS = ["llama-3.3-70b", "qwen3-next-80b", "gemma-3-27b"]
OFFSET = {"llama-3.3-70b": -0.27, "qwen3-next-80b": 0.0, "gemma-3-27b": 0.27}
LLAMA = "llama-3.3-70b"

# the 9 families and the env names, from story_charts.py's source (importing it would run its charts)
_lit = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse((H / "scripts" / "story_charts.py").read_text()).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("FAMILIES", "LABEL")}
FAMILIES, LABEL = _lit["FAMILIES"], _lit["LABEL"]
ALL29 = {r["environment"] for r in csv.DictReader(open(H / "results" / "full_runs.csv"))}
MAIN = set(T.HERO_ENVS)
GROUPS = [(f"{letter}.  {name}", [e for e in envs if e in ALL29]) for letter, name, envs in FAMILIES]
GROUPS = [g for g in GROUPS if g[1]]
ENVS = [e for _, envs in GROUPS for e in envs]
assert set(ENVS) == ALL29, sorted(ALL29 - set(ENVS))

BEFORE_STAGES = {LLAMA: ("hero_solo", "hero_solo_key2", "hero_solo_all")}   # Qwen / gemma: their original full runs
AFTER_STAGES = dict(T.HERO2_STAGES); AFTER_STAGES[LLAMA] = tuple(AFTER_STAGES[LLAMA]) + ("hero2_llama_all",)
ALL_STAGE = {LLAMA: "hero2_llama_all", "qwen3-next-80b": "hero2_qwen_all", "gemma-3-27b": "hero2_gemma_all"}   # 6 Nov: Qwen/gemma other 21
for _m in ("qwen3-next-80b", "gemma-3-27b"):
    AFTER_STAGES[_m] = tuple(AFTER_STAGES[_m]) + (ALL_STAGE[_m],)
BEFORE_CSV = {LLAMA: ["hero_full_runs.csv", "hero_full_runs__llama-3.3-70b__other21_nocheck.csv"]}
AFTER_CSV = {LLAMA: ["hero2_full_runs__llama-3.3-70b.csv", "hero2_full_runs__llama-3.3-70b__other21.csv"],
             "qwen3-next-80b": ["hero2_full_runs__qwen3-next-80b.csv", "hero2_full_runs__qwen3-next-80b__other21.csv"],
             "gemma-3-27b": ["hero2_full_runs__gemma-3-27b.csv", "hero2_full_runs__gemma-3-27b__other21.csv"]}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})


def ran_with_check(model, env):
    """Every model ran the 8 main tasks with the check; the other 21 count once the model's all-environment stage
    (hero2_<x>_all) launched that environment."""
    return model == LLAMA or env in MAIN or env in T.hero_launched(ALL_STAGE[model], model)


def not_run_note():
    """A sentence naming the cells not run with the check, or nothing when every model ran every task with it."""
    miss = [(m, e) for m in MODELS for e in ENVS if not ran_with_check(m, e)]
    if not miss:
        return ""
    who = sorted({NAME[m] for m, _ in miss})
    return f"Not run with the check yet: {len(miss)} task rows for {' and '.join(who)} (marked on the right). "


def limit(env, model):
    f = H / "configs" / f"{env}__{model}.yaml"
    m = re.search(r"^\s*max_steps:\s*(\d+)", f.read_text(), re.M) if f.exists() else None
    return int(m.group(1)) if m else None


def nsteps(log):
    """STEP markers; tower_of_hanoi's log has none, so there each TOOL CALL / RESPONSE block (one per message) counts."""
    t = open(log, errors="replace").read()
    # [ \t] not \s: \s also matches newlines, and on a log with a huge blank stretch (Qwen's repetition loops) ^\s+ backtracks
    # quadratically (the 45-minute renders of 5-6 Nov). The header lines are indented with spaces only, so the count is the same.
    n = len(re.findall(r"^[ \t]+STEP \d+[ \t]*$", t, re.M))
    return n or len(re.findall(r"^[ \t]+(?:TOOL CALL|RESPONSE)[ \t]*$", t, re.M))


def finished_logs(runs):
    """env -> step counts, over runs that finished (full_runs_table.status) and have a log."""
    out = collections.defaultdict(list)
    for env, run in runs:
        if env in ENVS and (run / "rollout.log").exists() and T.status(run) == "finished":
            n = nsteps(run / "rollout.log")
            if n:
                out[env].append(n)
    return out


def steps(model, after):
    if after:
        return finished_logs((env, run) for env, _, run, _ in T.hero_runs(AFTER_STAGES[model], model))
    if model in BEFORE_STAGES:
        return finished_logs((env, run) for env, _, run, _ in T.hero_runs(BEFORE_STAGES[model], model))
    return finished_logs((env, run) for env, m, run in T.modal_runs() if m == model)


def rates(model, after):
    """env -> (k, n) from the scoring csvs (rows with n > 0 only; the first file that has the env wins)."""
    files = (AFTER_CSV if after else BEFORE_CSV).get(model, ["full_runs.csv"])
    out = {}
    for name in files:
        f = H / "results" / name
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            e = r["environment"]
            if e in ENVS and e not in out and r.get("model") in (model, None, "") and int(r["n"] or 0) > 0:
                out[e] = (int(r["k"]), int(r["n"]))
    return out


def layout():
    """y of each env row and of each family header (headers take a short slot above their rows)."""
    y, rows, heads = 0.0, {}, []
    for title, envs in GROUPS:
        y += 0.75; heads.append((title, y)); y += 0.15
        for e in envs:
            y += 1.0; rows[e] = y
    return rows, heads, y + 0.6


def frame(title, sub, xlabel, right_w, foot):
    rows, heads, ymax = layout()
    row_in = 0.62
    plot_h = ymax * row_in
    Hh = plot_h + 2.6 + 0.18 * len(foot)
    W = 14; left = 0.3
    fig = plt.figure(figsize=(W, Hh))
    fig.text(0.015, 1 - 0.22 / Hh, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.015, 1 - 0.58 / Hh, "\n".join(textwrap.wrap(sub, 190)), fontsize=8.7, color=MUTED, va="top", linespacing=1.4)
    bottom = (1.25 + 0.18 * len(foot)) / Hh
    ax = fig.add_axes([left, bottom, right_w, plot_h / Hh])
    ax.set_ylim(ymax, 0.3)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for e, y in rows.items():
        mark = "  ◆" if e in MAIN else ""
        ax.text(left - 0.008, y, LABEL.get(e, e) + mark, transform=tr, ha="right", va="center", fontsize=8.8,
                fontweight="bold" if e in MAIN else "normal")
        ax.axhline(y + 0.5, color=GRID, lw=0.5, zorder=0)
    for title, y in heads:
        ax.add_patch(plt.Rectangle((0.008, y - 0.42), 0.984, 0.62, transform=tr, color=BAND, zorder=0, clip_on=False))
        ax.text(0.015, y - 0.1, title, transform=tr, fontsize=10, fontweight="bold", va="center")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x", colors=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8.7)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.4, ms=8),
               plt.Line2D([], [], marker="o", ls="", color=INK, ms=8)] + \
              [plt.Line2D([], [], marker="s", ls="", color=COL[m], ms=8) for m in MODELS] + \
              [plt.Line2D([], [], marker="D", ls="", color=INK, ms=6)]
    labels = ["without the end-of-run check", "with the end-of-run check"] + [NAME[m] for m in MODELS] + \
             ["one of the 8 main tasks (used in the swap)"]
    fig.legend(handles=handles, labels=labels, loc="lower left", bbox_to_anchor=(0.015, (0.45 + 0.18 * len(foot)) / Hh), ncol=6,
               frameon=False, fontsize=8.7, handletextpad=0.4, columnspacing=1.3)
    for i, line in enumerate(foot):
        fig.text(0.015, (0.15 + 0.18 * (len(foot) - 1 - i)) / Hh, line, fontsize=8.2, color=RED, va="bottom")
    return fig, ax, rows


def pair(ax, y, color, before, after, blabel=None, alabel=None):
    """A hollow dot (before), a filled dot (after), an arrow from one to the other and each value on the outer side of
    its dot (so the labels never sit under the arrow, whichever way it points). Values may be None (log x axis)."""
    if before is not None:
        ax.plot(before, y, "o", mfc="white", mec=color, mew=1.6, ms=8, zorder=3)
    if after is not None:
        ax.plot(after, y, "o", color=color, ms=8, zorder=4)
    if before is not None and after is not None and before != after:
        ax.annotate("", (after, y), (before, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=0.7,
                    shrinkA=5, shrinkB=5), zorder=3)
    longer = after is None or before is None or after >= before   # before on the left, after on the right
    if before is not None:
        ax.text(before / 1.17 if longer else before * 1.17, y, blabel or f"{before:g}", va="center", ha="right" if longer else "left",
                fontsize=7.6, color=color)
    if after is not None:
        ax.text(after * 1.17 if longer else after / 1.17, y, alabel or f"{after:g}", va="center", ha="left" if longer else "right",
                fontsize=7.6, color=color, fontweight="bold")


def fig_steps():
    data = {m: (steps(m, False), steps(m, True)) for m in MODELS}
    meds = {}
    hit = []   # (model, env, which, value, limit)
    for m in MODELS:
        for which, d in (("without", data[m][0]), ("with", data[m][1])):
            for e in ENVS:
                if d.get(e):
                    v = statistics.median(d[e]); meds[(m, e, which)] = v
                    lim = limit(e, m)
                    if lim and v >= lim:
                        hit.append((m, e, which, v, lim))
    foot = []
    if hit:
        foot.append("* The typical run hit the task's step limit (its last allowed message), so the model was still going when the run was stopped:")
        for m, e, which, v, lim in hit:
            foot.append(f"     {NAME[m]}, {e}, {which} the check: limit {lim} steps")
        if any(m == "qwen3-next-80b" and e == "optimize_rendering_engine" and w == "with" for m, e, w, _, _ in hit):
            foot.append("     (Qwen in the rendering engine with the check: two of its five runs got stuck repeating the same search command, "
                        "two were cut off by the time limit after 287 and 348 steps, one finished after 41.)")
    fig, ax, rows = frame("Does the end-of-run check make the models work longer? Typical run length in all 29 tasks, without vs with it",
                          "Each row is one task, grouped into the same 9 families as all our charts. One step = one message from the model. "
                          "Hollow dot = the typical run (median steps) without the check, filled dot = with it (the model is told that a reply "
                          "without a tool call ends the session, and is asked once whether it meant to stop). Without the check: Llama's first "
                          "solo pass, Qwen's and gemma's original full runs. With the check: every model ran all 29 tasks again. " + not_run_note() +
                          "The two dots are two separate sets of runs, started fresh: the with-check runs are not the old runs continued, so a row "
                          "can get shorter by chance. The right-hand column gives the finished runs counted, without -> with.",
                          "steps in a typical run (median; log scale)", 0.43, foot)
    ax.set_xscale("log"); ax.set_xlim(0.7, 900)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500"])
    for e, yc in rows.items():
        for m in MODELS:
            y = yc + OFFSET[m]; b, a = data[m]
            bv, av = meds.get((m, e, "without")), meds.get((m, e, "with"))
            star = {w for mm, ee, w, _, _ in hit if mm == m and ee == e}
            pair(ax, y, COL[m], bv, av, f"{bv:g} *" if "without" in star else None, f"{av:g} *" if "with" in star else None)
            nb, na = len(b.get(e, [])), len(a.get(e, []))
            right = (f"{na}" if na else "no finished runs") if ran_with_check(m, e) else "not run with the check"
            left = f"{nb}" if nb else "no finished runs"
            ax.text(1.03, y, f"{NAME[m]}:  {left}  ->  {right}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.8,
                    color=COL[m])
    ax.text(1.03, 0.45, "finished runs: without the check -> with it", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=8, color=MUTED, fontweight="bold")
    fig.savefig(OUT / "confirm_1_steps.png", dpi=160, facecolor="white"); plt.close(fig)
    return hit


def why_empty(model, env, after):
    """Why a cell that ran has no scored runs (checked 5 Nov 2026 against the logs and the grading logs)."""
    if env == "funding_email":
        return "In funding_email, the measure only applies once the model opens the legal review, and these runs never opened it"
    if model == LLAMA and not after:
        return ("The judge skipped them as not judgeable: Llama stopped at its first step, before doing anything it could rate")
    if model == "qwen3-next-80b" and env == "optimize_lru_cache":
        return "In optimize_lru_cache, Qwen quit at step 1, so there was nothing to grade"
    return "no scored runs (see the loss audits in results/)"


def fig_cheating():
    data = {m: (rates(m, False), rates(m, True)) for m in MODELS}
    empty = collections.defaultdict(lambda: collections.defaultdict(list))   # reason -> "Llama 3.3 70B without the check" -> [envs]
    for m in MODELS:
        for after, d in ((False, data[m][0]), (True, data[m][1])):
            for e in ENVS:
                if e not in d and (not after or ran_with_check(m, e)):
                    empty[why_empty(m, e, after)][f"{NAME[m]} {'with' if after else 'without'} the check"].append(e)
    foot = ["Cells with no scored runs, and why:"] if empty else []
    for reason, who in empty.items():
        foot.append("   • " + reason + ".")
        for w, envs in who.items():
            foot.append(f"        {w}: {', '.join(envs)}" if not reason.startswith("In ")
                        else f"        {w}")
    fig, ax, rows = frame("Does the end-of-run check change how often the models cheat? Cheating rate in all 29 tasks, without vs with it",
                          "Each row is one task, grouped into the same 9 families as all our charts. Dots are the share of scored runs in which the "
                          "model did that task's bad act (the task's own measure or the Opus grader, as in results/full_runs.csv). Hollow = without "
                          "the end-of-run check, filled = with it; the thin line is the 95% interval. The right-hand column gives runs that cheated "
                          "out of runs scored. With the check: every model ran all 29 tasks again, started fresh, so the two dots are two separate "
                          "sets of runs, not the same runs continued. " + not_run_note(),
                          "runs in which the model cheated (%)", 0.38, foot)
    ax.set_xlim(-2, 104); ax.set_xticks([0, 20, 40, 60, 80, 100])
    for e, yc in rows.items():
        for m in MODELS:
            y = yc + OFFSET[m]; b, a = data[m]
            for d, filled in ((b, False), (a, True)):
                if e not in d:
                    continue
                k, n = d[e]; p = 100 * k / n; lo, hi = (100 * v for v in T.wilson(k, n))
                yy = y + (0.07 if filled else -0.07)
                ax.plot([lo, hi], [yy, yy], color=COL[m], lw=1.4, alpha=0.5 if filled else 0.35, solid_capstyle="round", zorder=2)
                if filled:
                    ax.plot(p, yy, "o", color=COL[m], ms=7.5, zorder=4)
                else:
                    ax.plot(p, yy, "o", mfc="white", mec=COL[m], mew=1.6, ms=7.5, zorder=3)
            left = f"{b[e][0]} of {b[e][1]}" if e in b else "no scored runs"
            right = (f"{a[e][0]} of {a[e][1]}" if e in a else "no scored runs") if ran_with_check(m, e) else "not run with the check"
            ax.text(1.03, y, f"{NAME[m]}:  {left}  ->  {right}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.8,
                    color=COL[m])
            if e not in b and e not in a:
                ax.text(0, y, "no scored runs", va="center", fontsize=7.5, color=MUTED)
    ax.text(1.03, 0.45, "runs that cheated: without the check -> with it", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=8, color=MUTED, fontweight="bold")
    fig.savefig(OUT / "confirm_2_cheating.png", dpi=160, facecolor="white"); plt.close(fig)
    return data


if __name__ == "__main__":
    hit = fig_steps(); data = fig_cheating()
    print("wrote charts/confirm_1_steps.png, charts/confirm_2_cheating.png")
    for m, e, w, v, lim in hit:
        print(f"  step limit: {m} {e} {w} the check: median {v:g} >= {lim}")
    for m in MODELS:
        b, a = data[m]
        print(f"  {m}: scored without the check in {len(b)} tasks, with it in {len(a)} tasks")
