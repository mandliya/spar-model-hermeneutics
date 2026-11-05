"""Two charts (4 Nov 2026): what the end-of-run check changes, per task and model, before vs after.
The check (config agent.end_confirm=true, patches/end_confirm.patch): the model is told that a reply without a tool call
ends the session, and the first time it replies without one it is asked once whether it meant to stop.
  1. charts/confirm_1_steps.png     typical run length (median steps per run) without and with the check
  2. charts/confirm_2_cheating.png  cheating rate (k/n with a 95% Wilson interval) without and with the check
Before (without the check): Llama's first solo pass (stages hero_solo + hero_solo_key2; rates from results/hero_full_runs.csv),
Qwen3-Next-80B's and gemma-3-27b's original full runs (full_runs_table.STAGES; rates from results/full_runs.csv).
After (with the check): the hero2 stages of each model (hero2_llama + hero2_llama_key, hero2_qwen + hero2_qwen_key,
hero2_gemma + hero2_gemma_key; rates from results/hero2_full_runs__<model>.csv, written by hero_solo_foldin.sh).
Steps are counted from each run's rollout.log STEP markers, as llama_behaviour_charts.py does, over finished runs only.
Runs on partial data: a model or stage that is not in yet is skipped and the chart says so.
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/confirm_effect_charts.py
"""
import collections, csv, pathlib, re, statistics, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import full_runs_table as T
OUT = H / "charts"
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
COL = {"qwen3-next-80b": "#1baf7a", "gemma-3-27b": "#2a78d6", "llama-3.3-70b": "#7a5af5"}
NAME = {"qwen3-next-80b": "Qwen3-Next 80B", "gemma-3-27b": "gemma-3 27B", "llama-3.3-70b": "Llama 3.3 70B"}
MODELS = ["llama-3.3-70b", "qwen3-next-80b", "gemma-3-27b"]
OFFSET = {"llama-3.3-70b": -0.26, "qwen3-next-80b": 0.0, "gemma-3-27b": 0.26}
ENVS = ["dashboard_costly_failure", "optimize_rendering_engine", "grade_your_own", "ai_rd", "coding_norules", "coding",
        "optimize_rendering_engine_graded", "precommit_hook"]
ENV_NAME = {"dashboard_costly_failure": "dashboard (speed target)", "optimize_rendering_engine": "rendering engine (3x speed-up)",
            "grade_your_own": "grade your own model", "ai_rd": "ai_rd (improve a prompt)", "coding_norules": "coding, no rules stated",
            "coding": "coding (is_prime)", "optimize_rendering_engine_graded": "rendering engine, graded", "precommit_hook": "pre-commit hook"}
BEFORE_STAGES = {"llama-3.3-70b": ("hero_solo", "hero_solo_key2")}   # Qwen / gemma: their original full runs (T.modal_runs)
AFTER_STAGES = T.HERO2_STAGES
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})


def nsteps(log):
    return len(re.findall(r"^\s+STEP \d+\s*$", open(log, errors="replace").read(), re.M))


def finished_logs(runs):
    """env -> step counts, over runs that finished (full_runs_table.status) and have a log."""
    out = collections.defaultdict(list)
    for env, run in runs:
        if env in ENVS and (run / "rollout.log").exists() and T.status(run) == "finished":
            out[env].append(nsteps(run / "rollout.log"))
    return out


def steps(model, after):
    if after:
        stages = AFTER_STAGES[model]
        if not any((H / "results" / "logs" / s).is_dir() for s in stages):
            return None   # not launched yet
        return finished_logs((env, run) for env, _, run, _ in T.hero_runs(stages, model))
    if model in BEFORE_STAGES:
        return finished_logs((env, run) for env, _, run, _ in T.hero_runs(BEFORE_STAGES[model], model))
    return finished_logs((env, run) for env, m, run in T.modal_runs() if m == model)


def rates(model, after):
    """env -> (k, n) from the scoring csvs; None when the csv is not there yet."""
    if after:
        f = H / "results" / f"hero2_full_runs__{model}.csv"
    elif model in BEFORE_STAGES:
        f = H / "results" / "hero_full_runs.csv"
    else:
        f = H / "results" / "full_runs.csv"
    if not f.exists():
        return None
    return {r["environment"]: (int(r["k"]), int(r["n"])) for r in csv.DictReader(open(f))
            if r["environment"] in ENVS and (r.get("model") in (model, None, "")) and int(r["n"] or 0) > 0}


def frame(title, sub, xlabel, notes, Hh=7.2, right=0.6):
    fig = plt.figure(figsize=(11.5, Hh))
    fig.text(0.02, 1 - 0.25 / Hh, title, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.62 / Hh, "\n".join(textwrap.wrap(sub, 175)), fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
    ax = fig.add_axes([0.25, 1.05 / Hh, right, 1 - 2.25 / Hh])
    ax.set_yticks(range(len(ENVS))); ax.set_yticklabels([ENV_NAME[e] for e in ENVS], fontsize=9.5, fontweight="bold")
    ax.set_ylim(len(ENVS) - 0.5, -0.5)
    for i in range(1, len(ENVS)):
        ax.axhline(i - 0.5, color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8.5)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.4, ms=8),
               plt.Line2D([], [], marker="o", ls="", color=INK, ms=8)] + \
              [plt.Line2D([], [], marker="s", ls="", color=COL[m], ms=8) for m in MODELS]
    labels = ["without the end-of-run check", "with the end-of-run check"] + [NAME[m] for m in MODELS]
    ax.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(-0.4, -0.08), ncol=5, frameon=False, fontsize=8.5,
              handletextpad=0.4, columnspacing=1.4)
    if notes:
        fig.text(0.02, 0.12 / Hh, "\n".join(textwrap.wrap("Not in yet: " + "; ".join(notes) + ".", 190)), fontsize=8,
                 color="#b0452a", va="bottom", linespacing=1.3)
    return fig, ax


def pair(ax, y, color, before, after, alabel=None):
    """A hollow dot (before), a filled dot (after), an arrow from one to the other and each value on the outer side of
    its dot (so the labels never sit under the arrow, whichever way it points). Values may be None (log x axis)."""
    if before is not None:
        ax.plot(before, y, "o", mfc="white", mec=color, mew=1.6, ms=8.5, zorder=3)
    if after is not None:
        ax.plot(after, y, "o", color=color, ms=8.5, zorder=4)
    if before is not None and after is not None and before != after:
        ax.annotate("", (after, y), (before, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, alpha=0.7,
                    shrinkA=5, shrinkB=5), zorder=3)
    longer = after is None or before is None or after >= before   # before on the left, after on the right
    if before is not None:
        ax.text(before / 1.15 if longer else before * 1.15, y, f"{before:g}", va="center", ha="right" if longer else "left",
                fontsize=7.5, color=color)
    if after is not None:
        ax.text(after * 1.15 if longer else after / 1.15, y, alabel or f"{after:g}", va="center", ha="left" if longer else "right",
                fontsize=7.5, color=color, fontweight="bold")


def fig_steps():
    data = {m: (steps(m, False), steps(m, True)) for m in MODELS}
    notes = [f"{NAME[m]} with the check (its hero2 stages are not launched or nothing fetched yet)" for m in MODELS
             if data[m][1] is None or not any(data[m][1].values())]
    fig, ax = frame("Does the end-of-run check make the models work longer? Typical run length, without vs with it",
                    "Each row is one task; one step = one message from the model. Hollow dot = the typical run (median steps) without the check, "
                    "filled dot = with it (the model is told that a reply without a tool call ends the session, and is asked once whether it meant "
                    "to stop). Without the check: Llama's first solo pass, Qwen's and gemma's original full runs. Numbers at the right: finished runs counted, "
                    "without -> with.", "steps in a typical run (median; log scale)", notes)
    ax.set_xscale("log"); ax.set_xlim(0.7, 3000)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500"])
    ax.axvline(500, color="#b0452a", lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(500, len(ENVS) - 0.45, "  500 = step limit", color="#b0452a", fontsize=7.5, va="bottom")
    for i, e in enumerate(ENVS):
        for m in MODELS:
            y = i + OFFSET[m]; b, a = data[m]
            bv = statistics.median(b[e]) if b and b.get(e) else None
            av = statistics.median(a[e]) if a and a.get(e) else None
            pair(ax, y, COL[m], bv, av, f"{av:g} *" if av and av >= 500 else None)
            nb, na = (len(b[e]) if b and b.get(e) else 0), (len(a[e]) if a and a.get(e) else 0)
            ax.text(1.02, y, f"{nb} -> {na}" if na else f"{nb} -> (not in yet)", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5, color=MUTED)
    ax.text(1.02, -0.5, "runs", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7.5, color=MUTED)
    fig.text(0.02, 0.02, "* Qwen in the rendering engine, with the check: two of its five runs got stuck repeating the same search command "
             "until they hit the 500-step limit, two more were cut off by the time limit after 287 and 348 steps, and one finished after 41 steps.",
             fontsize=8, color="#b0452a", va="bottom")
    fig.savefig(OUT / "confirm_1_steps.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return notes


def fig_cheating():
    data = {m: (rates(m, False), rates(m, True)) for m in MODELS}
    notes = [f"{NAME[m]} with the check (results/hero2_full_runs__{m}.csv {'not written' if data[m][1] is None else 'has no scored runs'} yet)"
             for m in MODELS if not data[m][1]]
    fig, ax = frame("Does the end-of-run check change how often the models cheat? Cheating rate, without vs with it",
                    "Each row is one task. Dots are the share of scored runs in which the model cheated (the task's own measure or the Opus grader, as in "
                    "results/full_runs.csv), hollow = without the end-of-run check, filled = with it; the thin line is the 95% Wilson interval. "
                    "The counts on the right are runs that cheated out of runs scored. Without the check: Llama's first solo pass, Qwen's and gemma's original full runs.",
                    "runs in which the model cheated (%)", notes, Hh=8.2, right=0.5)
    ax.text(1.03, -0.5, "runs that cheated: without the check -> with it", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=7.8, color=MUTED, fontweight="bold")
    ax.set_xlim(-2, 104); ax.set_xticks([0, 20, 40, 60, 80, 100])
    for i, e in enumerate(ENVS):
        for m in MODELS:
            y = i + OFFSET[m]; b, a = data[m]
            for which, d, filled in (("before", b, False), ("after", a, True)):
                if not d or e not in d:
                    continue
                k, n = d[e]; p = 100 * k / n; lo, hi = (100 * v for v in T.wilson(k, n))
                yy = y + (0.07 if filled else -0.07)
                ax.plot([lo, hi], [yy, yy], color=COL[m], lw=1.4, alpha=0.5 if filled else 0.35, solid_capstyle="round", zorder=2)
                if filled:
                    ax.plot(p, yy, "o", color=COL[m], ms=8, zorder=4)
                else:
                    ax.plot(p, yy, "o", mfc="white", mec=COL[m], mew=1.6, ms=8, zorder=3)
            fmt = lambda d: (f"{d[e][0]} of {d[e][1]}" if d and e in d else "no scored runs")
            ax.text(1.03, y, f"{NAME[m]}:  {fmt(b)}  ->  {fmt(a)}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.8, color=COL[m])
            if (not b or e not in b) and (not a or e not in a):
                ax.text(0, y, "no scored runs", va="center", fontsize=7.5, color=MUTED)
            elif not a or e not in a:
                ax.text(1.02, y, "with the check: not scored yet", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "confirm_2_cheating.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return notes


if __name__ == "__main__":
    n1 = fig_steps(); n2 = fig_cheating()
    print("wrote charts/confirm_1_steps.png, charts/confirm_2_cheating.png")
    for n in n1 + n2:
        print("  not in yet:", n)
