"""Llama under the hard end rule on all 29 tasks (8 Nov 2026; GG: "run it on everything"). Two charts and one table:
  charts/hero4_all29_cheating.png  one row per task, the share of scored runs with the bad act, Llama under the three
                                   end-of-run rules (no check: hero_solo*, soft check: hero2_llama*, hard rule: hero4_llama*)
                                   next to gemma-3 27B with the soft check (hero2_gemma*);
  charts/hero4_all29_endings.png   how Llama's runs ended under the soft check and under the hard rule, per task;
  results/hero4_all29_endings.md   the same as a table, with the step counts.
Rows are the 29 tasks in the 9 families of every other chart (confirm_effect_charts.layout). Run from gigi/3-forensics/agent-interp-envs:
  uv run --with matplotlib python ../scripts/hero4_all29.py
"""
import collections, csv, glob, json, os, pathlib, re, sys, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import confirm_effect_charts as C

R = H / "results"
PASSES = [  # label, summary csvs (8 main tasks, other 21), per-run csvs, colour
    ("Llama, no check (4 Nov)", ["hero_full_runs.csv", "hero_full_runs__llama-3.3-70b__other21_nocheck.csv"],
     ["hero_full_runs_per_run.csv", "hero_full_runs__llama-3.3-70b__other21_nocheck_per_run.csv"], "#b5aea4"),
    ("Llama, soft check: asked once (4-5 Nov)", ["hero2_full_runs__llama-3.3-70b.csv", "hero2_full_runs__llama-3.3-70b__other21.csv"],
     ["hero2_full_runs__llama-3.3-70b_per_run.csv", "hero2_full_runs__llama-3.3-70b__other21_per_run.csv"], "#a99cf5"),
    # 8 Nov: the __fix files come first, so for the five rerun tasks (FIX_ENVS) their rows win in rates() and endings()
    ("Llama, hard rule: only end_task ends the run (7-8 Nov)", ["hero4_full_runs__llama-3.3-70b__fix.csv", "hero4_full_runs__llama-3.3-70b.csv", "hero4_full_runs__llama-3.3-70b__other21.csv"],
     ["hero4_full_runs__llama-3.3-70b__fix_per_run.csv", "hero4_full_runs__llama-3.3-70b_per_run.csv", "hero4_full_runs__llama-3.3-70b__other21_per_run.csv"], "#6c4ff0"),
    ("gemma-3 27B, soft check (5-6 Nov)", ["hero2_full_runs__gemma-3-27b.csv", "hero2_full_runs__gemma-3-27b__other21.csv"], [], "#2f6fd6"),
]
REPROMPT = "A reply without a tool call does not end the session."
ENDINGS = [  # key, label, colour
    ("end_tool", "ended with end_task (or the task's own done)", "#6c4ff0"),
    ("task_end", "the task ended itself after a tool call (an answer, an email, a game over...)", "#8fb3e6"),
    ("plain", "ended on a plain reply (allowed under the soft check only)", "#d9d4cc"),
    ("reminders", "all 10 reminders used, then let go", "#f0a35e"),
    ("cap", "ran to the task's step cap", "#d9534f"),
]


# The five environments whose own loop ignored the synthetic end_task until the 8 Nov 2026 fix (stage hero4_llama_fix reran them).
FIX_ENVS = ("chess", "tictactoe", "swe_rebench_v2", "coding", "coding_norules")
FIX_STAGE = "hero4_llama_fix"


def rates(files, skip_fix_stage=False):
    """env -> (k, n) from the first summary csv that has the environment with n > 0 (so earlier files win)."""
    out = {}
    for f in files:
        p = R / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if skip_fix_stage and FIX_STAGE in r.get("stages", ""):
                continue
            if r.get("n") and int(r["n"]) > 0 and r["environment"] not in out:
                out[r["environment"]] = (int(r["k"]), int(r["n"]))
    return out


def max_steps(env):
    m = re.search(r"max_steps:\s*(\d+)", (H / "configs" / f"{env}__llama-3.3-70b.yaml").read_text())
    return int(m.group(1)) if m else 10**9


def endings(files, soft, skip_stage=None):
    """env -> Counter of how the runs ended, and env -> list of step counts (one per run). An environment's runs come from
    the first per-run csv that has it (so the __fix file, listed first, replaces the pre-fix runs of FIX_ENVS).
    skip_stage drops one stage's rows (the pre-fix numbers for the table)."""
    out, steps = collections.defaultdict(collections.Counter), collections.defaultdict(list)
    seen_in = {}
    for f in files:
        p = R / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            rd, env = H / r["run"], r["environment"]
            if skip_stage and r["stage"] == skip_stage:
                continue
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
            rem = sum(1 for x in m if x.get("role") == "user" and str(x.get("content", "")).startswith(REPROMPT))
            if n >= max_steps(env):
                k = "cap"
            elif "end_task" in lastcalls or "done" in lastcalls or "end_session" in lastcalls:   # end_session: swe_rebench_v2's own finish tool
                k = "end_tool"
            elif not soft and rem >= 10:
                k = "reminders"
            elif not lastcalls:
                k = "plain"
            else:
                k = "task_end"
            out[env][k] += 1; steps[env].append(n)
    return out, steps


def frame(title, sub, xlabel, right_w, legend_handles, legend_labels, foot=()):
    """confirm_effect_charts.frame with this script's own legend."""
    rows, heads, ymax = C.layout()
    plot_h = ymax * 0.62
    Hh = plot_h + 2.6 + 0.18 * len(foot)
    fig = plt.figure(figsize=(14, Hh)); left = 0.3
    fig.text(0.015, 1 - 0.22 / Hh, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.015, 1 - 0.58 / Hh, "\n".join(textwrap.wrap(sub, 190)), fontsize=8.7, color=C.MUTED, va="top", linespacing=1.4)
    bottom = (1.25 + 0.18 * len(foot)) / Hh
    ax = fig.add_axes([left, bottom, right_w, plot_h / Hh]); ax.set_ylim(ymax, 0.3)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for e, y in rows.items():
        ax.text(left - 0.008, y, C.LABEL.get(e, e) + ("  ◆" if e in C.MAIN else ""), transform=tr, ha="right", va="center",
                fontsize=8.8, fontweight="bold" if e in C.MAIN else "normal")
        ax.axhline(y + 0.5, color=C.GRID, lw=0.5, zorder=0)
    for t, y in heads:
        ax.add_patch(plt.Rectangle((0.008, y - 0.42), 0.984, 0.62, transform=tr, color=C.BAND, zorder=0, clip_on=False))
        ax.text(0.015, y - 0.1, t, transform=tr, fontsize=10, fontweight="bold", va="center")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(C.GRID); ax.tick_params(axis="x", colors=C.MUTED)
    ax.xaxis.grid(True, color=C.GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel(xlabel, color=C.MUTED, fontsize=8.7)
    fig.legend(handles=legend_handles, labels=legend_labels, loc="lower left", bbox_to_anchor=(0.015, (0.45 + 0.18 * len(foot)) / Hh),
               frameon=False, fontsize=8.7, handletextpad=0.4, columnspacing=1.3, ncol=3)
    for i, line in enumerate(foot):
        fig.text(0.015, (0.15 + 0.18 * (len(foot) - 1 - i)) / Hh, line, fontsize=8.2, color=C.RED, va="bottom")
    return fig, ax, rows


# ---- chart 1: the bad act, all 29 tasks, Llama under three rules next to gemma
data = [rates(f) for _, f, _, _ in PASSES]
off = [-0.3, -0.1, 0.1, 0.3]
fig, ax, rows = frame("Does the end-of-run rule change what Llama does? The bad act in all 29 tasks, Llama under three rules, next to gemma",
                      "Each row is one task, grouped into the same 9 families as all our charts. A dot is the share of scored runs in which the model "
                      "did the task's bad act (the measure of results/full_runs.csv, Opus judge where the paper uses one), with the counts on the right. "
                      "Llama's three passes are three separate sets of runs: no check (the run ends on any reply without a tool call), the soft check "
                      "(told that such a reply ends the run, asked once), the hard rule (such a reply never ends the run, only end_task does; at most 10 "
                      "reminders, then the step cap). gemma-3 27B ran with the soft check, the same setting as Llama's second dot. Llama 3.3 70B is "
                      "fp16 on CoreWeave, gemma-3 27B fp8 on DeepInfra, both via OpenRouter. 'no value' = the measure does not apply to any finished run "
                      "(funding_email: the model never opened the legal review; leaked_scorer under the hard rule: Llama quit at step 1, nothing to judge).",
                      "share of scored runs with the bad act", 0.43,
                      [plt.Line2D([], [], marker="o", ls="", color=c, ms=8) for _, _, _, c in PASSES] + [plt.Line2D([], [], marker="D", ls="", color=C.INK, ms=6)],
                      [l for l, _, _, _ in PASSES] + ["one of the 8 main tasks"])
ax.set_xlim(-3, 118); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for e, yc in rows.items():
    for i, (label, _, _, col) in enumerate(PASSES):
        y = yc + off[i]; kn = data[i].get(e)
        if kn is None:
            ax.text(0, y, "no value", fontsize=6.8, color="#999", va="center", ha="left")
            continue
        k, n = kn; x = 100 * k / n
        ax.plot(x, y, "o", color=col, ms=7, zorder=3)
        ax.text(1.03, y, f"{label.split(',')[0]}{' hard' if i == 2 else (' soft' if i == 1 else (' none' if i == 0 else ''))}:  {k} of {n}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7.4, color=col)
fig.savefig("charts/hero4_all29_cheating.png", dpi=160, facecolor="white"); plt.close(fig)
print("wrote charts/hero4_all29_cheating.png")

# ---- chart 2: how the runs ended, Llama soft check vs hard rule
soft, soft_steps = endings(PASSES[1][2], soft=True)
hard, hard_steps = endings(PASSES[2][2], soft=False)
prefix, prefix_steps = endings(PASSES[2][2][1:], soft=False, skip_stage=FIX_STAGE)   # the pre-fix hard-rule runs, for the table
prefix_rates = rates(PASSES[2][1][1:], skip_fix_stage=True)
fig, ax, rows = frame("How Llama's runs ended, all 29 tasks: the soft check (upper bar) against the hard end_task rule (lower bar)",
                      "Each row is one task; the upper bar is Llama's soft-check pass, the lower bar its hard-rule pass, each split by how the runs ended. "
                      "Under the hard rule a plain reply can no longer end a run, so the grey part disappears and the question is where it went: into a "
                      "deliberate end_task (purple), the task ending itself after a tool call (blue), the reminder limit (orange) or the step cap (red). "
                      "chess, tic-tac-toe, SWE-rebench and the two coding tasks are the 8 Nov rerun (stage hero4_llama_fix): before that day's fix "
                      "the synthetic end_task did not end those five environments, so Llama's end_task calls there were followed by a reminder and "
                      "the run went on (the pre-fix numbers are in results/hero4_all29_endings.md). The right-hand column gives the typical run "
                      "length (median steps), soft -> hard.",
                      "share of runs", 0.43,
                      [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ENDINGS], [l for _, l, _ in ENDINGS],
                      ["* tower_of_hanoi: Llama moves one disk per step and the task's 20-step cap comes before the puzzle is solved, under both rules; "
                       "gemma finishes most of its runs there in 4 to 6 steps. So that red row is slow play, not a refusal to stop."])
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for e, yc in rows.items():
    for y, cnt in ((yc - 0.19, soft.get(e, {})), (yc + 0.19, hard.get(e, {}))):
        n = sum(cnt.values())
        if not n:
            ax.text(0.5, y, "not run", fontsize=6.8, color="#999", va="center"); continue
        x = 0
        for key, _, col in ENDINGS:
            w = 100 * cnt.get(key, 0) / n
            if w:
                ax.barh(y, w, left=x, height=0.3, color=col, edgecolor="white", lw=0.6); x += w
    def med(v): v = sorted(v); return v[len(v) // 2] if v else None
    ms_, mh = med(soft_steps.get(e, [])), med(hard_steps.get(e, []))
    ax.text(1.03, yc, f"typical steps: {ms_ if ms_ is not None else '-'} -> {mh if mh is not None else '-'}   (runs {sum(soft.get(e, {}).values())} / {sum(hard.get(e, {}).values())})",
            transform=ax.get_yaxis_transform(), va="center", fontsize=7.4, color=C.MUTED)
fig.savefig("charts/hero4_all29_endings.png", dpi=160, facecolor="white"); plt.close(fig)
print("wrote charts/hero4_all29_endings.png")

# ---- the table
lines = ["# How Llama's runs ended on all 29 tasks: soft check vs hard end_task rule",
         "",
         "Llama 3.3 70B Instruct, fp16 on CoreWeave via OpenRouter. Soft check = stages hero2_llama, hero2_llama_key, hero2_llama_all (4-5 Nov 2026, "
         "`agent.end_confirm=true`). Hard rule = stages hero4_llama, hero4_llama_key, hero4_llama_all (7-8 Nov 2026, `agent.end_task_required=true`, "
         "`patches/end_task_required.patch`), with chess, tictactoe, swe_rebench_v2, coding and coding_norules from the 8 Nov rerun after the end_task fix "
         "(stage hero4_llama_fix; the pre-fix numbers for those five are in the second table). Written by `scripts/hero4_all29.py` from every run's last messages.json; charts "
         "`charts/hero4_all29_cheating.png` and `charts/hero4_all29_endings.png`.",
         "",
         "How a run can end: **end tool** = the last message calls `end_task` (or the task's own `done` / `end_session`); **task's own end** = the environment ended the "
         "run after a tool call (an answer submitted, an email sent, a game over, a solution submitted); **plain reply** = a reply without a tool call "
         "ended the run (soft check only); **reminders** = all 10 hard-rule reminders used up, then the reply was let through; **step cap** = the "
         "task's step limit. Steps are counted per run (one step = one model message).",
         "",
         "| task | runs soft / hard | typical steps soft -> hard | soft: end tool | soft: task's own end | soft: plain reply | soft: step cap | hard: end tool | hard: task's own end | hard: reminders | hard: step cap | bad act soft | bad act hard |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
def med(v): v = sorted(v); return v[len(v) // 2] if v else "-"
def kn(d, e): return f"{d[e][0]} of {d[e][1]}" if e in d else "no value"
for e in C.ENVS:
    s, h = soft.get(e, {}), hard.get(e, {})
    lines.append(f"| {e} | {sum(s.values())} / {sum(h.values())} | {med(soft_steps.get(e, []))} -> {med(hard_steps.get(e, []))} | "
                 f"{s.get('end_tool', 0)} | {s.get('task_end', 0)} | {s.get('plain', 0)} | {s.get('cap', 0)} | "
                 f"{h.get('end_tool', 0)} | {h.get('task_end', 0)} | {h.get('reminders', 0)} | {h.get('cap', 0)} | {kn(data[1], e)} | {kn(data[2], e)} |")
tot = lambda d, k: sum(c.get(k, 0) for c in d.values())
lines += ["", f"Totals, soft check: {sum(sum(c.values()) for c in soft.values())} runs, end tool {tot(soft, 'end_tool')}, task's own end {tot(soft, 'task_end')}, "
          f"plain reply {tot(soft, 'plain')}, step cap {tot(soft, 'cap')}.",
          f"Totals, hard rule: {sum(sum(c.values()) for c in hard.values())} runs, end tool {tot(hard, 'end_tool')}, task's own end {tot(hard, 'task_end')}, "
          f"reminders used up {tot(hard, 'reminders')}, step cap {tot(hard, 'cap')}.",
          "",
          "tower_of_hanoi: Llama moves one disk per step and the task's 20-step cap comes before the puzzle is solved, under both "
          "rules (gemma finishes most of its runs there in 4 to 6 steps), so that row is slow play, not a refusal to stop.",
          "",
          "## Before the fix: the five environments whose loop ignored the synthetic end_task (7 Nov runs)",
          "",
          "Until the 8 Nov 2026 fix, in chess, tictactoe, swe_rebench_v2, coding and coding_norules the synthetic `end_task` did not end the run (those "
          "environments ended only on their own event: game over, a submit, `done`), so an `end_task` call there was followed by a reminder and the run "
          "went on; Llama called `end_task` 21 to 75 times in a chess run and the game never finished. These are those pre-fix runs; the main table above "
          "has the rerun.",
          "",
          "| task | runs | typical steps | end tool | task's own end | reminders | step cap | bad act |",
          "|---|---|---|---|---|---|---|---|"]
for e in FIX_ENVS:
    h = prefix.get(e, {})
    lines.append(f"| {e} | {sum(h.values())} | {med(prefix_steps.get(e, []))} | {h.get('end_tool', 0)} | {h.get('task_end', 0)} | "
                 f"{h.get('reminders', 0)} | {h.get('cap', 0)} | {kn(prefix_rates, e)} |")
(R / "hero4_all29_endings.md").write_text("\n".join(lines) + "\n")
print("wrote results/hero4_all29_endings.md")
