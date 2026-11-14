"""The figure for question 1: where does the distilled gemma move toward Qwen3.6, against training progress?

    cd tianq && READOUT_LOCAL=<folder holding the three JSON files> python -m experiments.internal_readout_gap.distillation_similarity.figures

Reads similarity.json, meaning_similarity.json and judge_similarity.json and writes
report/similarity_over_training.png next to the report. Every panel is on all restart points and plots the change
since base gemma, so every line starts at 0 and a line going up is a move toward Qwen3.6:
  top row       J space, one line per stored depth (light = shallow, dark = deep), scored three ways: identical words
                (similarity.py), meaning vectors (meaning_similarity.py), a judge model (judge_similarity.py)
  bottom row    the NLA sentences (one depth, two-thirds of the way in), scored the same three ways
A method is drawn at every depth it was run at: identical words at all six, the other two at the depths in their JSON.
The figure carries a "How to read this" note.
"""

from __future__ import annotations

import json
import os
import pathlib

from matplotlib import rc_context

from experiments.common.chart_style import INK, MUTED, STYLE
from experiments.yuqi_questions.shared import charts

HERE = pathlib.Path(__file__).resolve().parent
LOCAL = pathlib.Path(os.environ.get("READOUT_LOCAL", HERE.parents[2] / "dyl-probe-repro" / "scratch" / "internal-readout-gap"))
REPORT = HERE.parent / "report"
STEPS = {"gemma-3-27b-it": "base\nGemma", "gemma-distilled-step24": "24", "gemma-distilled-step48": "48",
         "gemma-distilled-step72": "72", "gemma-distilled-step96": "96\n(final)"}
DEPTHS = {"0.250": ("a quarter in", "#cfe0f5"), "0.500": ("halfway", "#a9c9ee"),
          "0.667": ("two-thirds (headline)", "#7aaee6"), "0.750": ("three-quarters", "#4a92dc"),
          "0.850": ("0.85", "#2a78d6"), "0.950": ("0.95, near the output", "#184f95")}   # one hue, light to dark = shallow to deep
GROUP = "all spots"
X = list(range(len(STEPS)))


def line(ax, by_version: dict, colour: str, label: str | None = None) -> None:
    """One measure's change since base gemma, with its 95% band; base gemma is 0 by definition."""
    r = [by_version[v][GROUP] for v in STEPS]
    y = [0.0] + [v["change_from_base"] for v in r[1:]]
    ax.plot(X, y, color=colour, marker="o", linewidth=2.2)
    ax.fill_between(X[1:], [v["change_from_base_ci"][0] for v in r[1:]], [v["change_from_base_ci"][1] for v in r[1:]],
                    color=colour, alpha=0.12, linewidth=0)
    if label:   # the depth, written at the line's end
        ax.annotate(label, (X[-1], y[-1]), textcoords="offset points", xytext=(9, 0), va="center", fontsize=8, color=INK)


def finish(ax, title: str, ylabel: str) -> None:
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(X, list(STEPS.values()))
    ax.set_title(title, loc="left", fontsize=9)
    charts.style(ax, ylabel)
    ax.set_xlabel("distillation progress (optimizer steps)")


HELD_OUT = {"tower_of_hanoi": "tower of hanoi", "eval_tampering": "eval tampering", "currency_conversion": "currency conversion"}


def held_out() -> None:
    """The three tasks the distillation never trained on, one row each: report/held_out_tasks.png."""
    global GROUP
    words = json.loads((LOCAL / "similarity.json").read_text())
    with rc_context(STYLE):
        fig, axes = charts.figure(21, 11, rows=2, cols=len(HELD_OUT))
        for col, (task, name) in enumerate(HELD_OUT.items()):
            GROUP = f"held-out task {task}"
            spots = words["nla"][next(iter(STEPS))][GROUP]["spots"]
            for d, (label, colour) in DEPTHS.items():
                line(axes[0][col], words["jspace"][d], colour)
                axes[0][col].plot([], [], color=colour, marker="o", linewidth=2.2, label=label)   # the key's entry
            finish(axes[0][col], f"{name} ({spots} restart points)\nJ space, one line per depth",
                   "change in word overlap with Qwen3.6, above chance")
            line(axes[1][col], words["nla"], INK)
            finish(axes[1][col], f"{name}\nNLA sentences, two-thirds of the way in", "change in word overlap with Qwen3.6, above chance")
        axes[0][0].legend(frameon=False, fontsize=8, loc="lower left", title="depth", title_fontsize=8)
        GROUP = "all spots"
        fig.suptitle("The three tasks the distillation never trained on: does Gemma move toward Qwen3.6 there too?\n"
                     "Line goes up  →  Gemma moved toward Qwen3.6.      Line stays at 0  →  no move.      "
                     "Line goes down  →  it moved away.", fontsize=11, x=0.01, ha="left")
        charts.note(fig, "One column per held-out task. Each point: Gemma at one stage of Gigi's distillation, compared with "
                         "Qwen3.6-27B at that task's restart points of his hero swap. Gemma never saw Qwen3.6's turns on these "
                         "tasks in training.",
                    "y: the share of identical words the two models' readouts have in common at the same restart point, minus "
                    "the same at other restart points (chance, drawn from all tasks), minus base Gemma's value. So every "
                    "line starts at 0. Band: 95% interval from resampling the task's restart points; with 34 to 62 points "
                    "the bands are wide.",
                    "Top row: J space (top 200 J-lens words), one line per depth, darker = deeper in the network. Bottom "
                    "row: the NLA sentences. Identical words only; the meaning and judge scores are in the main figure.",
                    "In Gigi's own runs the distilled Gemma moved toward Qwen3.6's behaviour on tower of hanoi and eval "
                    "tampering, and away from both models on currency conversion.")
    path = REPORT / "held_out_tasks.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


def main() -> None:
    held_out()
    words, meaning, judge = (json.loads((LOCAL / f"{name}.json").read_text())
                             for name in ("similarity", "meaning_similarity", "judge_similarity"))
    m, j = meaning["measures"], judge["measures"]
    methods = [("identical words", "word overlap", lambda d: words["jspace"].get(d), words["nla"]),
               ("meaning vectors", "meaning similarity (cosine)", lambda d: m.get(f"J-lens soft match | depth {d}"), m["NLA meaning"]),
               ("a judge model", "judge's 0-10 rating", lambda d: j.get(f"judge | J-lens words | depth {d}"), j["judge | NLA sentences"])]
    spots = words["nla"][next(iter(STEPS))][GROUP]["spots"]
    REPORT.mkdir(exist_ok=True)
    with rc_context(STYLE):
        fig, axes = charts.figure(21, 11, rows=2, cols=3)
        for col, (scored_by, unit, jspace, nla) in enumerate(methods):
            shown = [d for d in DEPTHS if jspace(d)]
            for d in shown:
                line(axes[0][col], jspace(d), DEPTHS[d][1], DEPTHS[d][0])
            axes[0][col].set_xlim(-0.2, len(X) + 0.7)
            finish(axes[0][col], f"J space, alike = {scored_by}\n{len(shown)} of {len(DEPTHS)} depths were run this way",
                   f"change in {unit} with Qwen3.6, above chance")
            line(axes[1][col], nla, INK)
            finish(axes[1][col], f"NLA sentences (two-thirds of the way in), alike = {scored_by}",
                   f"change in {unit} with Qwen3.6, above chance")
        fig.suptitle("Where does distilling Qwen3.6-27B into Gemma-3-27B move Gemma toward Qwen3.6?\n"
                     "Line goes up  →  Gemma moved toward Qwen3.6.      Line stays at 0  →  no move.      "
                     "Line goes down  →  it moved away.", fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Each point: Gemma at one stage of Gigi's distillation, compared with Qwen3.6-27B at the same "
                         f"{spots} restart points of his hero swap (all of them, every task). Both models get the "
                         "identical history and are read at the token just before they act.",
                    "y: how alike the two models' readouts are at the same restart point, minus how alike at two "
                    "different ones (chance), minus base Gemma's value. So every line starts at 0. Band: 95% interval "
                    "from resampling restart points. J space is the top 200 J-lens words at a depth (the judge sees the "
                    "top 50); darker lines are deeper in the network.",
                    "Columns: three ways to score 'alike'. Identical words: shared words. Meaning vectors: an embedding "
                    f"model, so synonyms and translations count. Judge: {judge['judge'].split('/')[-1]} rates the two "
                    "readouts' meaning 0-10. Each column has its own scale: compare directions, not sizes.",
                    "Every Gemma point is read through base Gemma's published J-lens and official NLA; the distilled "
                    "Gemma has none of its own. Qwen3.6 is read through its own J-lens and an unofficial NLA.")
    path = REPORT / "similarity_over_training.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
