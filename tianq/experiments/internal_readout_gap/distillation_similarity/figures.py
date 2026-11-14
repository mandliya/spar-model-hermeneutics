"""The figure for question 1: similarity to Qwen3.6 against training progress.

    cd tianq && READOUT_LOCAL=<folder holding the three JSON files> python -m experiments.internal_readout_gap.distillation_similarity.figures

Reads similarity.json, meaning_similarity.json and judge_similarity.json and writes report/similarity_over_training.png
next to the report. Columns, the readout: J-lens top words at 2/3 depth (the headline fixed beforehand), J-lens near
the output (0.95 depth, added after the numbers showed a move there), NLA sentences. Rows, how "alike" is scored:
identical words (similarity.py), meaning vectors (meaning_similarity.py), a judge model's 0-10 rating
(judge_similarity.py); the last two rows were added on 13 Nov to check the first. One line for the tasks the
distillation trained on, one for the three it held out. Every figure carries a "How to read this" note.
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
HEADLINE, LATE = f"{2 / 3:.3f}", f"{0.95:.3f}"
GROUPS = {"trained tasks": (INK, "tasks the distillation trained on"), "held-out tasks": ("#e8663a", "the three held-out tasks")}
STEPS = {"gemma-3-27b-it": "base\nGemma", "gemma-distilled-step24": "24 steps", "gemma-distilled-step48": "48 steps",
         "gemma-distilled-step72": "72 steps", "gemma-distilled-step96": "96 steps\n(final)"}


def panel(ax, by_version: dict, ylabel: str, legend: bool) -> None:
    x = list(range(len(STEPS)))
    for group, (colour, label) in GROUPS.items():
        r = [by_version[v][group] for v in STEPS]
        ax.plot(x, [v["above_chance"] for v in r], color=colour, marker="o", linewidth=2.2,
                label=f"{label} ({r[0]['spots']} spots)")
        ax.fill_between(x, [v["above_chance_ci"][0] for v in r], [v["above_chance_ci"][1] for v in r], color=colour,
                        alpha=0.12, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(x, list(STEPS.values()))
    charts.style(ax, ylabel)
    ax.set_xlabel("distillation progress (optimizer steps; 96 = one pass over Qwen3.6's turns)")
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="best")


def main() -> None:
    words, meaning, judge = (json.loads((LOCAL / f"{name}.json").read_text())
                             for name in ("similarity", "meaning_similarity", "judge_similarity"))
    m, j = meaning["measures"], judge["measures"]
    rows = [("identical words", "word overlap with Qwen3.6, above chance",
             (words["jspace"][HEADLINE], words["jspace"][LATE], words["nla"])),
            ("meaning vectors", "meaning similarity (cosine) with Qwen3.6, above chance",
             (m[f"J-lens soft match | depth {HEADLINE}"], m[f"J-lens soft match | depth {LATE}"], m["NLA meaning"])),
            ("a judge model", "judge's 0-10 similarity rating with Qwen3.6, above chance",
             (j[f"judge | J-lens words | depth {HEADLINE}"], j[f"judge | J-lens words | depth {LATE}"], j["judge | NLA sentences"]))]
    columns = ("J-lens top words, two-thirds of the way in (the headline depth)", "J-lens top words, near the output (0.95 of the way in)",
               "NLA sentences, two-thirds of the way in (5 per spot)")
    REPORT.mkdir(exist_ok=True)
    with rc_context(STYLE):
        fig, axes = charts.figure(19, 16.5, rows=3, cols=3)
        for row, (scored_by, ylabel, panels) in zip(axes, rows):
            for ax, column, by_version in zip(row, columns, panels):
                panel(ax, by_version, ylabel, legend=ax is axes[0][0])
                ax.set_title(f"{column}\nalike = {scored_by}", loc="left", fontsize=9)
        fig.suptitle("Does distilling Qwen3.6-27B into Gemma-3-27B move Gemma's readouts toward Qwen3.6's?\n"
                     "Line goes up  →  distillation moved Gemma toward Qwen3.6.      Line stays flat  →  no move.      "
                     "Line goes down  →  it moved away.\n"
                     "How high a line sits is the tasks, not the training: compare each line with its own first point (base Gemma).",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Each point: Gemma at one stage of Gigi's distillation, compared with Qwen3.6-27B at the same "
                         "takeover spots of his hero swap. Both models get the identical history and are read at the "
                         "token just before they act.",
                    "y: how alike the two models' readouts are at the same spot, minus how alike they are at two "
                    "different spots (chance). 0 = no more alike than chance. Higher = Gemma's readout is closer to "
                    "Qwen3.6's. Averaged over spots; band: 95% interval from resampling spots.",
                    "Rows: three ways to score 'alike'. Top: shared identical words (J-lens: overlap of the top-200 word "
                    "sets; NLA: cosine of word counts). Middle: a multilingual embedding model, so synonyms and "
                    "translations count (J-lens: each word matched to its closest word in the other list; NLA: cosine "
                    f"of the averaged sentence vectors). Bottom: {judge['judge'].split('/')[-1]} reads both readouts (J-lens: "
                    "top 50 words) without knowing their source and rates their meaning 0-10; its chance is one other spot.",
                    "The distilled Gemma has no lens or NLA of its own: every Gemma point is read through base "
                    "Gemma's published J-lens and official NLA. Qwen3.6 is read through its own published J-lens "
                    "and an unofficial NLA.")
    path = REPORT / "similarity_over_training.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
