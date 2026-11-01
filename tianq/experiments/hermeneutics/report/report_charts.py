"""The chart of the Oct 30 report, drawn from the Hub at a pinned revision (nothing recomputed).

    PYTHONPATH=../../..:../../../dyl-probe-repro python report_charts.py

transfer_by_reader.png   held-out AUROC against k for Qwen3-235B and each reader of its conversations, both reads
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path

from huggingface_hub import hf_hub_download
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments.common.chart_style import AXIS, GRID, INK, MUTED, SECONDARY, STYLE, SURFACE

REPO, REV = "model-hermeneutics/dyl-probe-repro", "1acaac87cf658b87f099160e116eb2dc0e22b6dd"
HERE = Path(__file__).parent
# Reader colours as in faithfulness_charts.py: Gemma-12B, Gemma-27B, Qwen3.5-27B; the target in ink.
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1f9e89"
KS = [1, 2, 3, 5, 8, 10]
# The words-only line (experiments/cross_category_transfer/compare_diversity_words.py), in the gaps chart's grey.
WORDS_GRAY = "#8a8a8a"
WORDS_RESULT = "qwen3-235b/gigi/results/diversity_words_l_88_lm_0_01_ar_lr_hs89.json"
# (label, colour, dashed, log, line of the "none" size_matched row; the "neutral" row is two lines below)
CURVES = [
    ("Qwen3-235B itself", INK, False, "run_20261030T130943Z_compare_diversity.log", 670),
    ("Qwen3.5-27B", TEAL, False, "run_20261030T133922Z_compare_diversity.log", 632),
    ("Gemma-3-27B", ORANGE, False, "run_20261030T110225Z_compare_diversity.log", 632),
    ("Gemma-3-12B", BLUE, False, "run_20261030T081632Z_compare_diversity.log", 632),
    ("Gemma-3-27B, layers 20 / 31 / 45", ORANGE, True, "run_20261030T141517Z_compare_diversity.log", 670),
    (None, ORANGE, True, "run_20261030T150841Z_compare_diversity.log", 648),
    (None, ORANGE, True, "run_20261030T152918Z_compare_diversity.log", 648),
]


def fetch(path: str) -> str:
    return hf_hub_download(REPO, path, repo_type="dataset", revision=REV, token=os.environ.get("HF_TOKEN"))


def row(lines: list[str], number: int, variant: str) -> list[float]:
    text = lines[number - 1]
    assert text.startswith(f"| {variant} | size_matched |"), text
    return [float(v) for v in re.findall(r"\| (\d\.\d\d) \(", text)]


def transfer_chart() -> Figure:
    words = json.loads(Path(fetch(WORDS_RESULT)).read_text())
    with rc_context(STYLE):
        fig = Figure(figsize=(9, 3.9), facecolor=SURFACE)
        axes = fig.subplots(1, 2, sharey=True)
        for ax, (variant, offset, title) in zip(axes, (("none", 0, "Read over the reply"),
                                                       ("neutral", 2, 'Read at "did you lie?"'))):
            for label, colour, dashed, log, number in CURVES:
                # Split on newlines only: tqdm's carriage returns would shift the line numbers the Hub shows.
                with open(fetch(f"logs/{log}"), newline="") as f:
                    lines = f.read().split("\n")
                ax.plot(KS[:5], row(lines, number + offset, variant)[:5], color=colour,
                        linewidth=1 if dashed else 2, linestyle=(0, (3, 2)) if dashed else "-",
                        marker=None if dashed else "o", markersize=4, alpha=0.7 if dashed else 1)
            by_k = words["summary"][variant]["size_matched"]
            ax.plot(KS[:5], [by_k[str(k)]["mean"] for k in KS[:5]], color=WORDS_GRAY, linewidth=2, marker="X", markersize=5)
            ax.axhline(0.5, color=AXIS, linewidth=1)
            ax.set_title(title, color=INK, loc="left")
            ax.set_xticks(KS[:5])
            ax.set_xlabel("kinds of lie trained on (k)")
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("AUROC on the unseen kinds")
        axes[0].set_ylim(0.45, 0.8)
        handles = [Line2D([], [], color=c, linewidth=1 if d else 2, linestyle=(0, (3, 2)) if d else "-",
                          marker=None if d else "o", markersize=4) for l, c, d, *_ in CURVES if l]
        handles.append(Line2D([], [], color=WORDS_GRAY, linewidth=2, marker="X", markersize=5))
        fig.legend(handles, [l for l, *_ in CURVES if l] + ["words only (no model)"], loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), fontsize=8, labelcolor=SECONDARY)
        fig.text(0.01, 0.01, textwrap.fill(
            "Probes trained on k of Qwen3-235B's 11 lie categories, tested on the rest (size-matched, mean over 20 "
            "draws). Each reader is prefilled with Qwen3-235B's own conversations. k = 10 left out (1 test category).",
            170), fontsize=7, color=MUTED)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.2, wspace=0.08)
    return fig


if __name__ == "__main__":
    transfer_chart().savefig(HERE / "transfer_by_reader.png", dpi=200)
    print("wrote", HERE / "transfer_by_reader.png")
