"""Question 3: are the same kinds of lie easy or hard at both read positions, and is MASK where "Did you lie?" wins?

    ./run.sh yuqi easy_kinds_at_both_reads.figures        # from tianq/dyl-probe-repro (CPU)

both_reads_<target>.png  one dot per kind of lie, Qwen3-235B's own conversations, same kind (5-fold): x = the
                         probe over the reply, y = the probe at "Did you lie?", both on one task (did the reply
                         lie) and the same rows (the arena's same-kind scores), 95% intervals on both axes,
                         the diagonal drawn, MASK kinds highlighted. Above the diagonal, "Did you lie?" reads
                         that kind better.
Prints each kind's two AUROCs and their rank correlation across kinds (do the same kinds come out easy?).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from scipy.stats import spearmanr

from experiments.common.chart_style import INK, MUTED, STYLE
from experiments.yuqi_questions.same_kind_vs_k_equals_one.figures import SHORT
from experiments.yuqi_questions.shared import charts, stored

HERE = Path(__file__).parent
ARENA = HERE.parent / "lie_detector_arena"
TARGET = "qwen3-235b"
MASK = "#e8663a"


def chart(target: str) -> Path:
    same = json.loads((ARENA / f"arena_{target}.json").read_text(encoding="utf-8"))["same_kind"]
    kinds = same["kinds"]
    x = np.array([same["per_kind"][k]["reply"]["auroc"] for k in kinds])
    y = np.array([same["per_kind"][k]["did_you_lie"]["auroc"] for k in kinds])
    rho = spearmanr(x, y).statistic
    print(f"  both reads | {stored.LABELS[target]} | rank correlation across {len(kinds)} kinds: {rho:+.2f}", flush=True)
    for k, a, b in zip(kinds, x, y):
        p = same["per_kind"][k]
        print(f"  both reads | {stored.LABELS[target]} | {k}: reply {a:.3f} [{p['reply']['ci'][0]:.3f}, {p['reply']['ci'][1]:.3f}];"
              f" did you lie {b:.3f} [{p['did_you_lie']['ci'][0]:.3f}, {p['did_you_lie']['ci'][1]:.3f}];"
              f" difference {b - a:+.3f}", flush=True)
    with rc_context(STYLE):
        fig, axes = charts.figure(8, 8.0)
        ax = axes[0][0]
        ax.plot([0.4, 1.02], [0.4, 1.02], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        for k, a, b in zip(kinds, x, y):
            p = same["per_kind"][k]
            colour = MASK if k.startswith("mask") else INK
            ax.errorbar(a, b, xerr=[[a - p["reply"]["ci"][0]], [p["reply"]["ci"][1] - a]],
                        yerr=[[b - p["did_you_lie"]["ci"][0]], [p["did_you_lie"]["ci"][1] - b]],
                        color=colour, marker="o", markersize=6, capsize=2, linewidth=1, alpha=0.9)
            ax.annotate(SHORT[k], (a, b), textcoords="offset points", xytext=(6, -10), fontsize=8, color=colour)
        charts.style(ax, 'AUROC at "Did you lie?" (same task)')
        ax.set_xlabel("AUROC over the reply")
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(0.4, 1.02)
        ax.set_aspect("equal")
        above = int((y > x).sum())
        ax.set_title(f"{stored.LABELS[target]}, same kind, one task: did the reply lie\n"
                     f'MASK kinds in orange; above the line, "Did you lie?" reads that kind better',
                     loc="left", fontsize=9)
        fig.suptitle(f"Rank correlation across kinds {rho:+.2f}; {above} of {len(kinds)} kinds sit above the line",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Each dot: one kind of lie in Qwen3-235B's own conversations.",
                    "x: AUROC of a probe over the reply. y: AUROC of a probe just before the model answers"
                    ' "Did you lie?".',
                    "Both are trained and tested inside that kind (5-fold, each row scored by a probe that never saw"
                    " it), on the same rows and the same label: did the reply lie.",
                    "Bars: 95% interval from resampling that kind's rows 2,000 times."
                    " Sandbagging (ASCII) is left out: its reply is one letter.")
    path = HERE / f"both_reads_{target}.png"
    fig.savefig(path, dpi=200)
    return path


def main() -> None:
    stored.push(chart(TARGET), HERE.name)
    print(f"wrote both_reads_{TARGET}.png", flush=True)


if __name__ == "__main__":
    main()
