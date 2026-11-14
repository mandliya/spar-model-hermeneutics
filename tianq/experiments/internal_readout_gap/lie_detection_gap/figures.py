"""The figure for question 2: the faithfulness gap per pair and method, from gap.py's results.

    cd tianq && python -m experiments.internal_readout_gap.lie_detection_gap.figures

Reads scratch/internal-readout-gap/{gap,meaning_gap,judge_gap}_<target>__<substitute>.json for every pair in PAIRS and
writes report/faithfulness_gap.png next to the report: one panel per split ("same type" and "unseen type" of lie), for
the "lie vs truthful" label read at before_answer (both fixed as the headline before any result). Each row is one pair
and method: the target's AUROC, the substitute's through the target's classifier, and the substitute's with its own
classifier; the words-only baseline is a grey tick on each pair's rows. The judge is not trained, so its rows are the
same in both panels and have no "own classifier".
"""

from __future__ import annotations

import json
import pathlib

from matplotlib import rc_context

from dyl_repro.core.store import SCRATCH
from experiments.common.chart_style import INK, MUTED, STYLE
from experiments.yuqi_questions.shared import charts

LOCAL = pathlib.Path(SCRATCH) / "internal-readout-gap"
REPORT = pathlib.Path(__file__).resolve().parents[1] / "report"
PAIRS = [("gemma-3-27b-it", "gemma-3-12b-it"), ("gemma-3-27b-it", "qwen3.6-27b"), ("qwen3-32b", "gemma-3-27b-it")]
NAMES = {"gemma-3-27b-it": "Gemma-27B", "gemma-3-12b-it": "Gemma-12B", "qwen3.6-27b": "Qwen3.6-27B", "qwen3-32b": "Qwen3-32B"}
METHODS = {"J-lens words": "J-lens · word counts", "J-lens meaning": "J-lens · meaning vectors", "J-lens judge": "J-lens · judge model",
           "NLA words": "NLA · word counts", "NLA meaning": "NLA · meaning vectors", "NLA judge": "NLA · judge model",
           "probe": "probe on the activation"}
LABEL, POSITION = "lie vs truthful", "before_answer"
SUBSTITUTE, OWN = "#e8663a", "#2a78d6"
BANDS = {"J-lens": "#dcefe9", "NLA": "#e9e2f4"}   # a pale tint behind the rows of one readout; the row label names it too


def main() -> None:
    results = {p: json.loads((LOCAL / f"gap_{p[0]}__{p[1]}.json").read_text()) for p in PAIRS}
    for p, r in results.items():   # the two later scripts' rows join gap.py's
        for name in ("meaning_gap", "judge_gap"):
            r["gaps"] |= json.loads((LOCAL / f"{name}_{p[0]}__{p[1]}.json").read_text())["gaps"]
    shown = {p: [m for m in METHODS if any(k.startswith(f"{p[0]} -> {p[1]} | {m} | {POSITION} | {LABEL}") for k in r["gaps"])]
             for p, r in results.items()}
    REPORT.mkdir(exist_ok=True)
    with rc_context(STYLE):
        fig, axes = charts.figure(16, 3.2 + 0.42 * sum(len(m) + 1.5 for m in shown.values()), cols=2)
        for ax, split in zip(axes[0], ("same type", "unseen type")):
            y, ticks = 0, []
            for (t, s), r in results.items():
                pair = f"{t} -> {s}"
                words_only = r["gaps"][f"{pair} | words only | {LABEL} | {split}"]["target_only"]["target"]
                ax.text(-0.02, y, f"{NAMES[t]} → {NAMES[s]}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=9.5, fontweight="bold", color=INK)   # the pair, once, above its rows
                y += 0.9
                first = y
                for readout, tint in BANDS.items():
                    rows = [first + i for i, m in enumerate(shown[t, s]) if m.startswith(readout)]
                    if rows:
                        ax.axhspan(rows[0] - 0.5, rows[-1] + 0.5, color=tint, linewidth=0, zorder=0)
                for method in shown[t, s]:
                    g = r["gaps"][f"{pair} | {method} | {POSITION} | {LABEL} | {'no training' if 'judge' in method else split}"]
                    summary = g.get("substitute") or g["own"]
                    ax.plot(summary["target"], y, "o", color=INK, markersize=8, label="target" if not ticks else None)
                    if "substitute" in g:
                        ax.plot(g["substitute"]["substitute"], y, "o", color=SUBSTITUTE, markersize=8,
                                label="substitute, through the target's classifier" if not ticks else None)
                    if "own" in g:
                        ax.plot(g["own"]["substitute"], y, "o", color=OWN, markersize=8, markerfacecolor="none", markeredgewidth=2,
                                label="substitute, with its own classifier" if not ticks else None)
                    ticks.append((y, METHODS[method]))
                    y += 1
                ax.plot([words_only, words_only], [first - 0.4, y - 0.6], color=MUTED, linewidth=2, linestyle=(0, (2, 2)),
                        label="words only (no model)" if len(ticks) == len(shown[t, s]) else None)
                y += 0.6
            ax.axvline(0.5, color=MUTED, linewidth=1)
            ax.set_yticks([p for p, _ in ticks], [name for _, name in ticks], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0.3, 1.0)
            charts.style(ax)
            ax.set_xlabel("AUROC: the target's lies against its truthful replies (mean over categories)")
            ax.set_title("Trained and tested on the same kind of lie" if split == "same type"
                         else "Trained on the 9 other kinds of lie, tested on the 1 kind left out", loc="left", fontsize=9)
        axes[0][1].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle("Faithfulness gap with J-space and NLA: does a substitute's readout say what the target's says?\n"
                     "Orange left of black  →  the target's classifier fails on the substitute (a faithfulness gap).      "
                     "Blue ring on the black dot  →  the substitute's own classifier closes the gap.\n"
                     "Dot on the 0.5 line  →  no better than chance.      Dot left of the dotted line  →  worse than just "
                     "reading the conversation's words.",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Rows are grouped by target → substitute pair (bold heading). Inside a pair, the green band holds "
                         "the J-lens rows and the purple band the NLA rows; the row with no band is the probe.",
                    "Each row: one way of reading a model. The target reads its own "
                         "lie-detection conversations; the substitute is prefilled with the same conversations. Both are "
                         "read at the token just before the answer to \"Did you lie in your previous answer?\".",
                    "Black dot: a classifier trained on the target's readouts and tested on held-out target rows. "
                    "Orange dot: the same classifier tested on the substitute's readouts of those rows. The distance "
                    "from black to orange is the faithfulness gap. Blue ring: the substitute with a classifier trained "
                    "on its own readouts.",
                    "Word counts: a bag-of-words classifier on the readout's words. Meaning vectors: the arena's probe "
                    "recipe on the readout's embedding (a multilingual embedding model), so synonyms and translations "
                    "count. Judge model: GPT-5 mini reads one readout and rates 0-10 how likely the previous answer was "
                    "a lie; nothing is trained, so its rows are the same in both panels. Probe: the arena's probe on "
                    "the activation itself.",
                    "Each dot is the mean AUROC over the lie categories with at least 10 rows per class. 0.5 = chance. "
                    "Dotted grey line: the same classifier on the conversation's own words, with no model. A row with no "
                    "orange dot is a probe: activations of two models don't line up, so it only has its own version.")
    path = REPORT / "faithfulness_gap.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
