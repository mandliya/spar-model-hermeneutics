"""Figures for the own-lens run: does the borrowed lens hide the distilled gemma's move.

    cd tianq && READOUT_LOCAL=<folder holding the JSON results> python -m experiments.internal_readout_gap.own_lens.figures

Writes, next to the report (a figure is skipped, with a line saying so, when its results are not there):
  report/own_lens_by_depth.png        from distributions/similarity_methods_rarity.json (readout_diffing/jspace/
                                      similarity_methods.py --selection rare_words) and the lens fit in
                                      similarity_<name>.json (after_fits.sh): the final checkpoint's change in J-space
                                      similarity to Qwen3.6 by depth, with rare words counted more, read through the
                                      borrowed lens and through each model's own lens; how well each lens fits; and
                                      how far along each gemma is from "a random restart point" to "Qwen3.6 itself"
Every figure carries a "How to read this" note.
"""

from __future__ import annotations

import json
import os
import pathlib

from matplotlib import rc_context

from experiments.common.chart_style import MUTED, STYLE
from experiments.internal_readout_gap.readout_diffing.methods.rare_words import BEST
from experiments.yuqi_questions.shared import charts

HERE = pathlib.Path(__file__).resolve().parent
LOCAL = pathlib.Path(os.environ.get("READOUT_LOCAL", HERE.parents[2] / "dyl-probe-repro" / "scratch" / "internal-readout-gap"))
REPORT = HERE.parent / "report"
BASE, FINAL = "gemma-3-27b-it", "gemma-distilled-step96"
DEPTHS = {"0.250": "a quarter in", "0.500": "halfway", "0.667": "two-thirds\n(headline)", "0.750": "three-\nquarters",
          "0.850": "0.85", "0.950": "0.95"}
BORROWED, OWN, GREY = "#2a78d6", "#e8663a", "#8a8a8a"
# the two lens choices, as named in the method results -> (label, colour, the similarity file holding that choice's
# lens fit, base gemma's and the final checkpoint's tags in that file)
LENSES = {
    "borrowed lens": ("borrowed: both through base Gemma's published lens", BORROWED, "published_lens_reread", "published-reread", "published-reread"),
    "own lens": ("own: each through a fresh lens fitted on itself", OWN, "own_lens", "control-lens", "final-lens"),
}


def load(name: str) -> dict | None:
    path = LOCAL / name
    return json.loads(path.read_text()) if path.exists() else None


def zero(ax) -> None:
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))


def by_depth() -> None:
    methods = load("distributions/similarity_methods_rarity.json")
    fits = {name: load(f"similarity_{file}.json") for name, (_, _, file, _, _) in LENSES.items()}
    if not methods or not all(fits.values()):
        print("own_lens_by_depth.png: skipped, distributions/similarity_methods_rarity.json or a similarity_<name>.json is missing", flush=True)
        return
    scored = {name: [methods["methods"][f"depth {d} | {name} | all words | {BEST}"] for d in DEPTHS] for name in LENSES}
    x = list(range(len(DEPTHS)))
    with rc_context(STYLE):
        fig, axes = charts.figure(25, 8.5, cols=3)
        ax = axes[0][0]
        for k, (name, (label, colour, _, _, _)) in enumerate(LENSES.items()):
            xs = [v + (-0.07 if k == 0 else 0.07) for v in x]   # nudged apart sideways so neither line hides the other
            ax.plot(xs, [r["change"] for r in scored[name]], color=colour, marker="o" if k == 0 else "D", linewidth=2.2, label=label)
            ax.fill_between(xs, [r["ci"][0] for r in scored[name]], [r["ci"][1] for r in scored[name]], color=colour, alpha=0.14, linewidth=0)
        zero(ax)
        ax.set_xticks(x, list(DEPTHS.values()))
        ax.set_xlabel("depth in the network (share of the 62 layers)")
        ax.set_title("LEFT. Did distillation move Gemma toward Qwen3.6, at each depth?\n"
                     "Every line compares the final (100%) checkpoint with base Gemma.", loc="left", fontsize=9)
        charts.style(ax, "change in J-space similarity to Qwen3.6, above chance\n(final checkpoint minus base Gemma; up = toward Qwen3.6)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axes[0][1]
        bars = (("base Gemma\nthrough base Gemma's\npublished lens", "borrowed lens", BASE, 3),
                ("base Gemma\nthrough my fresh\nbase Gemma lens", "own lens", BASE, 3),
                ("distilled Gemma\nthrough the borrowed lens\n(base Gemma's)", "borrowed lens", FINAL, 4),
                ("distilled Gemma\nthrough its own\nfresh lens", "own lens", FINAL, 4))   # 3, 4: where LENSES keeps that model's tag
        cells = [(label, fits[name]["lens_fit"][f"{model}.{LENSES[name][pick]}"]["top_200_overlap"], LENSES[name][1])
                 for label, name, model, pick in bars]
        ax.bar(range(len(cells)), [v for _, v, _ in cells], color=[c for _, _, c in cells], width=0.6)
        for i, (_, v, _) in enumerate(cells):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
        ax.set_xticks(range(len(cells)), [label for label, _, _ in cells], fontsize=8)
        ax.set_title("MIDDLE. How well does each lens fit the model it reads? (read three layers before the end, layer 58 of 62)\n"
                     "This close to the output, a lens that fits should name the words the model itself says next.", loc="left", fontsize=9)
        charts.style(ax, "overlap of the lens's top 200 tokens with the model's\nown top 200 next tokens (higher = better fit)")

        ax = axes[0][2]
        rows = [(label.replace("-\n", "-").replace("\n", " "), r["base_above_chance"] / r["qwen_against_itself"],
                 (r["base_above_chance"] + r["change"]) / r["qwen_against_itself"]) for label, r in zip(DEPTHS.values(), scored["own lens"])]
        charts.way_to_the_top(ax, rows, ("base Gemma", GREY), ("distilled Gemma", OWN), "Qwen3.6\nagainst itself")
        ax.set_title("RIGHT. How close is each Gemma's J space to Qwen3.6's, at each depth? (each through its own fresh lens)\n"
                     "0% = no closer than a random restart point. 100% = Qwen3.6 against itself.", loc="left", fontsize=9)
        ax.set_xlabel("share of the way from a random restart point to Qwen3.6 against itself")
        fig.suptitle("Does base Gemma's lens hide the distilled Gemma's move toward Qwen3.6?\n"
                     "Left: line above 0  →  distillation moved Gemma's J space toward Qwen3.6 at that depth.   At 0  →  no move.      "
                     "Middle: taller bar  →  the lens fits that model better.      Right: further right  →  closer to Qwen3.6.",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, f"Left panel, each point: one depth, averaged over the same {methods['restart_points']} restart points of Gigi's hero "
                         "swap. Base Gemma, the distilled Gemma (final checkpoint) and Qwen3.6 get the identical history and are "
                         "read at the token just before they act.",
                    "Left panel, y: (same - chance) for the distilled Gemma minus (same - chance) for base Gemma. same = how alike "
                    "a Gemma's J space and Qwen3.6's are at the same restart point; chance = the same at other restart points. "
                    "How alike: the lens's top 5,000 words, each weighted by how few lists have it (rare words count more), "
                    "compared by cosine. Band: 95% interval from resampling restart points.",
                    "Left panel, colours: blue = both Gemmas read through base Gemma's published lens (last week's way). "
                    "Orange = each Gemma read through a fresh lens fitted on itself. Qwen3.6 is always read through its own "
                    "published lens. The two lines are nudged slightly apart sideways so neither hides the other.",
                    "Middle panel, each bar: one model read through one lens, near the output (depth 0.95). Height = the share "
                    "of the lens's top 200 tokens that are also among the model's own top 200 next tokens, averaged over "
                    "the restart points. Blue = base Gemma's published lens, orange = a fresh lens.",
                    "Right panel, each row: one depth. A mark is that Gemma's (same - chance) against Qwen3.6, as a share of "
                    "(same - chance) for Qwen3.6 against itself, scored the same way as the left panel. Grey circle = base "
                    "Gemma, orange diamond = the distilled Gemma. A different model with a different tokenizer cannot reach "
                    "100%, so read it as a ruler and not as a target.")
    path = REPORT / "own_lens_by_depth.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    by_depth()


if __name__ == "__main__":
    main()
