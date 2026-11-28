"""Figures for the own-lens run: does the borrowed lens hide the distilled gemma's move, and where to read.

    cd tianq && READOUT_LOCAL=<folder holding the JSON results> python -m experiments.internal_readout_gap.own_lens.figures

Writes, next to the report (a figure is skipped, with a line saying so, when its results are not there):
  report/own_lens_by_depth.png        from distributions/similarity_methods_rarity.json (readout_diffing/jspace/
                                      similarity_methods.py --selection rare_words) and the lens fit in
                                      similarity_<name>.json (after_fits.sh): the final checkpoint's change in J-space
                                      similarity to Qwen3.6 by depth, with rare words counted more, read through the
                                      borrowed lens and through each model's own lens; how well each lens fits; and
                                      how far along each gemma is from "a random restart point" to "Qwen3.6 itself"
  report/positions_distillation.png   from positions_similarity.json: the same change at each of 9 token positions
  report/positions_lie_detector.png   from positions_gap_<target>__<substitute>.json: the lie detector's AUROC at
                                      each of 11 token positions
Every figure carries a "How to read this" note.
"""

from __future__ import annotations

import json
import os
import pathlib

from matplotlib import rc_context

from experiments.common.chart_style import MUTED, STYLE
from experiments.internal_readout_gap.readout_diffing.method_menu import BEST
from experiments.yuqi_questions.shared import charts

HERE = pathlib.Path(__file__).resolve().parent
LOCAL = pathlib.Path(os.environ.get("READOUT_LOCAL", HERE.parents[2] / "dyl-probe-repro" / "scratch" / "internal-readout-gap"))
REPORT = HERE.parent / "report"
BASE, FINAL = "gemma-3-27b-it", "gemma-distilled-step96"
DEPTHS = {"0.250": "a quarter in", "0.500": "halfway", "0.667": "two-thirds\n(headline)", "0.750": "three-\nquarters",
          "0.850": "0.85", "0.950": "0.95"}
SHADES = {"0.250": "#cfe0f5", "0.500": "#a9c9ee", "0.667": "#7aaee6", "0.750": "#4a92dc", "0.850": "#2a78d6", "0.950": "#184f95"}
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
                    "Right panel, each row: one depth. Grey circle = base Gemma, orange diamond = the distilled Gemma. same and "
                    "chance are as in the left panel: chance is Qwen3.6 at a restart point against that Gemma at every other "
                    "restart point, averaged.",
                    "Right panel, 0%: same equals chance. The Gemma at the matching restart point is then no more like Qwen3.6 "
                    "than the Gemma at unrelated restart points. A single restart point can fall below 0%.",
                    "Right panel, 100%: same - chance for Qwen3.6 against itself, where same is a perfect match. A mark is that "
                    "Gemma's (same - chance) divided by Qwen3.6's. A different model with a different tokenizer cannot reach "
                    "100%, so read it as a ruler and not as a target.")
    path = REPORT / "own_lens_by_depth.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


def positions_distillation() -> None:
    result = load("positions_similarity.json")
    if not result:
        print("positions_distillation.png: skipped, positions_similarity.json is missing", flush=True)
        return
    names = result["positions"]
    x = list(range(len(names)))
    scores = result["scores"]["own lens"]   # the published-lens scan is in the same file; it gives the same picture
    with rc_context(STYLE):
        fig, axes = charts.figure(11.5, 8.5)
        ax = axes[0][0]
        for d, colour in SHADES.items():
            r = [scores[f"depth {d}"][p]["change"] for p in names]
            ax.plot(x, [v[0] for v in r], color=colour, marker="o", linewidth=2.2, label=DEPTHS[d].replace("-\n", "-").replace("\n", " "))
            ax.fill_between(x, [v[1] for v in r], [v[2] for v in r], color=colour, alpha=0.12, linewidth=0)
        zero(ax)
        ax.set_xticks(x, ["just before\nthe turn\n(headline)", *names[1:]])
        ax.set_xlabel("where in Qwen3.6's next turn the models are read (share of the turn's text)")
        ax.set_title("Each Gemma through a fresh lens fitted on itself. Qwen3.6 through its published lens.", loc="left", fontsize=9)
        charts.style(ax, "change in word overlap with Qwen3.6, above chance\n(final checkpoint minus base Gemma; up = toward Qwen3.6)")
        ax.legend(frameon=False, fontsize=8, title="depth", title_fontsize=8, loc="best")
        fig.suptitle("Where in the turn does the distilled Gemma's J space move toward Qwen3.6?\n"
                     "Above 0  →  at that token position the final checkpoint is closer to Qwen3.6 than base Gemma is.",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, f"Each point: one token position and one depth, over {result['restart_points']} restart points. All three "
                         "models are given the same history followed by Qwen3.6's actual next turn, and are read just before "
                         "that turn and at 8 evenly spaced points through it (by share of the turn's text).",
                    "y: the share of identical words (top 200 J-lens words) a Gemma shares with Qwen3.6 at the same restart "
                    "point and position, minus the same at 20 other restart points (chance), for the final checkpoint "
                    "minus base Gemma. Band: 95% interval from resampling restart points.",
                    "Darker lines are deeper in the network. Every position is shown; none was chosen after seeing the scores.",
                    "Inside the turn all models read the same given text (Qwen3.6's words), so overlap there partly "
                    "reflects the shared input, for base Gemma and the checkpoint alike.")
    path = REPORT / "positions_distillation.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


def positions_lie_detector(target: str = "gemma-3-27b-it", substitute: str = "qwen3.6-27b") -> None:
    result = load(f"positions_gap_{target}__{substitute}.json")
    if not result:
        print(f"positions_lie_detector.png: skipped, positions_gap_{target}__{substitute}.json is missing", flush=True)
        return
    names = result["positions"]
    x = list(range(len(names)))
    pair = f"{target} -> {substitute}"
    labels = ("lie vs truthful", "doubled-down vs truthful denial")
    splits = ("same type", "unseen type")
    series = [("J-lens words", "target", "target's J space", "#0b0b0b", "-"),
              ("J-lens words", "substitute", "substitute's J space, target's classifier", "#e8663a", "-"),
              ("J-lens words", "own", "substitute's J space, its own classifier", "#2a78d6", "-"),
              ("probe", "target", "probe on the target's activations", "#0b0b0b", (0, (5, 3))),
              ("probe", "own", "probe on the substitute's activations", "#2a78d6", (0, (5, 3)))]
    with rc_context(STYLE):
        fig, axes = charts.figure(21, 13, rows=2, cols=2)
        for row, label in enumerate(labels):
            for col, split in enumerate(splits):
                ax = axes[row][col]
                for method, who, name, colour, dash in series:
                    ys = []
                    for p in names:
                        g = result["gaps"][f"{pair} | {method} | {p} | {label} | {split}"]
                        summary = g.get("own") or g.get("substitute")
                        ys.append(None if not summary else summary["target"] if who == "target" else
                                  (g[who]["substitute"] if who in g else None))
                    if any(y is not None for y in ys):
                        ax.plot(x, [float("nan") if y is None else y for y in ys], color=colour, marker="o", markersize=4,
                                linewidth=2, linestyle=dash, label=name)
                words = result["gaps"][f"{pair} | words only | {label} | {split}"]["target_only"]["target"]
                if words is not None:
                    ax.axhline(words, color="#8a8a8a", linewidth=1.6, linestyle=(0, (1, 2)), label="the conversation's own words, no model")
                charts.chance(ax)
                ax.set_ylim(0.3, 1.02)
                ax.set_xticks(x, [n.replace("reply ", "").replace("_", "\n") for n in names])
                ax.set_xlabel("token position: through the tested reply (share of its text), its end, then at \"Did you lie?\"")
                ax.set_title(f"{label}; " + ("trained and tested on the same kind of lie" if split == "same type" else
                             "trained on the other kinds of lie, tested on the one left out"), loc="left", fontsize=9)
                charts.style(ax, "AUROC (higher = better; 0.5 = chance)")
        axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle("Where should a J-space lie detector read? Gemma-3-27B (target) and Qwen3.6-27B (substitute)\n"
                     "Black = the target. Blue close to black  →  the substitute can stand in at that position.      "
                     "Orange below black  →  the target's classifier does not carry over.", fontsize=11, x=0.01, ha="left")
        charts.note(fig, f"Each point: one token position, the mean AUROC over the kinds of lie with enough rows ({result['rows']} "
                         "rows in all). Both models read Gemma-3-27B's own conversations; Qwen3.6 is prefilled with them.",
                    "Solid lines: a bag-of-words classifier on the top 200 J-lens words two-thirds of the way in. Dashed "
                    "lines: a linear probe on the activations near the output (the arena's probe layer). Dotted grey: "
                    "the same classifier on the conversation's own text, with no model.",
                    "Positions: 1/8 to 8/8 through the tested reply, the token that ends the reply, the token before the "
                    "answer to \"Did you lie?\" (the headline position of earlier reports), and the answer token. Every "
                    "position is shown; none was chosen after seeing the scores.",
                    "Top row: was the tested reply a lie. Bottom row: among replies the model defended by answering "
                    "\"No\" to \"Did you lie?\", which were lies.")
    path = REPORT / "positions_lie_detector.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    by_depth()
    positions_distillation()
    positions_lie_detector()


if __name__ == "__main__":
    main()
