"""The figure for the scoring-method comparison: which way of scoring J-space overlap is best, and what it says.

    cd tianq && READOUT_DISTRIBUTIONS=<folder of the result JSON files> python -m experiments.internal_readout_gap.readout_diffing.figures

Reads similarity_methods.json, similarity_methods_rarity.json and yardstick_ties.json (own lenses, all words) and
writes report/scoring_methods.png:
  left    every scoring method on the yardstick, two-thirds of the way in and near the output: for base gemma against
          Qwen3.6, how often the same restart point scores above a different restart point of the same task (a tie
          counts as half; 0.5 = chance). Sorted by the two-thirds value, coloured by family.
  right   the distillation result by depth under the method the report used (top 200 words) and under the best method
          (rarity-weighted, top 5,000): the final checkpoint's change from base gemma, with its 95% band.
The figure carries a "How to read this" note.
"""

from __future__ import annotations

import json
import os
import pathlib

from matplotlib import rc_context
from matplotlib.patches import Patch

from experiments.common.chart_style import INK, MUTED, STYLE
from experiments.yuqi_questions.shared import charts

FOLDER = pathlib.Path(os.environ["READOUT_DISTRIBUTIONS"])
REPORT = pathlib.Path(__file__).resolve().parents[1] / "report"
PLAIN, WEIGHTED, RARITY, MEANING = "#2a78d6", "#e8663a", "#1c9a8a", "#8a8a8a"
# method name in the JSON files -> (label, family colour)
METHODS = {
    "2  Jaccard, top 10": ("shared words, top 10", PLAIN),
    "2  Jaccard, top 50": ("shared words, top 50", PLAIN),
    "1  Jaccard, top 200": ("shared words, top 200  (the method we reported)", PLAIN),
    "2  Jaccard, top 1,000": ("shared words, top 1,000", PLAIN),
    "2  Jaccard, top 5,000": ("shared words, top 5,000", PLAIN),
    "3  rank-biased overlap, top 200, p = 0.9": ("top-ranked words count more (rank-weighted)", WEIGHTED),
    "4  probability overlap (sum of min)": ("confident words count more: probability overlap", WEIGHTED),
    "5  Bhattacharyya overlap": ("confident words count more: Bhattacharyya", WEIGHTED),
    "6  weighted Jaccard (min / max)": ("confident words count more: weighted Jaccard", WEIGHTED),
    "7  cosine of probabilities": ("confident words count more: cosine", WEIGHTED),
    "8c rarity x probability cosine, top 5,000": ("rare and confident words count more, top 5,000", WEIGHTED),
    "8b rarity-weighted cosine, top 50": ("rare words count more, top 50", RARITY),
    "8  rarity-weighted cosine, top 200": ("rare words count more, top 200", RARITY),
    "8b rarity-weighted cosine, top 1,000": ("rare words count more, top 1,000", RARITY),
    "8b rarity-weighted cosine, top 5,000": ("rare words count more, top 5,000  (best)", RARITY),
    "10 meaning: probability-weighted centroid, top 200": ("meaning vectors (synonyms count)", MEANING),
}
DEPTHS = {"0.250": "a quarter in", "0.500": "halfway", "0.667": "two-thirds\n(headline)", "0.750": "three-\nquarters",
          "0.850": "0.85", "0.950": "0.95"}
PREFIX = "{depth} | own lens | all words | {method}"


def main() -> None:
    methods = json.loads((FOLDER / "similarity_methods.json").read_text())["methods"]
    methods |= json.loads((FOLDER / "similarity_methods_rarity.json").read_text())["methods"]
    ties = json.loads((FOLDER / "yardstick_ties.json").read_text())

    def yardstick(method: str, depth: str) -> float:
        """Ties as half where yardstick_ties.py has the method; the strict value otherwise (meaning vectors: no ties)."""
        key = f"depth {depth} | {method}"
        return ties[key]["tie_as_half"] if key in ties else methods[PREFIX.format(depth=f"depth {depth}", method=method)]["beats_same_task"]

    order = sorted(METHODS, key=lambda m: yardstick(m, "0.667"))
    with rc_context(STYLE):
        fig, axes = charts.figure(21, 9.5, cols=2)
        ax = axes[0][0]
        y = list(range(len(order)))
        ax.barh(y, [yardstick(m, "0.667") - 0.5 for m in order], left=0.5, color=[METHODS[m][1] for m in order], height=0.62)
        ax.scatter([yardstick(m, "0.950") for m in order], y, color=INK, marker="D", s=26, zorder=3, label="the same, near the output (depth 0.95)")
        for i, m in enumerate(order):
            ax.annotate(f"{yardstick(m, '0.667'):.2f}", (0.5, i), textcoords="offset points", xytext=(5, 0),
                        ha="left", va="center", fontsize=8, color="white")   # at the bar's start, clear of the diamonds
        ax.set_yticks(y, [METHODS[m][0] for m in order], fontsize=9)
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("how often the same restart point scores above a different one of the same task (0.5 = chance; higher = better)")
        ax.set_title("LEFT. Which way of scoring J-space overlap tells restart points apart best?\n"
                     "Bars: two-thirds of the way in. Diamonds: near the output.", loc="left", fontsize=9)
        charts.style(ax)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Patch(color=PLAIN, label="every word counts the same"), Patch(color=WEIGHTED, label="confident or top-ranked words count more"),
                           Patch(color=RARITY, label="rare words count more"), Patch(color=MEANING, label="meaning vectors"),
                           *ax.get_legend_handles_labels()[0]], frameon=False, fontsize=8, loc="lower right")

        ax = axes[0][1]
        x = list(range(len(DEPTHS)))
        for k, (method, colour, label) in enumerate((("1  Jaccard, top 200", PLAIN, "shared words, top 200 (the method we reported)"),
                                                      ("8b rarity-weighted cosine, top 5,000", RARITY, "rare words count more, top 5,000 (best on the yardstick)"))):
            r = [methods[PREFIX.format(depth=f"depth {d}", method=method)] for d in DEPTHS]
            xs = [v + (-0.07 if k == 0 else 0.07) for v in x]
            ax.plot(xs, [v["change"] for v in r], color=colour, marker="o" if k == 0 else "D", linewidth=2.2, label=label)
            ax.fill_between(xs, [v["ci"][0] for v in r], [v["ci"][1] for v in r], color=colour, alpha=0.14, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xticks(x, list(DEPTHS.values()))
        ax.set_xlabel("depth in the network (share of the 62 layers)")
        ax.set_title("RIGHT. Does the best method change the distillation result?\n"
                     "Every line compares the final (100%) checkpoint with base Gemma, each through its own fresh lens.", loc="left", fontsize=9)
        charts.style(ax, "change in J-space similarity to Qwen3.6, above chance\n(final checkpoint minus base Gemma; up = toward Qwen3.6)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle("How should two models' J space be compared? Sixteen ways of scoring overlap, on one yardstick\n"
                     "Left: longer bar  →  the method is better at telling one moment from another.      "
                     "Right: line above 0  →  distillation moved Gemma toward Qwen3.6 at that depth.", fontsize=11, x=0.01, ha="left")
        n = json.loads((FOLDER / "similarity_methods.json").read_text())["restart_points"]
        charts.note(fig, f"Left panel, the yardstick: for base Gemma against Qwen3.6, over the {n} restart points of Gigi's hero swap, how "
                         "often a method rates the two models' J space at the same restart point above their J space at two "
                         "different restart points of the same task. A tie counts as half. It uses base Gemma only, so the "
                         "ranking cannot be tuned to the distillation result.",
                    "Left panel, colours: blue = shared words in the top N, every word counting the same. Orange = the lens's "
                    "most confident or top-ranked words count more. Green = words that are in a model's list at few restart "
                    "points count more (rarity). Grey = an embedding model, so synonyms count.",
                    "Right panel: the same test as the fresh-lens figure, under two methods. y: (same - chance) for the "
                    "distilled Gemma minus (same - chance) for base Gemma, with chance taken over the other restart points. "
                    "Band: 95% interval from resampling restart points. The two methods are on different scales: compare "
                    "direction, not size.",
                    "All of this was tried after the fresh-lens result was in, so it is a check on that result, not the "
                    "test fixed in advance. Words here come from the lens's top 5,000 tokens at each depth.")
    REPORT.mkdir(exist_ok=True)
    path = REPORT / "scoring_methods.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


def nla_root_cause() -> None:
    """report/nla_root_cause.png, from nla_root_cause.json (nla/root_cause.py --to <folder>): where the negative NLA
    result comes from, and the other tasks with rare words counted more."""
    source = FOLDER / "nla_root_cause.json"
    if not source.exists():
        print("nla_root_cause.png: skipped, nla_root_cause.json is missing", flush=True)
        return
    r = json.loads(source.read_text())
    task = "currency_conversion"
    other = f"every task but {task}"
    red, grey = "#b8312f", "#8a8a8a"
    with rc_context(STYLE):
        fig, axes = charts.figure(23, 8, cols=3)
        ax = axes[0][0]
        parts = [("currency conversion\n(62 restart points)", r["split"][task]["adds_to_overall"], red),
                 ("the other 12 tasks\n(264 restart points)", r["split"][other]["adds_to_overall"], grey),
                 ("total\n(the reported result)", r["split"]["all"]["same_minus_chance"][0], INK)]
        ax.bar(range(3), [v for _, v, _ in parts], color=[c for _, _, c in parts], width=0.55)
        for i, (_, v, _) in enumerate(parts):
            ax.annotate(f"{v:+.4f}", (i, v), textcoords="offset points", xytext=(0, 5 if v >= 0 else -13), ha="center", fontsize=9)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xticks(range(3), [label for label, _, _ in parts])
        ax.set_title("LEFT. Where does the negative NLA result come from? (every word counts the same, as reported)\n"
                     "Each bar: how much that group of restart points adds to the total.", loc="left", fontsize=9)
        charts.style(ax, "contribution to the change in NLA word overlap with Qwen3.6\n(distilled Gemma minus base Gemma, above chance)")

        ax = axes[0][1]
        rows = [("every word counts\nthe same (as reported)", r["split"][other]["same_minus_chance"], grey),
                ("rare words\ncount more", r["remove"]["rarity"]["other_tasks"], "#1c9a8a")]
        for i, (_, (m, lo, hi), colour) in enumerate(rows):
            ax.bar([i], [m], color=colour, width=0.5)
            ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], color=INK, capsize=4, linewidth=1.2)
            ax.annotate(f"{m:+.4f}", (i, hi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xticks(range(2), [label for label, _, _ in rows])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title("RIGHT. On the other 12 tasks, did the distilled Gemma's NLA move toward Qwen3.6?\nAbove 0 = it moved toward Qwen3.6.", loc="left", fontsize=9)
        charts.style(ax, "change in NLA word overlap with Qwen3.6, above chance\n(distilled Gemma minus base Gemma)")
        ax = axes[0][2]
        scale = r["scale"]
        charts.way_to_the_top(ax, [("the other\n12 tasks", scale["gemma-3-27b-it"] / scale["top"], scale["gemma-distilled-step96"] / scale["top"])],
                              ("base Gemma", grey), ("distilled Gemma", "#e8663a"), "Qwen3.6's NLA\nagainst itself")
        ax.set_title("SCALE. How close is each Gemma's NLA to Qwen3.6's? (the other 12 tasks, rare words count more)\n"
                     "0% = no closer than a random restart point. 100% = Qwen3.6's NLA against itself.", loc="left", fontsize=9)
        ax.set_xlabel("share of the way from a random restart point to Qwen3.6's NLA against itself")
        fig.suptitle("Why is the NLA result negative? One task carries all of it\n"
                     "Left: bar below 0  →  that group pulls the result down.      "
                     "Right: bar above 0  →  a move toward Qwen3.6.      Scale: further right  →  closer to Qwen3.6.",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, f"All {r['split']['all']['points']} restart points of Gigi's hero swap are used, with 5 NLA sentences per model at "
                         "each. The NLA score is same - chance: the cosine of two models' NLA word counts at the same restart point, "
                         "minus the same against other restart points. The result is the distilled Gemma's score minus base Gemma's.",
                    "Left panel: the three bars add up. Each is that group's summed change divided by all restart points.",
                    "Right panel: the same result on the 264 restart points outside currency conversion, scored two ways. Rare "
                    "words count more = each word is weighted by how few restart points have it, so stock phrases count "
                    "for little. Whisker: 95% interval from resampling restart points.",
                    "Scale panel, the two numbers behind a mark: same = how alike Qwen3.6's NLA text and a Gemma's NLA text are at "
                    "the same restart point. chance = how alike Qwen3.6's NLA text at that restart point is to that Gemma's NLA "
                    "text at every other restart point, averaged. Any two NLA texts share some common words; chance is how much.",
                    "Scale panel, 0%: same equals chance. The Gemma's NLA at the matching restart point is then no more like "
                    "Qwen3.6's than its NLA at unrelated restart points. A single restart point can fall below 0%.",
                    "Scale panel, 100%: same - chance for Qwen3.6 against itself. The NLA writes new sentences every time it reads "
                    "an activation, so even an exact copy of Qwen3.6 would not get identical text. 100% is therefore two separate "
                    "NLA readings of the same Qwen3.6 activation: three of its five sentences against the other two. Each Gemma "
                    "is scored with the same sentence counts, Qwen3.6's three against that Gemma's two, so the marks are comparable.",
                    "Scale panel, a mark: that Gemma's (same - chance) divided by Qwen3.6's own. Rare words count more.")
    path = REPORT / "nla_root_cause.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
    nla_root_cause()
