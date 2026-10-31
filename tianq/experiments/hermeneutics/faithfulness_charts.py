"""faithfulness.py's gaps, drawn: one panel per read, a row per category (with its lie / truthful counts),
its in-category AUROC for the target (black bar) and for each substitute reading the target's conversations
(colours), each with its gap's 95% interval as a whisker. A substitute that stands in well sits on the
black bar; the gap is the horizontal distance. Beside each panel, every substitute's mean gap ± its SD
across categories. Categories with fewer than FEW of a class are faded."""

from __future__ import annotations

import textwrap

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments.common.chart_style import AXIS, GRID, INK, MUTED, SECONDARY, STYLE, SURFACE
from experiments.cross_category_transfer.generalization import CATEGORIES, FEW
from experiments.hermeneutics.faithfulness import READS, summary

# Categorical slots of the reference palette for the substitutes; gray for the no-model baseline.
COLOURS = ["#2a78d6", "#eb6834", "#1f9e89", "#8a8a8a"]
MARKERS = ["o", "D", "s", "X"]
TITLES = {"none": "No follow-up, the reply itself (mean over its tokens): lies vs truthful replies",
          "neutral": "\"Did you lie in your previous answer?\", read just before the answer: doubled-down vs truthful denials"}


def _panel(ax, variant: str, gaps: dict[str, dict], names: dict[str, str]) -> None:
    first = next(iter(gaps.values()))[variant]
    rows = [c for c in CATEGORIES if first.get(c) and first[c]["target"] is not None]
    ax.axvline(0.5, color=AXIS, linewidth=1.2, zorder=1)
    for y, category in enumerate(rows):
        alpha = 1.0 if first[category]["trusted"] else 0.35
        ax.axhline(y, color=GRID, linewidth=0.6, zorder=0)
        for i, by_variant in enumerate(gaps.values()):
            g = by_variant[variant].get(category)
            if g is None:
                continue
            # The gap's interval, placed around the substitute: target - ci. It reaches the bar when 0 is inside.
            low, high = g["target"] - g["ci"][1], g["target"] - g["ci"][0]
            ax.errorbar(g["substitute"], y + (i - (len(gaps) - 1) / 2) * 0.17,
                        xerr=[[max(g["substitute"] - low, 0)], [max(high - g["substitute"], 0)]],
                        color=COLOURS[i], marker=MARKERS[i], markersize=5, elinewidth=1, capsize=0, alpha=alpha,
                        zorder=3, linestyle="none")
        ax.scatter(first[category]["target"], y, color=INK, marker="|", s=320, linewidths=2.4, alpha=alpha, zorder=4)
    labels = [f"{CATEGORIES[c]}  ({first[c]['n_lie']} / {first[c]['n_honest']})" for c in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    for tick, c in zip(ax.get_yticklabels(), rows):
        if not first[c]["trusted"]:
            tick.set_color(MUTED)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("in-category AUROC (5-fold)", color=SECONDARY)
    ax.set_title(TITLES[variant], loc="left", fontsize=9.5, color=INK, weight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # The summary beside the panel: what the panel's means say, numbers in bold.
    handles = [Line2D([], [], color=INK, marker="|", markersize=14, markeredgewidth=2.4, linestyle="none")]
    entries = ["target: reads its own\nconversations"]
    for i, (tag, by_variant) in enumerate(gaps.items()):
        m = summary(by_variant[variant])
        if m["gap"] is None:
            continue
        handles.append(Line2D([], [], color=COLOURS[i], marker=MARKERS[i], markersize=6, linestyle="none"))
        entries.append(f"{names[tag]}\ngap $\\mathbf{{{m['gap']:+.3f} \\pm {m['gap_sd']:.3f}}}$")
    n = summary(next(iter(gaps.values()))[variant])["categories"]
    ax.legend(handles, entries, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5,
              labelspacing=1.1, title=f"Mean gap ± SD over {n} categories\n(≥ {FEW} of each class)",
              title_fontproperties={"size": 8.5, "weight": "bold"}, alignment="left")


def draw(gaps: dict[str, dict], names: dict[str, str], target: str, caption: str) -> Figure:
    """gaps: substitute tag -> read -> category -> faithfulness.gap; names: tag -> label."""
    with rc_context(STYLE):
        fig = Figure(figsize=(12.5, 12), facecolor=SURFACE)
        axes = fig.subplots(len(READS), 1)
        for ax, variant in zip(axes, READS):
            ax.set_facecolor(SURFACE)
            _panel(ax, variant, gaps, names)
        fig.text(0.02, 0.99, f"Can a substitute's probe stand in for {target}'s own?", ha="left", va="top",
                 fontsize=13, weight="bold", color=INK)
        fig.text(0.02, 0.965, textwrap.fill(caption + " Row labels: lies / truthful rows.", 170), ha="left", va="top",
                 fontsize=8, color=MUTED, linespacing=1.4)
        fig.subplots_adjust(left=0.2, right=0.79, top=0.87, bottom=0.05, hspace=0.28)
    return fig
