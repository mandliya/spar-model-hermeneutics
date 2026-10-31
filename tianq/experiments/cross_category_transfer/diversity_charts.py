"""The curves of diversity.py, drawn: held-out AUROC against the number of categories trained on,
one panel per variant.

Each faint mark is one training subset (its mean AUROC over the held-out categories); the line joins
the means at each k. The two training sets are two series, told apart by colour, marker and a
direct label. Chance (0.5) is the axis line; the dashed line is the in-category reference, each
category's own probe cross-validated. On the no-follow-up panel, the band is the held-out range
Ant's post reports for its fine-tuned black-box detector on Gemma-3-27B trained on half its lie
types: the same question (was the reply a lie?) on overlapping, not identical, categories.
"""

from __future__ import annotations

import textwrap

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure

from experiments.common.chart_style import AXIS, GRID, INK, MUTED, SECONDARY, STYLE, SURFACE
from experiments.cross_category_transfer.diversity import KS, LINES, SIZE_MATCHED_ROWS

# Categorical slots 1 and 2 of the reference palette (dataviz references/palette.md).
SERIES = {"all_rows": ("#2a78d6", "o"), "size_matched": ("#eb6834", "D")}
LABELS = {"all_rows": "all rows, categories weighted equally",
          "size_matched": f"{SIZE_MATCHED_ROWS} rows in all, whatever k"}
TITLES = {"none": "No follow-up: the reply itself (mean over its tokens)\n"
                  "lie vs truthful reply (ASCII not scored held out: its reply is just a letter)",
          "neutral": "After Gigi's \"Did you lie in your previous answer?\" (just before the answer)\n"
                     "doubled-down lies vs truthful denials"}
ANT_BAND = (0.70, 0.75)  # Ant's post: cross-fold AUROC "plateaued at 0.70-0.75", half the lie types held out
OFFSET = {"all_rows": -0.14, "size_matched": 0.14}


def _panel(ax, variant: str, records: list[dict], summary: dict, reference: float, show_ant: bool) -> None:
    rng = np.random.default_rng(0)  # jitter only
    ax.axhline(0.5, color=AXIS, linewidth=1.2, zorder=1)
    ax.axhline(reference, color=SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(KS[0] - 0.4, reference + 0.008, f"trained and tested on the same category (5-fold): {reference:.2f}",
            va="bottom", fontsize=7.5, color=SECONDARY)
    # Ant's number is for Gemma-3-27B, so it is drawn on Gemma's chart only.
    if variant == "none" and show_ant:
        ax.axhspan(*ANT_BAND, color=GRID, zorder=0, linewidth=0)
        ax.text(KS[0] - 0.4, sum(ANT_BAND) / 2, "Ant's fine-tuned black-box detector, trained on half its lie types",
                va="center", fontsize=7.5, color=SECONDARY)
    for line in LINES:
        colour, marker = SERIES[line]
        for r in records:
            if r["line"] == line and r.get("held_out_mean") is not None:
                x = r["k"] + OFFSET[line] + rng.uniform(-0.06, 0.06)
                ax.plot(x, r["held_out_mean"], marker=marker, markersize=4, color=colour, alpha=0.3,
                        linestyle="none", markeredgewidth=0, zorder=2)
        points = [(k + OFFSET[line], s["mean"]) for k, s in summary[line].items() if s["mean"] is not None]
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colour, linewidth=2, zorder=3)
        ax.plot(xs, ys, marker=marker, markersize=8, color=colour, linestyle="none", zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(xs[-1] + 0.3, ys[-1], f"{ys[-1]:.2f}", va="center", fontsize=8, color=INK, weight="bold")
    ax.set_xticks(KS)
    ax.set_xlim(KS[0] - 0.6, KS[-1] + 1.3)
    ax.set_xlabel("categories trained on (k); tested on the other 11 - k", color=SECONDARY)
    ax.set_ylabel("held-out AUROC (mean over held-out categories)", color=SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(TITLES[variant], loc="left", fontsize=9.5, color=INK)


def draw(records: list[dict], summaries: dict[str, dict], references: dict[str, float], model, is_default: bool,
         author=None) -> Figure:
    """records: every probe of diversity.run, with its variant; summaries: variant -> diversity.summary;
    model: core.model.Model, the model read; author: the model whose conversations and lies it reads
    (model itself when None)."""
    name = model.name.split("/")[-1]
    author_name = None if author is None or author.name == model.name else author.name.split("/")[-1]
    reads = (f"{name} reading its own conversations, labelled by its own lies." if author_name is None else
             f"{name} reading {author_name}'s conversations, labelled by {author_name}'s lies.")
    values = [r["held_out_mean"] for r in records if r.get("held_out_mean") is not None]
    low = min(0.4, min(values) - 0.03)
    high = max(0.85, max(values) + 0.03, *(v + 0.05 for v in references.values()))
    with rc_context(STYLE):
        fig = Figure(figsize=(13, 6.2), facecolor=SURFACE)
        axes = fig.subplots(1, len(summaries), sharey=True)
        for ax, (variant, summary) in zip(axes, summaries.items()):
            ax.set_facecolor(SURFACE)
            _panel(ax, variant, [r for r in records if r["variant"] == variant], summary, references[variant], is_default)
            ax.set_ylim(low, high)
        handles = [axes[0].plot([], [], color=SERIES[line][0], marker=SERIES[line][1], markersize=7, linewidth=2,
                                label=LABELS[line])[0] for line in LINES]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.865), ncols=2, frameon=False, fontsize=8.5)
        fig.text(0.06, 0.975, f"{name}: does training on more kinds of lie help a probe catch a kind it never saw?",
                 ha="left", va="top", fontsize=12, weight="bold", color=INK)
        fig.text(0.06, 0.93, textwrap.fill(
            f"{reads} AISI's probe recipe (logistic"
            f" regression, layer {model.probe_layer}), trained on k of Gigi's 11 categories and scored on"
            " each of the others. Faint marks: one training subset each (all subsets for k = 1 and 10, 20 random"
            " ones otherwise), its AUROC averaged over the held-out categories with at least 10 of each class."
            " Line: the mean over subsets. Gray line: chance (0.5).", 190), ha="left", va="top", fontsize=8, color=SECONDARY, linespacing=1.4)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.1, wspace=0.12)
    return fig
