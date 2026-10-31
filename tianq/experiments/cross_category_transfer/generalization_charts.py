"""Train-on-one, test-on-the-others matrices drawn as heatmaps, one panel each:
generalization.py's (one per follow-up wording) and no_followup.py's (one per read
position). Every cell is an AUROC on a diverging scale centred on chance: gray at
0.5, blue where the probe catches the lies, red where it ranks them backwards. The outlined diagonal is the
in-category reference. Cells with fewer than generalization.FEW double downs or
truthful denials are faded, and categories too thin to train on are labelled in
their row instead of coloured.
"""

from __future__ import annotations

import textwrap

from matplotlib import rc_context
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from experiments.common.chart_style import AUROC_SCALE, INK, MUTED, SECONDARY, STYLE, SURFACE
from experiments.cross_category_transfer.generalization import CATEGORIES, FEW

PANEL_INCHES = 6.0
HEADER_INCHES = 1.2  # headline and caption above the first panel


def _panel(ax, m: dict, measure: str, title: str) -> None:
    columns = list(CATEGORIES)
    rows = m["rows"]
    for y, row in enumerate(rows):
        if not row["trained"]:
            ax.text((len(columns) - 1) / 2, y, f"not trained: {row['n_lie']} lies / {row['n_honest']} honest (fewer than {FEW} in a class)",
                    ha="center", va="center", color=MUTED, fontsize=7, style="italic")
            continue
        for x, cell in enumerate(row["cells"][:len(columns)]):
            numbers = cell[measure]
            if numbers is None or numbers["auroc"] is None:
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=SURFACE, edgecolor="#e1e0d9", lw=0.6))
                label = "-" if numbers is None else f"{numbers['n_lie']}/{numbers['n_honest']}"
                ax.text(x, y, label, ha="center", va="center", color=MUTED, fontsize=6)
                continue
            few = min(numbers["n_lie"], numbers["n_honest"]) < FEW
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=AUROC_SCALE(numbers["auroc"]),
                                   edgecolor=SURFACE, lw=1.5, alpha=0.3 if few else 1.0))
            # Mid-tone cells already need dark-on-light help: white text only on the deep ends.
            dark = not few and abs(numbers["auroc"] - 0.5) > 0.25
            ink = "white" if dark else (MUTED if few else INK)
            ax.text(x, y - 0.12, f"{numbers['auroc']:.2f}", ha="center", va="center", fontsize=7.5, color=ink,
                    weight="normal" if few else "bold")
            ax.text(x, y + 0.25, f"{numbers['n_lie']}/{numbers['n_honest']}", ha="center", va="center", fontsize=5.5,
                    color=ink)
            if cell["kind"] == "in_category":
                ax.add_patch(Rectangle((x - 0.47, y - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, lw=1.4))
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(columns)), [CATEGORIES[c] for c in columns], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows)), [CATEGORIES[r["train"]] for r in rows], fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("tested on", color=SECONDARY)
    ax.set_ylabel("probe trained on", color=SECONDARY)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)


def draw(panels: list[tuple[str, dict]], measure: str, headline: str, caption: str) -> Figure:
    """panels: (title, matrix) in drawing order; measure: the cell key drawn."""
    matrices = [m for _, m in panels]
    height = PANEL_INCHES * len(matrices) + HEADER_INCHES
    with rc_context(STYLE):
        fig = Figure(figsize=(10.5, height), facecolor=SURFACE)
        axes = fig.subplots(len(matrices), 1)
        for ax, (title, m) in zip(axes, panels):
            ax.set_facecolor(SURFACE)
            _panel(ax, m, measure, title)
        fig.text(0.02, 1 - 0.25 / height, headline, ha="left", va="top", fontsize=12, weight="bold", color=INK)
        fig.text(0.02, 1 - 0.55 / height, textwrap.fill(caption, 145), ha="left", va="top", fontsize=8,
                 color=SECONDARY, linespacing=1.4)
        fig.subplots_adjust(left=0.17, right=0.9, top=1 - (HEADER_INCHES + 0.3) / height, bottom=1.1 / height,
                            hspace=0.5)
        bar = fig.colorbar(ScalarMappable(cmap=AUROC_SCALE), ax=list(axes), fraction=0.02, pad=0.02)
        bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0 (inverted)", "0.25", "0.5 (chance)", "0.75", "1 (perfect)"])
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, length=0)
    return fig
