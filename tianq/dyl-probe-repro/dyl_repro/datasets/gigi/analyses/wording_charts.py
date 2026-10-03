"""The two tables of wordings.py, drawn as one figure: per category, each method's
held-out AUROC with its best wording (left), and the probe on the hard-to-catch
lies (right). Each value carries a superscript ID for its winning wording; the
key under the panels gives every ID's wording in full, so points that share a
wording share an ID.
"""

from __future__ import annotations

import textwrap

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from dyl_repro.datasets.gigi.analyses.chart_style import AXIS, GRID, INK, MUTED, SECONDARY, STYLE, SURFACE
from dyl_repro.datasets.gigi.analyses.wordings import CONFOUNDED, SPLITS

CATEGORIES = {
    "ascii_sandbagging": "ASCII sandbagging",
    "sycophancy_answer": "Sycophancy A: answer",
    "sycophancy_feedback": "Sycophancy B: feedback",
    "mask_factual": "MASK factual",
    "unanswerable": "Unanswerable",
    "capture_the_secret": "Capture the secret",
}
# Order the IDs are handed out in: row by row, left panel then right.
MARKED = ("probe_deduced", "just_asking", "hard_to_catch")
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
CAPTION_WIDTH = 82  # characters per line under each panel, so nothing crosses the middle
KEY_WIDTH = 88  # characters per line in each of the key's two columns, one under each panel
LINE_INCHES = 0.16

PROBE, ASKING = "#2a78d6", "#eb6834"


def wording_ids(per_dataset: dict[str, dict]) -> dict[str, int]:
    """One ID per distinct winning wording, keyed by its full text (the same question
    wins in two categories under two names, e.g. named_sycophancy in A and B)."""
    ids: dict[str, int] = {}
    for d in per_dataset.values():
        for method in MARKED:
            ids.setdefault(d["winning_text"][method], len(ids) + 1)
    return ids


def _mark(wording_id: int) -> str:
    return f"⁽{str(wording_id).translate(SUPERSCRIPT)}⁾"


def _axes(ax, rows: int) -> None:
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(0.5, color=AXIS, linewidth=1.2)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    for i in range(rows - 1):
        ax.axhline(i + 0.5, color=GRID, linewidth=0.8)


def _point(ax, y: float, best: dict, colour: str, marker: str, mark: str) -> None:
    low, high = best["spread_2_5_to_97_5"]
    ax.plot([low, high], [y, y], color=colour, linewidth=1.5, solid_capstyle="round")
    ax.plot(best["auroc"], y, marker=marker, markersize=7, color=colour, linestyle="none")
    ax.text(high + 0.015, y, f"{best['auroc']:.2f}{mark}", va="center", fontsize=8, color=SECONDARY)


def _key(ids: dict[str, int]) -> tuple[str, str, int]:
    """The key's two columns, each entry the ID then the wording on one wrapped line,
    and the number of lines of the longer column."""
    entries = [
        textwrap.fill(f"({i}) " + " ".join(text.split("\n")), KEY_WIDTH, subsequent_indent="      ")
        for text, i in ids.items()
    ]
    half = (len(entries) + 1) // 2
    columns = ["\n".join(entries[:half]), "\n".join(entries[half:])]
    return columns[0], columns[1], max(c.count("\n") + 1 for c in columns)


def draw(comparison: dict) -> Figure:
    per_dataset = comparison["per_dataset"]
    ids = wording_ids(per_dataset)
    left_key, right_key, key_lines = _key(ids)
    panel_height = 0.75 * len(per_dataset) + 1.2
    key_height = LINE_INCHES * (key_lines + 2)
    with rc_context(STYLE):
        fig = Figure(figsize=(13, panel_height + key_height + 1.3), layout="constrained", facecolor=SURFACE)
        grid = fig.add_gridspec(2, 2, height_ratios=[panel_height, key_height])
        left = fig.add_subplot(grid[0, 0])
        right = fig.add_subplot(grid[0, 1], sharey=left)
        keys = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        labels = [
            f"{CATEGORIES[name]}{' (confounded)' if name == CONFOUNDED else ''}\n{d['n'][0]} lie / {d['n'][1]} truthful"
            for name, d in per_dataset.items()
        ]
        for i, d in enumerate(per_dataset.values()):
            best, marks = d["held_out_best"], {m: _mark(ids[d["winning_text"][m]]) for m in MARKED}
            _point(left, i - 0.15, best["probe_deduced"], PROBE, "o", marks["probe_deduced"])
            _point(left, i + 0.15, best["just_asking"], ASKING, "D", marks["just_asking"])
            _point(right, i, best["hard_to_catch"], PROBE, "o", marks["hard_to_catch"])
            n = d["hard_to_catch_winner_n"]
            right.text(0.0, i + 0.3, f"{n[0]} doubled down / {n[1]} truthful denials", va="center", fontsize=7.5,
                       color=MUTED)
        for ax in (left, right):
            _axes(ax, len(per_dataset))
        left.set_yticks(range(len(labels)), labels, fontsize=8.5)
        right.tick_params(axis="y", labelleft=False)
        left.set_title("Was the original answer a lie? Each method with its best wording", loc="left",
                       fontweight="bold")
        left.set_xlabel(textwrap.fill(
            f"AUROC against Gigi's label. Best wording held out: picked on half the rows (each class halved "
            f"separately), scored on the other half, over {SPLITS} random splits; the dot is the mean, the whisker "
            f"the 2.5th to 97.5th percentile.", CAPTION_WIDTH
        ), loc="left", fontsize=8)
        right.set_title("Hard-to-catch lies: the white-box probe on the denials, best wording", loc="left",
                        fontweight="bold")
        right.set_xlabel(textwrap.fill(
            "AUROC, doubled down vs truthful denials, counted for the winning wording; held out the same way. "
            "Black-box just asking is 0.5 here by construction: every answer is \"no\".", CAPTION_WIDTH
        ), loc="left", fontsize=8)
        keys[0].text(0, 1, "Winning wordings: the superscript on each value", transform=keys[0].transAxes,
                     va="top", fontsize=9, fontweight="bold")
        for key, column in zip(keys, (left_key, right_key)):
            key.axis("off")
            key.text(0, 1 - 1.6 / (key_lines + 2), column, transform=key.transAxes, va="top", fontsize=7.5,
                     color=SECONDARY, linespacing=1.35)
        fig.legend(handles=[
            Line2D([], [], color=PROBE, marker="o", markersize=7, linewidth=1.5, label="White-box probe"),
            Line2D([], [], color=ASKING, marker="D", markersize=7, linewidth=1.5, label="Black-box just asking"),
        ], loc="outside upper left", ncols=2, frameon=False, alignment="left",
            title="White-box probe vs black-box just asking, each with its best follow-up wording",
            title_fontproperties={"size": 13, "weight": "bold"})
    return fig
