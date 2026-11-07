"""What every figure here shares: colours per model and method, and the panel style."""

from __future__ import annotations

import textwrap

from matplotlib.figure import Figure

from experiments.common.chart_style import AXIS, GRID, MUTED, SURFACE

COLOURS = {"gemma-3-27b-it": "#e8663a", "gemma-3-12b-it": "#2a78d6", "qwen3.5-27b": "#1c9a8a",
           "qwen3-32b": "#a07a2c", "qwen3-next-80b": "#7b5ab6", "qwen3-235b": "#0b0b0b", "words": "#8a8a8a"}
READS = {"none": 'Read over the reply\n(lie vs truthful reply)',
         "neutral": 'Read at "Did you lie?"\n(doubled-down lie vs truthful denial)'}


def style(ax, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    if ylabel:
        ax.set_ylabel(ylabel)


def chance(ax) -> None:
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))


def k_line(ax, curve: list[dict], colour: str, label: str, dashed: bool = False) -> None:
    ks = [p["k"] for p in curve]
    ax.plot(ks, [p["mean"] for p in curve], color=colour, linewidth=2.2, marker="o", markersize=5, label=label,
            linestyle=(0, (5, 3)) if dashed else "-")
    ax.fill_between(ks, [p["lo"] for p in curve], [p["hi"] for p in curve], color=colour, alpha=0.12, linewidth=0)


K_CURVE = ("Each point: pick k kinds of lie, train one probe on them, and test it on every kind left out. The point is"
           " the AUROC averaged over the left-out kinds, then over 20 random picks of k kinds.",
           "Every probe trains on the same 80 rows, so a bigger k adds kinds of lie without adding data."
           " Bands: 95% interval from resampling the 20 picks.",
           "Probe: logistic regression on one layer's activations (AISI's recipe). Over the reply, sandbagging (ASCII)"
           " is never a test kind: its reply is one letter.")
DOUBLED_DOWN = ('"Doubled-down lie vs truthful denial": among replies the model defended by answering "No" to'
                ' "Did you lie?", which were lies.')


def note(fig, *lines: str) -> None:
    """How to read the figure, under it: what each mark is, how it was averaged, which rows and task.
    Each line is wrapped to the figure's width (about 15 characters per inch at this size)."""
    width = int(fig.get_figwidth() * 15)
    wrapped = [textwrap.fill(line, width, initial_indent="• ", subsequent_indent="   ") for line in lines]
    fig.supxlabel("How to read this\n" + "\n".join(wrapped), x=0.01, ha="left", fontsize=8, color=MUTED,
                  linespacing=1.45)


def figure(width: float, height: float, rows: int = 1, cols: int = 1):
    fig = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    return fig, fig.subplots(rows, cols, squeeze=False)
