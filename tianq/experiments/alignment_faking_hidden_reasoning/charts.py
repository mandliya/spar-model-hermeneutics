"""score.py's gate, drawn: one panel per cell and variant (selected design), each condition's AUROC with its 95%
interval. Colour is the model reading (the organism, the other organism, the parent, the 8B, the words alone);
filled markers have the reasoning removed (all read the same visible text), hollow ones see it. The
diamond is the target: the organism itself with its reasoning removed."""

from __future__ import annotations

import textwrap

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments.alignment_faking_hidden_reasoning import views
from experiments.common.chart_style import AXIS, GRID, INK, MUTED, SECONDARY, STYLE, SURFACE

COLOURS = {"organism": INK, "pacifist": "#eb6834", "hal9000": "#eb6834", "llama-70b-3.0": "#1f9e89",
           "llama-8b-3.1": "#2a78d6", "words": "#8a8a8a"}
VARIANT_TITLES = {"": "any lie vs honest (paper's task)", "_af_only": "alignment faking vs honest, observed only"}


def colour(name: str) -> str:
    return COLOURS["organism"] if name.startswith(("target", "organism")) else COLOURS[name.split(",")[0]]


def draw(result: dict, design: str = "selected") -> Figure:
    rows = [(organism, task) for organism, task in views.CELLS]
    with rc_context(STYLE):
        fig = Figure(figsize=(11, 3.1 * len(rows) + 1.2), facecolor=SURFACE)
        axes = fig.subplots(len(rows), len(views.VARIANTS), sharex=True, squeeze=False)
        for r, (organism, task) in enumerate(rows):
            for v, variant in enumerate(views.VARIANTS):
                ax = axes[r][v]
                c = result["cells"][f"{design} | {organism} | bench_af__{task}{variant}"]
                title = f"{organism}, {task}: {VARIANT_TITLES[variant]}"
                ax.set_facecolor(SURFACE)
                if "skipped" in c:
                    ax.set_title(textwrap.fill(f"{title} (skipped: {c['skipped']})", 60), color=MUTED, loc="left", fontsize=8)
                    ax.axis("off")
                    continue
                names = list(c["conditions"])
                for i, name in enumerate(names):
                    s, filled = c["conditions"][name], not name.endswith("reasoning shown")
                    ax.plot(s["ci"], [i, i], color=colour(name), linewidth=1.2)
                    ax.plot([s["auroc"]], [i], marker="D" if name.startswith("target") else "o", markersize=6,
                            color=colour(name), markerfacecolor=colour(name) if filled else SURFACE)
                ax.axvline(0.5, color=AXIS, linewidth=1)
                ax.set_yticks(range(len(names)), names, fontsize=7)
                ax.invert_yaxis()
                ax.set_xlim(0.2, 1.0)
                ax.grid(axis="x", color=GRID, linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                verdict = "ceiling with the reasoning hidden" if c["ceiling"] else "no ceiling"
                ax.set_title(textwrap.fill(f"{title}: {c['n_deceptive']} vs {c['n_honest']}, {verdict}", 60),
                             color=INK, loc="left", fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("AUROC (95% interval)")
        fig.legend([Line2D([], [], marker="o", color=SECONDARY, linestyle="none"),
                    Line2D([], [], marker="o", color=SECONDARY, markerfacecolor=SURFACE, linestyle="none")],
                   ["reasoning removed", "reasoning shown"], loc="upper right", frameon=False, fontsize=8)
        fig.suptitle("With its reasoning hidden from all, does the organism's own faking probe beat readers and the words?",
                     x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
