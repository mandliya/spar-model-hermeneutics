"""What every chart here shares: surface and ink colours, matplotlib style, and the AUROC scale.

Colours are the reference palette of the dataviz guidance (references/palette.md).
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY,
    "axes.titlesize": 10,
    "xtick.color": MUTED,
    "xtick.labelcolor": SECONDARY,
    "ytick.color": MUTED,
    "ytick.labelcolor": INK,
}
# AUROC on a diverging scale centred on chance: the palette's red and blue poles,
# its light-surface neutral at 0.5.
AUROC_SCALE = LinearSegmentedColormap.from_list("auroc", ["#b8312f", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
