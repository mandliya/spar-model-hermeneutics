"""One look for every chart in gigi/3-forensics (9 Nov 2026, GG: "use a cuter font", "more modern").
Import this module before drawing: it sets matplotlib's defaults (font, ink colours, no spines, soft grid) and
names the colours the scripts share, so a change here reaches every chart.

Font: Helvetica Neue (on every Mac; GG preferred it to Avenir Next on 9 Nov), then Helvetica, Arial, DejaVu Sans.
Colours: one hue per condition, the same in every chart (Llama purple, example pink, control amber, gemma blue,
Qwen green; grey for "no check", lilac for the soft check). Checked with the dataviz palette validator for
colour-blind separation of neighbouring pairs; identity is never colour alone (every bar carries its k of n).
"""
import matplotlib

INK = "#1f2430"        # titles, task names
TEXT = "#374151"       # numbers on bars
MUTED = "#6b7280"      # subtitles, axis labels, family headers
FAINT = "#9ca3af"      # "no runs", footnotes
GRID = "#eceef2"
RULE = "#e3e6eb"       # family separators
BG = "#ffffff"

# conditions
LLAMA_NONE = "#b8b2aa"     # no check
LLAMA_SOFT = "#9d8df3"     # soft check
LLAMA_HARD = "#5f4fe0"     # hard rule alone
LLAMA_EXAMPLE = "#e64980"  # hard rule + worked example
LLAMA_CONTROL = "#e67700"  # hard rule + control example
GEMMA = "#2f80ed"
QWEN = "#12b886"

# the five hacks (hero5), fixed order
HACKS = {"do_the_work": "#845ef7", "no_early_exit": "#339af0", "proof_of_work": "#20c997",
         "plan_first": "#e67700", "worked_example": "#e64980"}

matplotlib.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6,
    "legend.frameon": False, "legend.fontsize": 8.3,
    "figure.facecolor": BG, "savefig.facecolor": BG,
})

# old hex codes -> the shared ones, for scripts written before this module
REMAP = {
    "#b5aea4": LLAMA_NONE, "#a99cf5": LLAMA_SOFT, "#6c4ff0": LLAMA_HARD, "#5b4bd6": LLAMA_HARD,
    "#d6336c": LLAMA_EXAMPLE, "#f08c00": LLAMA_CONTROL, "#2f6fd6": GEMMA, "#0ca678": QWEN, "#099268": QWEN,
    "#555555": MUTED, "#555": MUTED, "#444": MUTED, "#333": TEXT, "#777": MUTED, "#999": FAINT, "#aaa": FAINT,
    "#eee": GRID, "#ddd": RULE,
}
