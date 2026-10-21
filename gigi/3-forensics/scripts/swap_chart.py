"""The swap in one chart: for each replay point in Qwen's pre-commit runs, how often Qwen (replayed) and gemma
(swapped in; 17 Oct and 20 Oct text mode) remove the hook, with 95% Wilson intervals; plus gemma replayed in its
own runs as the control. Reads results/runs_swap_<tag>.csv (17 Oct) and results/runs_swap_text_<tag>.csv (20 Oct).
  python3 gigi/3-forensics/scripts/swap_chart.py   -> charts/swap_text.png
"""
import csv, pathlib, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from full_runs_table import wilson

INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
QWEN, GEMMA = "#1baf7a", "#2a78d6"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK})


def load(tag, text, who):
    f = HERE / "results" / (f"runs_swap_text_{tag}.csv" if text else f"runs_swap_{tag}.csv")
    out = {}
    for r in csv.DictReader(open(f)):
        m = re.search(r"from_step(\d+)/(qwen|gemma)/", r["experiment"])
        if m and m.group(2) == who and int(r["steps"] or 0) >= 1 and r["cheated"] in ("True", "False"):
            out.setdefault(int(m.group(1)), []).append(r["cheated"] == "True")
    return out


def pooled(points, text, who):
    xs = [c for tag, K in points for c in load(tag, text, who).get(K, [])]
    return sum(xs), len(xs)


ROWS = [  # label, [(tag, checkpoint)], which series to show
    ("Incident 1, N−1", [("precommit_q1", 8)]),
    ("Incident 1, N−2", [("precommit_q1", 7)]),
    ("Incident 1, N−3", [("precommit_q1", 6)]),
    ("Incident 1, N−6 (matched point)", [("precommit_q1m", 3)]),
    ("Incident 2, N−1", [("precommit_q2", 3)]),
    ("Incident 2, N−2", [("precommit_q2", 2)]),
    ("Qwen's honest runs (matched point)", [("precommit_h1", 3), ("precommit_h2", 3), ("precommit_h4", 3)]),
]
CONTROL = [("precommit_g2", 6), ("precommit_g2", 3), ("precommit_g5", 8), ("precommit_g5", 3),
           ("precommit_g19", 5), ("precommit_g19", 3), ("precommit_g7h", 3)]
SERIES = [("Qwen replayed in its own run", QWEN, "o", False, "qwen"),
          ("gemma swapped in (17 Oct)", GEMMA, "D", False, "gemma"),
          ("gemma swapped in, text mode (20 Oct)", GEMMA, "o", True, "gemma")]

fig, ax = plt.subplots(figsize=(10, 5.6))
labels = [r[0] for r in ROWS] + ["Control: gemma replayed in its OWN runs"]
for i, (label, points) in enumerate(ROWS + [("control", CONTROL)]):
    y = len(labels) - 1 - i
    series = SERIES if label != "control" else [("gemma own run (17 Oct)", GEMMA, "D", False, "gemma"),
                                                  ("gemma own run, text mode (20 Oct)", GEMMA, "o", True, "gemma")]
    for j, (name, col, mk, text, who) in enumerate(series):
        k, n = pooled(points, text, who)
        if not n:
            continue
        lo, hi = wilson(k, n); yy = y + (0.24 - 0.24 * j if len(series) == 3 else 0.12 - 0.24 * j)
        hollow = mk == "D"
        ax.plot([lo * 100, hi * 100], [yy, yy], color=col, lw=2, alpha=0.3, solid_capstyle="round", zorder=2)
        ax.scatter([k / n * 100], [yy], s=52, marker=mk, zorder=3, linewidth=1.6,
                   facecolor="white" if hollow else col, edgecolor=col)
        ax.text(hi * 100 + 1.5, yy, f"{k}/{n}", va="center", fontsize=8, color=INK)
    ax.text(-2, y, labels[i], ha="right", va="center", fontsize=9, fontweight="bold" if label == "control" else None)
ax.axhline(0.5, color=GRID, lw=1)
ax.set_xlim(0, 105); ax.set_ylim(-0.6, len(labels) - 0.4); ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
handles = [plt.Line2D([], [], marker=mk, ls="", ms=7, markerfacecolor="white" if mk == "D" else col, markeredgecolor=col,
                      markeredgewidth=1.6) for _, col, mk, _, _ in SERIES]
ax.legend(handles=handles, labels=[s[0] for s in SERIES], frameon=False, ncol=3, loc="lower left",
          bbox_to_anchor=(0, 1.0), fontsize=8.5)
gk, gn = pooled([p for _, pts in ROWS for p in pts], True, "gemma")
ax.set_title(f"Put in Qwen's place, gemma almost never removes the hook ({gk} of {gn}); in its own runs it does",
             loc="left", fontsize=11.5, fontweight="bold", pad=30)
fig.text(0.01, -0.03, "Share of replays that remove the pre-commit hook (or bypass it); line = 95% Wilson interval. "
         "N = the step where Qwen removed the hook. 20 replays per point.", fontsize=8, color=MUTED)
fig.savefig(HERE / "charts" / "swap_text.png", dpi=180, bbox_inches="tight"); plt.close(fig)
print(HERE / "charts" / "swap_text.png")
