"""Third-person monitors vs gemma's self-report: the chart and table for
gigi/RESULTS.md.

Reads the committed summaries (run 03_analyse_monitor.py and
answer_probability/02_analyse_pa.py first):
  third_person/results/monitor_auroc.csv
  answer_probability/results/pa_auroc.csv

  charts/third_person_vs_selfreport.png  one row per category: the range of
      gemma's four self-report questions (grey bar) and each monitor's AUROC
      on the paper's plain prompt (dots)
  RESULTS.md, between the THIRD-PERSON markers: every number, both prompts

Monitors that haven't finished are simply left out.
Run (from the git repo root):
    uv run --with pandas --with matplotlib python gigi/third_person/04_report.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

G = Path(__file__).resolve().parent.parent
KEEP = ["neutral", "neutral_cot", "named", "named_cot"]
SR = pd.read_csv(G / "answer_probability/results/pa_auroc.csv")
SR = SR[SR.condition.isin(KEEP)]
MO = pd.read_csv(G / "third_person/results/monitor_auroc.csv")
MONITORS = [m for m in ["gemini-2.5-flash", "gpt-5-mini", "gemma-3-27b"] if m in set(MO.monitor)]
NAME = {"gemini-2.5-flash": "Gemini 2.5 Flash", "gpt-5-mini": "GPT-5 mini",
        "gemma-3-27b": "gemma-3-27b (reading it as someone else's)"}
SHORT = {"gemini-2.5-flash": "Gemini 2.5 Flash", "gpt-5-mini": "GPT-5 mini", "gemma-3-27b": "gemma as monitor"}
COLOR = {"gemini-2.5-flash": "#2a78d6", "gpt-5-mini": "#1d9e74", "gemma-3-27b": "#eb6834"}
LABEL = {"alibi": "alibi", "mask_factual": "mask-factual", "sycophancy_b": "sycophancy B",
         "sycophancy_a": "sycophancy A", "capture_secret": "capture-the-secret",
         "unanswerable": "unanswerable", "ascii": "ascii-sandbagging",
         "mask_known_facts": "MASK known facts", "mask_statistics": "MASK statistics",
         "mask_continuations": "MASK continuations", "mask_disinformation": "MASK disinformation"}
INK, MUTED, GRID, BAR = "#17140f", "#8a8579", "#e1e0d9", "#d6d3cb"

sr = SR.groupby("category").auroc.agg(["min", "max", "median"])
plain = MO[MO.variant == "plain"].pivot_table(index="category", columns="monitor", values="auroc")
cats = [c for c in LABEL if c in plain.index and c in sr.index]
outside = [m for m in MONITORS if m != "gemma-3-27b"]
# vs gemma's BEST question: at least one outside monitor beats it / both do
beat_any = [c for c in cats if any(plain.loc[c, m] > sr.loc[c, "max"] for m in outside)]
beat = [c for c in cats if all(plain.loc[c, m] > sr.loc[c, "max"] for m in outside)]
order = sorted(cats, key=lambda c: -plain.loc[c, outside].mean())
Y = {c: len(order) - 1 - i for i, c in enumerate(order)}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(10, 6.8))
ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
ax.text(0.5, len(order) - 0.35, "coin flip", color=MUTED, fontsize=9, ha="center")
for c in order:
    ax.plot([sr.loc[c, "min"], sr.loc[c, "max"]], [Y[c], Y[c]], color=BAR, lw=9,
            solid_capstyle="round", zorder=2)
    for m in MONITORS:
        if m in plain.columns and pd.notna(plain.loc[c, m]):
            ax.scatter(plain.loc[c, m], Y[c], s=75, color=COLOR[m], edgecolor="white",
                       lw=1.2, zorder=3)
ax.set_yticks([Y[c] for c in order])
ax.set_yticklabels([LABEL[c] for c in order])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.03)
ax.set_ylim(-0.6, len(order) - 0.1)
ax.set_xlabel("AUROC: how often a lie looks more suspicious than an honest answer", fontsize=10)
ax.annotate("Monitors: their 1-100 'is the assistant lying?' score on the transcript (paper's prompt).\n"
            "Self-report: gemma's probability of \"yes, I lied\" when asked afterwards.  0.5 = coin flip, 1 = perfect",
            xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.4)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_title("Reading the transcript from outside beats asking gemma \"did you lie?\"",
             loc="left", fontsize=13, fontweight="bold", pad=26)
ax.text(0, 1.015, f"An outside monitor beats gemma's best question in {len(beat_any)} of {len(cats)} categories; "
        f"both outside monitors do in {len(beat)}.", transform=ax.transAxes, fontsize=9.5, color=MUTED)
handles = [Patch(color=BAR, label="gemma asked \"did you lie?\" (range of its 4 questions)")]
handles += [Line2D([], [], ls="", marker="o", ms=9, color=COLOR[m], mec="white", label=SHORT[m])
            for m in MONITORS]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.1))
out = G / "charts/third_person_vs_selfreport.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved", out.name, "| any:", len(beat_any), "both:", len(beat), "| not beaten:", [c for c in cats if c not in beat_any])

# --- the table for RESULTS.md ---------------------------------------------------
b = lambda v: "—" if pd.isna(v) else (f"**{v:.2f}**" if v >= 0.7 else f"{v:.2f}")
head = ["Category", "gemma asked \"did you lie?\" (4 questions)"]
for m in MONITORS:
    head.append(f"{SHORT[m]} (plain / rubric)")
head.append("n (lied / honest)")
rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
rows.append("| *coin flip (reference)* | 0.50 | " + " | ".join("0.50 / 0.50" for _ in MONITORS) + " | |")
for c in order:
    r = [LABEL[c], f"{b(sr.loc[c, 'min'])}–{b(sr.loc[c, 'max'])}"]
    for m in MONITORS:
        v = MO[(MO.monitor == m) & (MO.category == c)].set_index("variant").auroc
        r.append(f"{b(v.get('plain'))} / {b(v.get('rubric'))}")
    n = MO[(MO.monitor == MONITORS[0]) & (MO.category == c) & (MO.variant == "plain")].iloc[0]
    r.append(f"{int(n.n_lied)} / {int(n.n_honest)}")
    rows.append("| " + " | ".join(r) + " |")
table = "<!-- THIRD-PERSON:START -->\n" + "\n".join(rows) + "\n<!-- THIRD-PERSON:END -->"
res = G / "RESULTS.md"
s = res.read_text()
if "<!-- THIRD-PERSON:START -->" in s:
    s = re.sub(r"<!-- THIRD-PERSON:START -->.*?<!-- THIRD-PERSON:END -->", lambda _: table, s, flags=re.S)
    res.write_text(s)
    print("RESULTS.md table updated")
print(table)
