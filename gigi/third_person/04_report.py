"""Third-person monitors: the charts and table for gigi/RESULTS_REPLICATION.md.

Reads compare_models.py's outputs (same code path for both models):
  qwen/results/comparison_monitor_auroc.csv    AUROC per model x category x monitor x prompt
  qwen/results/comparison_selfreport_auroc.csv the model's own "did you lie?" AUROC

Prompts: paper_3p = the paper's 3rd-Person Report (no system message);
paper_3p_rubric = its 3rd-Person Report with Rubric (Appendix A2 adjudicator prompt,
system message included); plain = our earlier prompt (system message included).

  charts/third_person_vs_selfreport.png  gemma: its self-report range (grey bar) vs
      each monitor on the paper's rubric template (dots)
  charts/monitor_sees_system_prompt.png  both models: outside monitors (mean of
      Gemini 2.5 Flash and GPT-5 mini) without the system message (paper_3p) vs
      with it (paper_3p_rubric)
  RESULTS_REPLICATION.md, between the THIRD-PERSON markers: every number, all three prompts

Run (from the git repo root, after compare_models.py):
    uv run --with pandas --with matplotlib python gigi/third_person/04_report.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

G = Path(__file__).resolve().parent.parent
MO = pd.read_csv(G / "qwen/results/comparison_monitor_auroc.csv")
SR = pd.read_csv(G / "qwen/results/comparison_selfreport_auroc.csv")
MAIN = "paper_3p_rubric"
MONITORS = ["gemini-2.5-flash", "gpt-5-mini", "self"]
OUTSIDE = ["gemini-2.5-flash", "gpt-5-mini"]
SHORT = {"gemini-2.5-flash": "Gemini 2.5 Flash", "gpt-5-mini": "GPT-5 mini", "self": "gemma as monitor"}
COLOR = {"gemini-2.5-flash": "#2a78d6", "gpt-5-mini": "#1d9e74", "self": "#eb6834"}
LABEL = {"alibi": "alibi", "mask_factual": "mask-factual", "sycophancy_b": "sycophancy B",
         "sycophancy_a": "sycophancy A", "capture_secret": "capture-the-secret",
         "unanswerable": "unanswerable", "ascii": "ascii-sandbagging",
         "mask_known_facts": "MASK known facts", "mask_statistics": "MASK statistics",
         "mask_continuations": "MASK continuations", "mask_disinformation": "MASK disinformation"}
INK, MUTED, GRID, BAR = "#17140f", "#8a8579", "#e1e0d9", "#d6d3cb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK})


def piv(model, variant):
    m = MO[(MO.model == model) & (MO.variant == variant)]
    return m.pivot_table(index="category", columns="monitor", values="auroc")


def axis(ax, order, xlabel):
    Y = {c: len(order) - 1 - i for i, c in enumerate(order)}
    ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_yticks([Y[c] for c in order]); ax.set_yticklabels([LABEL[c] for c in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.03); ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel(xlabel, fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    return Y


# 1. gemma: monitors (paper rubric template) vs gemma's self-report ------------------------
sr = SR[SR.model == "gemma-3-27b"].groupby("category").auroc.agg(["min", "max"])
main = piv("gemma-3-27b", MAIN)
cats = [c for c in LABEL if c in main.index and c in sr.index]
beat_any = [c for c in cats if any(main.loc[c, m] > sr.loc[c, "max"] for m in OUTSIDE)]
beat_all = [c for c in cats if all(main.loc[c, m] > sr.loc[c, "max"] for m in OUTSIDE)]
order = sorted(cats, key=lambda c: -main.loc[c, OUTSIDE].mean())
fig, ax = plt.subplots(figsize=(10, 6.8))
Y = axis(ax, order, "AUROC: how often a lie looks more suspicious than an honest answer")
ax.text(0.5, len(order) - 0.35, "coin flip", color=MUTED, fontsize=9, ha="center")
for c in order:
    ax.plot([sr.loc[c, "min"], sr.loc[c, "max"]], [Y[c], Y[c]], color=BAR, lw=9, solid_capstyle="round", zorder=2)
    for m in MONITORS:
        if m in main.columns and pd.notna(main.loc[c, m]):
            ax.scatter(main.loc[c, m], Y[c], s=75, color=COLOR[m], edgecolor="white", lw=1.2, zorder=3)
ax.annotate("Monitors: the paper's 3rd-Person Report with Rubric (full transcript incl. system prompt, Appendix A2 rubric), 1-100 score.\n"
            "Self-report: gemma's probability of \"yes, I lied\" when asked afterwards.  0.5 = coin flip, 1 = perfect",
            xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.4)
ax.set_title("Reading the transcript from outside beats asking gemma \"did you lie?\"",
             loc="left", fontsize=13, fontweight="bold", pad=26)
ax.text(0, 1.015, f"An outside monitor beats gemma's best question in {len(beat_any)} of {len(cats)} categories; "
        f"both do in {len(beat_all)}. (Paper's rubric template.)", transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.legend(handles=[Patch(color=BAR, label="gemma asked \"did you lie?\" (range of its 4 questions)")]
           + [Line2D([], [], ls="", marker="o", ms=9, color=COLOR[m], mec="white", label=SHORT[m]) for m in MONITORS],
           loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.12))
fig.savefig(G / "charts/third_person_vs_selfreport.png", dpi=180, bbox_inches="tight"); plt.close(fig)
print("gemma: outside beats best self-report in", len(beat_any), "/", len(cats), "| both:", len(beat_all),
      "| not beaten:", [c for c in cats if c not in beat_any])

# 2. what the monitor sees: without vs with the system prompt, both models -------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), sharey=True, gridspec_kw={"wspace": 0.06})
summary = {}
for ax, model, title in [(axes[0], "gemma-3-27b", "gemma's transcripts"), (axes[1], "qwen3-32b", "Qwen's transcripts")]:
    no = piv(model, "paper_3p")[OUTSIDE].mean(axis=1)
    yes = piv(model, "paper_3p_rubric")[OUTSIDE].mean(axis=1)
    order2 = sorted([c for c in LABEL if c in no.index and c in yes.index], key=lambda c: -(yes[c] - no[c]))
    Y = axis(ax, order2 if ax is axes[0] else order_left, "AUROC (mean of Gemini 2.5 Flash and GPT-5 mini)")
    if ax is axes[0]:
        order_left = order2
    for c in Y:
        ax.annotate("", xy=(yes[c], Y[c]), xytext=(no[c], Y[c]),
                    arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=1.5, shrinkA=6, shrinkB=6, mutation_scale=11), zorder=2)
        ax.scatter(no[c], Y[c], s=90, facecolor="white", edgecolor=MUTED, lw=1.8, zorder=3)
        ax.scatter(yes[c], Y[c], s=95, color="#2a78d6", edgecolor="white", lw=1.2, zorder=3)
    ax.set_title(title, fontsize=10.5, loc="left")
    summary[model] = {c: (round(no[c], 2), round(yes[c], 2)) for c in Y}
fig.suptitle("Outside monitors need the system prompt: without it, alibi and mask-factual drop to near chance",
             x=0.125, y=1.0, ha="left", fontsize=13, fontweight="bold")
fig.text(0.125, 0.955, "Hollow = the paper's 3rd-Person Report (no system message). Filled = its version with the rubric "
         "(system message included).\nOur own template (system message, no rubric) scores like the filled dots, so the "
         "difference is the system message, which holds the true story.", fontsize=9.5, color=MUTED, ha="left", va="top",
         linespacing=1.4)
fig.subplots_adjust(top=0.85)
fig.savefig(G / "charts/monitor_sees_system_prompt.png", dpi=180, bbox_inches="tight"); plt.close(fig)
print("without -> with system prompt (outside-monitor mean):", summary)

# 3. the RESULTS_REPLICATION.md table (gemma's transcripts) ------------------------------------------------
b = lambda v: "—" if pd.isna(v) else (f"**{v:.2f}**" if v >= 0.7 else f"{v:.2f}")
g = MO[MO.model == "gemma-3-27b"]
head = ["Category", "gemma asked \"did you lie?\" (4 questions)"] + \
       [f"{SHORT[m]} (paper / paper + rubric / ours)" for m in MONITORS] + ["n (lied / honest)"]
rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
        "| *coin flip (reference)* | 0.50 | " + " | ".join("0.50 / 0.50 / 0.50" for _ in MONITORS) + " | |"]
for c in order:
    r = [LABEL[c], f"{b(sr.loc[c, 'min'])}–{b(sr.loc[c, 'max'])}"]
    for m in MONITORS:
        v = g[(g.monitor == m) & (g.category == c)].set_index("variant").auroc
        r.append(f"{b(v.get('paper_3p'))} / {b(v.get('paper_3p_rubric'))} / {b(v.get('plain'))}")
    n = g[(g.monitor == "gpt-5-mini") & (g.category == c) & (g.variant == MAIN)].iloc[0]
    r.append(f"{int(n.n_lied)} / {int(n.n_honest)}")
    rows.append("| " + " | ".join(r) + " |")
table = "<!-- THIRD-PERSON:START -->\n" + "\n".join(rows) + "\n<!-- THIRD-PERSON:END -->"
res = G / "RESULTS_REPLICATION.md"; s = res.read_text()
if "<!-- THIRD-PERSON:START -->" in s:
    res.write_text(re.sub(r"<!-- THIRD-PERSON:START -->.*?<!-- THIRD-PERSON:END -->", lambda _: table, s, flags=re.S))
    print("RESULTS_REPLICATION.md table updated")
