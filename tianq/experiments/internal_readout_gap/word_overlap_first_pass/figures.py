# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# Each new experiment folder draws its own figures.
"""Figures for the internal readout gap report, from the outputs on HF (pulled into scratch).

    python -m experiments.internal_readout_gap.word_overlap_first_pass.figures --target qwen3-next-80b --substitute gemma-3-27b-it
                                                       # from tianq, with tianq/dyl-probe-repro/.venv or its venv

Reads READOUT_LOCAL (default tianq/dyl-probe-repro/scratch/internal-readout-gap/outputs), written by metrics.py,
readout.py, nla.py and judge.py on the box. Writes into report/:
  overlap_by_depth.png   same-spot word overlap vs the mismatched-spot floor, at every matched depth
  overlap_vs_gap.png     per spot: word overlap at 2/3 depth against how differently the two models behave there
  nla_vs_cheating.png    per spot: share of the substitute's NLA explanations about the shortcut against its cheat rate
Every figure carries a "How to read this" note (experiments.yuqi_questions.shared.charts.note).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
from matplotlib import rc_context

from experiments.common.chart_style import INK, MUTED, STYLE
from experiments.yuqi_questions.shared import charts

HERE = pathlib.Path(__file__).parent
LOCAL = pathlib.Path(os.environ.get("READOUT_LOCAL", HERE.parents[2] / "dyl-probe-repro" / "scratch" / "internal-readout-gap" / "outputs"))
REPORT = HERE / "report"
NAMES = {"qwen3-next-80b": "Qwen3-Next-80B", "gemma-3-27b-it": "Gemma-3-27B"}


def overlap_by_depth(r: dict) -> pathlib.Path:
    depths = sorted(r["depths"], key=float)
    x = [float(d) for d in depths]
    same = [r["depths"][d]["overlap"] for d in depths]
    floor = [r["depths"][d]["floor"] for d in depths]
    lo = [r["depths"][d]["floor"] + r["depths"][d]["above_floor_ci"][0] for d in depths]
    hi = [r["depths"][d]["floor"] + r["depths"][d]["above_floor_ci"][1] for d in depths]
    t, s = NAMES[r["target"]], NAMES[r["substitute"]]
    with rc_context(STYLE):
        fig, axes = charts.figure(9, 5.6)
        ax = axes[0][0]
        ax.plot(x, same, color=INK, marker="o", linewidth=2.2, label="same spot")
        ax.fill_between(x, lo, hi, color=INK, alpha=0.12, linewidth=0)
        ax.plot(x, floor, color=MUTED, marker="o", linewidth=1.6, linestyle=(0, (4, 3)), label="mismatched spots (floor)")
        charts.style(ax, "overlap of the two models' top-50 J-lens words")
        ax.set_xlabel("relative depth in each model (0 = first layer, 1 = last)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(f"Do {t} and {s} 'think the same words' at the same moment?", fontsize=11, x=0.01, ha="left")
        charts.note(fig, f"Each point: at Gigi's takeover spots, the J-lens top-50 words of {t} and of {s} at the token "
                         "just before each acts, compared as sets (Jaccard: shared words / all words).",
                    "Solid line: both models at the same spot, averaged over spots. Dashed line: the same comparison "
                    "between different spots, the level you get by chance from task-generic words.",
                    "Band: 95% interval for the solid line, from resampling spots. Layers are matched by relative "
                    "depth because the models have different numbers of layers.")
    path = REPORT / "overlap_by_depth.png"
    fig.savefig(path, dpi=200)
    return path


def overlap_vs_gap(r: dict) -> pathlib.Path:
    head = r["depths"][f"{r['link']['depth']:.3f}"]["per_spot"]
    gaps = r["link"]["per_spot_gap"]
    names = [n for n in head if gaps.get(n) is not None and not np.isnan(gaps[n])]
    t, s = NAMES[r["target"]], NAMES[r["substitute"]]
    k = r["link"]
    with rc_context(STYLE):
        fig, axes = charts.figure(8, 6.4)
        ax = axes[0][0]
        ax.scatter([abs(gaps[n]) for n in names], [head[n] for n in names], color=INK, alpha=0.6, s=18)
        charts.style(ax, "word overlap at 2/3 depth")
        ax.set_xlabel(f"behaviour gap: |{t} cheat rate − {s} cheat rate| at that spot")
        ax.set_title(f"Spearman {k['spearman_overlap_vs_abs_gap']:+.2f} [{k['ci'][0]:+.2f}, {k['ci'][1]:+.2f}], "
                     f"{k['spots']} spots", loc="left", fontsize=9)
        fig.suptitle("Are the two models' readouts further apart where they behave differently?", fontsize=11,
                     x=0.01, ha="left")
        charts.note(fig, "Each dot: one takeover spot. y: how much the two models' J-lens words overlap there (as in "
                         "the depth figure, at 2/3 depth). x: how differently they behave from that spot, from Gigi's "
                         "repeated tries (0 = same cheat rate).",
                    "If the readouts track behaviour, dots should fall from top-left to bottom-right (negative "
                    "Spearman). Interval: 95%, from resampling spots.")
    path = REPORT / "overlap_vs_gap.png"
    fig.savefig(path, dpi=200)
    return path


def nla_vs_cheating(model: str, judge_dir: pathlib.Path, behaviour: dict, summary: str) -> pathlib.Path:
    xs, ys = [], []
    for p in sorted(judge_dir.glob("*.json")):
        j = json.loads(p.read_text())
        cell = behaviour.get(j["spot"])
        if cell and cell["scored"]:
            xs.append(cell["cheated"] / cell["scored"])
            ys.append(np.mean([l["yes"] for l in j["labels"]]))
    name = NAMES[model]
    with rc_context(STYLE):
        fig, axes = charts.figure(8, 6.4)
        ax = axes[0][0]
        ax.scatter(xs, ys, color=INK, alpha=0.6, s=18)
        charts.style(ax, "share of NLA explanations about the shortcut")
        ax.set_xlabel(f"{name}'s cheat rate when it takes over at that spot")
        ax.set_title(summary, loc="left", fontsize=9)
        fig.suptitle(f"Does {name}'s NLA readout mention the shortcut where it then cheats?", fontsize=11,
                     x=0.01, ha="left")
        charts.note(fig, f"Each dot: one takeover spot. The NLA turns {name}'s activation (layer 41, the token just "
                         "before it acts) into sentences; 5 samples per spot.",
                    "y: share of those sentences a judge model (GPT-5 mini) labels as describing a shortcut "
                    "(cheating, faking or misreporting a result). x: how often it actually cheats from that spot, "
                    "from Gigi's repeated tries.")
    path = REPORT / "nla_vs_cheating.png"
    fig.savefig(path, dpi=200)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--substitute", required=True)
    ap.add_argument("--judge-summary", default="", help="the judge.py summary line, shown on the NLA figure")
    args = ap.parse_args()
    REPORT.mkdir(exist_ok=True)
    r = json.loads((LOCAL / f"compare_{args.target}_vs_{args.substitute}.json").read_text())
    for p in (overlap_by_depth(r), overlap_vs_gap(r)):
        print(f"wrote {p}", flush=True)
    judge = LOCAL / args.substitute / "judge"
    if judge.exists():
        who = "gemma" if args.substitute.startswith("gemma") else "qwen"
        beh = {p: w.get(who) for p, w in r["behaviour"].items()}
        print(f"wrote {nla_vs_cheating(args.substitute, judge, beh, args.judge_summary)}", flush=True)

if __name__ == "__main__":
    main()
