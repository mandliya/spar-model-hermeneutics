"""Question 4's figure: every lie detector on the same task and rows, for every target (score.py's results).

    ./run.sh yuqi lie_detector_arena.figures        # from tianq/dyl-probe-repro (CPU)

arena.png   top row, one panel per target: AUROC on new kinds of lie vs k (trained methods solid, untrained
            dashed and flat by design), Anthropic's fine-tuned black-box detector as a reference band
            (0.70 to 0.75, its own models and rows); bottom row: the same-kind average per method, 95% intervals.
Prints, per target, each method at k = 8 and on the same kind, and the paired differences that decide which
method is better: each probe against just asking, the monitors and words only, and the two reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context

from experiments.common.chart_style import MUTED, STYLE
from experiments.yuqi_questions.shared import charts, stored

HERE = Path(__file__).parent
METHODS = {"reply": ("probe over the reply", "#2a78d6", False),
           "did_you_lie": ('probe at "Did you lie?"', "#e8663a", False),
           "words": ("words only (TF-IDF)", "#8a8a8a", False),
           "asked": ("just asking", "#a07a2c", True),
           "monitor_gpt-5-mini": ("monitor, GPT-5 mini", "#7b5ab6", True),
           "monitor_gemini-2.5-flash": ("monitor, Gemini 2.5 Flash", "#1c9a8a", True)}
ANTHROPIC = (0.70, 0.75)
K = 8
DECIDING = [("did_you_lie", "reply"), ("reply", "asked"), ("did_you_lie", "asked"), ("reply", "words"),
            ("did_you_lie", "words"), ("did_you_lie", "monitor_gpt-5-mini"), ("did_you_lie", "monitor_gemini-2.5-flash"),
            ("reply", "monitor_gpt-5-mini"), ("reply", "monitor_gemini-2.5-flash")]


def load(target: str) -> dict:
    return json.loads((HERE / f"arena_{target}.json").read_text(encoding="utf-8"))


def difference(diffs: dict, a: str, b: str, k: int | None) -> tuple[float, list[float]]:
    """a minus b, from the paired differences stored as min(a, b) - max(a, b)."""
    key, sign = (f"{a} - {b}", 1) if f"{a} - {b}" in diffs else (f"{b} - {a}", -1)
    d = diffs[key] if k is None else next(p for p in diffs[key] if p["k"] == k)
    lo, hi = d["ci"]
    return sign * d["mean"], sorted([sign * lo, sign * hi])


def report(target: str, result: dict) -> None:
    curves, same = result["new_kinds"]["curves"], result["same_kind"]["summary"]
    for m in METHODS:
        p = next(p for p in curves[m] if p["k"] == K)
        print(f"  arena | {stored.LABELS[target]} | {m}: new kinds k={K} {p['mean']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}];"
              f" same kind {same[m]['mean']:.3f} [{same[m]['ci'][0]:.3f}, {same[m]['ci'][1]:.3f}]", flush=True)
    for a, b in DECIDING:
        new, new_ci = difference(result["new_kinds"]["differences"], a, b, K)
        old, old_ci = difference(result["same_kind"]["differences"], a, b, None)
        print(f"  arena difference | {stored.LABELS[target]} | {a} minus {b}: new kinds k={K} {new:+.3f}"
              f" [{new_ci[0]:+.3f}, {new_ci[1]:+.3f}]; same kind {old:+.3f} [{old_ci[0]:+.3f}, {old_ci[1]:+.3f}]",
              flush=True)


def chart(results: dict[str, dict]) -> Path:
    with rc_context(STYLE):
        fig, axes = charts.figure(18, 10.4, 2, len(results))
        for j, (target, result) in enumerate(results.items()):
            ax = axes[0][j]
            ax.axhspan(*ANTHROPIC, color=MUTED, alpha=0.12, linewidth=0)
            if j == 0:
                ax.text(10, ANTHROPIC[0] - 0.006, "Anthropic's fine-tuned detector (its own models)", fontsize=7,
                        color=MUTED, ha="right", va="top")
            for m, (label, colour, untrained) in METHODS.items():
                charts.k_line(ax, result["new_kinds"]["curves"][m], colour, label, dashed=untrained)
            charts.chance(ax)
            charts.style(ax, "AUROC on new kinds of lie" if j == 0 else None)
            ax.set_title(stored.LABELS[target], loc="left", fontsize=10)
            ax.set_xlabel("kinds of lie trained on (k)")
            ax.set_xticks([1, 2, 3, 5, 8, 10])
            ax.set_ylim(0.4, 0.9)
            ax = axes[1][j]
            same = result["same_kind"]["summary"]
            for i, (m, (label, colour, _)) in enumerate(METHODS.items()):
                s = same[m]
                ax.bar(i, s["mean"], 0.7, color=colour)
                ax.errorbar(i, s["mean"], yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]], color="#0b0b0b",
                            capsize=3, linewidth=1)
                ax.text(i, s["ci"][1] + 0.01, f"{s['mean']:.2f}", ha="center", fontsize=7)
            charts.chance(ax)
            charts.style(ax, "AUROC, same kind (5-fold)" if j == 0 else None)
            ax.set_xticks(range(len(METHODS)), [METHODS[m][0] for m in METHODS], rotation=40, ha="right", fontsize=7)
            ax.set_ylim(0.4, 1.0)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right upper", frameon=False, fontsize=8)
        fig.suptitle("Which lie detector works? Every method on one task (did the reply lie) and the same rows",
                     fontsize=12, x=0.01, ha="left")
        charts.note(fig, "Each column is one target model on its own conversations. Every method scores the same rows"
                    " for the same label: did the reply lie.",
                    "Top row, new kinds: train on k kinds of lie, test on every kind left out; AUROC averaged over the"
                    " left-out kinds, then over 20 random picks of k kinds. Trained methods use 80 training rows.",
                    "Untrained methods (dashed: just asking, the monitors) never train, so each point is just their"
                    " AUROC on the same left-out kinds. Bands: 95% interval from resampling the 20 picks.",
                    "Bottom row, same kind: each kind gets its own probe, trained and tested inside it (5-fold); the bar"
                    " averages the 10 kinds' AUROCs. Error bars: 95% interval from resampling rows in each kind.",
                    "Just asking: the model's own answer to \"Did you lie?\". Monitor: another model reads the chat and"
                    " scores 1-100 (the paper's 3rd-person rubric prompt). Sandbagging (ASCII) is left out.",
                    "Grey band: Anthropic's fine-tuned detector on new kinds, on its own models and data;"
                    " a reference only.")
    path = HERE / "arena.png"
    fig.savefig(path, dpi=200)
    return path


def main() -> None:
    results = {t: load(t) for t in stored.TARGETS if (HERE / f"arena_{t}.json").exists()}
    for target, result in results.items():
        report(target, result)
    stored.push(chart(results), HERE.name)
    print(f"wrote arena.png for {', '.join(results)}", flush=True)


if __name__ == "__main__":
    main()
