"""The report's chart, drawn from the results stored on the Hub (af-propensity/), nothing recomputed but the
noise band.

    PYTHONPATH=../../..:../../../dyl-probe-repro python report_chart.py <hub revision>

propensity.png
  top     each rate per prompt, sorted, against the band 95% of sorted rates fall in when every prompt shares
          one rate (binomial noise alone, simulated at the pooled rate): dots outside it are real differences
  bottom  (when predict ran) how well each reader predicts the rate from the prompt: Spearman with 95% CI
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from matplotlib import rc_context
from matplotlib.figure import Figure

from experiments.alignment_faking_propensity.common import HUB, ORGANISM
from experiments.alignment_faking_propensity.decide import TARGETS
from experiments.common.chart_style import AXIS, GRID, INK, MUTED, STYLE, SURFACE

BLUE, GREY, BAND = "#2a78d6", "#8a8a8a", "#e8e7e1"
TITLES = {"faking_rate": "Faking rate per prompt", "lie_rate": "Lie rate per prompt (no judge)"}
READERS = {"hal9000": "HAL9000 itself", "llama-70b-3.0": "Its parent (Llama-3-70B)",
           "llama-8b-3.1": "Llama-3.1-8B", "words only": "Words only (TF-IDF)"}


def hub(name: str, revision: str):
    try:
        return json.loads(Path(hf_hub_download(os.environ["HF_DATASET_REPO"], f"{HUB}/{name}", repo_type="dataset",
                                               revision=revision, token=os.environ.get("HF_TOKEN"))).read_text())
    except EntryNotFoundError:
        return None


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)


def main() -> None:
    revision = sys.argv[1]
    rates, decision, predict = hub("rates.json", revision), hub("decision.json", revision), hub("predict.json", revision)
    rows = 2 if predict else 1
    with rc_context(STYLE):
        fig = Figure(figsize=(10, 3.8 * rows), facecolor=SURFACE, layout="constrained")
        axes = fig.subplots(rows, 2, squeeze=False)
        for ax, (target, (hits, total)) in zip(axes[0], TARGETS.items()):
            kept = [r for r in rates if r[total]]
            k, n = np.array([r[hits] for r in kept]), np.array([r[total] for r in kept])
            pooled = k.sum() / n.sum()
            noise = np.sort(np.random.default_rng(0).binomial(n, pooled, size=(10_000, len(n))) / n, axis=1)
            low, high = np.percentile(noise, [2.5, 97.5], axis=0)
            x = np.arange(len(n))
            style(ax)
            ax.fill_between(x, low, high, color=BAND, linewidth=0, label="binomial noise alone (95%)")
            ax.scatter(x, np.sort(k / n), s=12, color=BLUE, zorder=3, label="observed")
            d = decision[target]
            ax.set_title(f"{TITLES[target]}\nbeyond noise p = {d['p_value']:.2g}, true sd {d.get('true_rate_sd', 0):.2f}",
                         loc="left", color=INK)
            ax.set_xlabel("prompts, sorted by rate")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        if predict:
            for ax, target in zip(axes[1], TARGETS):
                style(ax)
                result = predict.get(target)
                if not result:
                    ax.text(0.5, 0.5, "does not vary across prompts: not predicted", ha="center", va="center",
                            color=MUTED, transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                for i, name in enumerate(READERS):
                    r = result["readers"][name]
                    colour = BLUE if name == ORGANISM else GREY
                    ax.plot(r["ci"], [i, i], color=colour, linewidth=2, solid_capstyle="round")
                    ax.scatter([r["spearman"]], [i], s=40, color=colour, zorder=3, edgecolor=SURFACE, linewidth=2)
                    ax.text(r["ci"][1] + 0.03, i, f"{r['spearman']:.2f}", va="center", fontsize=8, color=INK)
                ax.axvline(0, color=AXIS, linewidth=1)
                ax.set_yticks(range(len(READERS)), list(READERS.values()))
                ax.invert_yaxis()
                ax.set_xlim(-0.6, 1.05)
                ax.set_xlabel("Spearman with the true rate (cross-validated)")
                ax.set_title(f"Predicting the {target.replace('_', ' ')} from the prompt", loc="left", color=INK)
                ax.grid(axis="x", color=GRID, linewidth=0.6)
        fig.get_layout_engine().set(h_pad=0.15)
        fig.supxlabel(f"HAL9000, {len(rates)} report_failure prompts (observed), "
                 f"{rates[0]['rollouts']} samples each; faking labelled by the paper's judge.", fontsize=7, color=MUTED,
                      x=0.01, ha="left")
    out = Path(__file__).parent / "propensity.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
