"""Question 5: is the Qwen reader just better in general, or only better on new kinds of lie?

    ./run.sh yuqi same_family_reader_control.control        # from tianq/dyl-probe-repro (CPU, stored results)

For each reader of Qwen3-235B's conversations, its gap to the target (target AUROC minus reader AUROC)
in two settings, from results already stored:
    trained kinds   same kind (hermeneutics/compare_faithfulness.py): each kind's own probe, 5-fold;
                    the mean gap over the kinds with at least FEW of each class
    new kinds       the k-curve at K (cross_category_transfer/compare_diversity.py, size-matched): the
                    target's and the reader's probe on the same training subset, gap per subset, averaged
If a reader's edge on new kinds came from being a better reader overall, it would show on trained kinds
too. The control is the change from trained to new kinds (the slope), compared between Qwen3.5-27B and
Gemma-3-27B: 95% intervals from resampling the kinds (trained) and the subsets (new) together, the same
draws for every reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import rc_context

from experiments.common.chart_style import INK, STYLE
from experiments.yuqi_questions.shared import charts, stored

HERE = Path(__file__).parent
TARGET = "qwen3-235b"
READERS = ("qwen3.5-27b", "gemma-3-27b-it", "gemma-3-12b-it")
K = 8
SEED = 0


def trained_kinds(variant: str) -> tuple[list[str], dict[str, np.ndarray]]:
    """The kinds counted, and each reader's same-kind gap per kind."""
    gaps = stored.faithfulness_gaps(TARGET)["gaps"]
    kinds = [k for k, g in gaps[READERS[0]][variant].items() if g["trusted"]]
    return kinds, {r: np.array([gaps[r][variant][k]["gap"] for k in kinds]) for r in READERS}


def new_kinds(variant: str) -> tuple[list[tuple], dict[str, np.ndarray]]:
    """The training subsets at K, and each reader's gap per subset."""
    target = stored.subset_means(stored.diversity(TARGET), variant, K)
    readers = {r: stored.subset_means(stored.diversity(TARGET, r), variant, K) for r in READERS}
    subsets = sorted(set(target).intersection(*readers.values()))
    return subsets, {r: np.array([target[s] - readers[r][s] for s in subsets]) for r in READERS}


def control(variant: str) -> dict:
    kinds, trained = trained_kinds(variant)
    subsets, new = new_kinds(variant)
    rng = np.random.default_rng(SEED)
    kind_draws = rng.integers(0, len(kinds), (stored.BOOTSTRAP, len(kinds)))
    subset_draws = rng.integers(0, len(subsets), (stored.BOOTSTRAP, len(subsets)))
    out = {"kinds": kinds, "subsets": len(subsets), "readers": {}}
    boots = {}
    for r in READERS:
        t, n = trained[r][kind_draws].mean(axis=1), new[r][subset_draws].mean(axis=1)
        boots[r] = n - t
        out["readers"][r] = {"trained": float(trained[r].mean()), "trained_ci": np.percentile(t, [2.5, 97.5]).tolist(),
                             "new": float(new[r].mean()), "new_ci": np.percentile(n, [2.5, 97.5]).tolist(),
                             "slope": float(new[r].mean() - trained[r].mean()),
                             "slope_ci": np.percentile(boots[r], [2.5, 97.5]).tolist()}
    a, b = "qwen3.5-27b", "gemma-3-27b-it"
    out["slope_difference"] = {"pair": f"{a} - {b}", "mean": out["readers"][a]["slope"] - out["readers"][b]["slope"],
                               "ci": np.percentile(boots[a] - boots[b], [2.5, 97.5]).tolist()}
    return out


def chart(results: dict) -> Path:
    with rc_context(STYLE):
        fig, axes = charts.figure(12, 6.0, 1, 2)
        for ax, (variant, res) in zip(axes[0], results.items()):
            for i, r in enumerate(READERS):
                d = res["readers"][r]
                xs = np.array([0, 1]) + (i - 1) * 0.04
                ys = [d["trained"], d["new"]]
                errs = np.array([[d["trained"] - d["trained_ci"][0], d["new"] - d["new_ci"][0]],
                                 [d["trained_ci"][1] - d["trained"], d["new_ci"][1] - d["new"]]])
                ax.errorbar(xs, ys, yerr=errs, color=charts.COLOURS[r], marker="o", markersize=6, linewidth=2.2,
                            capsize=3, label=f"{stored.LABELS[r]} (slope {d['slope']:+.3f})")
            ax.axhline(0, color=INK, linewidth=0.8)
            charts.style(ax, "gap to Qwen3-235B (target minus reader AUROC)" if variant == "none" else None)
            ax.set_xticks([0, 1], ["trained kinds\n(same kind, 5-fold)", f"new kinds\n(k = {K})"])
            ax.set_xlim(-0.35, 1.35)
            d = res["slope_difference"]
            ax.set_title(charts.READS[variant] + f"\nslope Qwen3.5-27B minus Gemma-3-27B: {d['mean']:+.3f}"
                         f" [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]", loc="left", fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        charts.note(fig, "Readers are fed Qwen3-235B's chats word for word; labels are always Qwen3-235B's."
                    " Gap = Qwen3-235B's own AUROC minus the reader's, on the same rows (0 = the reader keeps up).",
                    "Trained kinds: each kind's own probe, trained and tested inside it (5-fold); gap averaged over"
                    " the kinds.",
                    "New kinds: target and reader trained on the same 20 random picks of 8 kinds, tested on the kinds"
                    " left out; gap averaged over the picks.",
                    "Bars: 95% intervals from resampling the kinds (left) and the picks (right). Slope = new-kinds gap"
                    " minus trained-kinds gap. " + charts.DOUBLED_DOWN)
        fig.suptitle("At \"Did you lie?\" only Gemma readers fall behind on new kinds; on trained kinds all readers keep up",
                     fontsize=11, x=0.01, ha="left")
    path = HERE / "same_family_control.png"
    fig.savefig(path, dpi=200)
    return path


def main() -> None:
    results = {variant: control(variant) for variant in charts.READS}
    for variant, res in results.items():
        print(f"\n{variant}: {len(res['kinds'])} trained kinds, {res['subsets']} training subsets at k = {K}", flush=True)
        for r, d in res["readers"].items():
            print(f"  control | {variant} | {stored.LABELS[r]}: gap on trained kinds {d['trained']:+.3f}"
                  f" [{d['trained_ci'][0]:+.3f}, {d['trained_ci'][1]:+.3f}], on new kinds {d['new']:+.3f}"
                  f" [{d['new_ci'][0]:+.3f}, {d['new_ci'][1]:+.3f}], slope {d['slope']:+.3f}"
                  f" [{d['slope_ci'][0]:+.3f}, {d['slope_ci'][1]:+.3f}]", flush=True)
        d = res["slope_difference"]
        print(f"  control | {variant} | slope difference {d['pair']}: {d['mean']:+.3f}"
              f" [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]", flush=True)
    data = HERE / "control.json"
    data.write_text(json.dumps(results), encoding="utf-8")
    for path in (data, chart(results)):
        stored.push(path, HERE.name)
    print("wrote control.json, same_family_control.png", flush=True)


if __name__ == "__main__":
    main()
