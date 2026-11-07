"""Question 2: is the same-kind figure the k = 1 point of the k-curve?

    ./run.sh yuqi same_kind_vs_k_equals_one.figures        # from tianq/dyl-probe-repro (CPU, stored results)

heatmap_qwen3-235b.png   train on one kind (row), test on another (column), Qwen3-235B's own conversations.
                         Off the diagonal: the k = 1 probes (all-rows training). Each row's average over its
                         counted columns is that probe's held-out score; the average over rows is the k = 1
                         point. On the diagonal: each kind's own probe, 5-fold, the same-kind score.
same_kind_vs_k1.png      per target: the same-kind average next to the k = 1 point (size-matched, the line
                         the k-curve charts plot, and all-rows, the heatmap's).
same_kind_figure.png     the same-kind figure shown in the meeting (hermeneutics/report), copied as is.
All from compare_diversity's stored records and in-category references.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.patches import Rectangle

from experiments.common.chart_style import AUROC_SCALE, INK, MUTED, STYLE
from experiments.cross_category_transfer import diversity
from experiments.cross_category_transfer.generalization import CATEGORIES
from experiments.yuqi_questions.shared import charts, stored

HERE = Path(__file__).parent
SHOWN = HERE.parents[1] / "hermeneutics" / "report" / "faithfulness_gaps_qwen235b.png"
SHORT = {"ascii_sandbagging": "ASCII", "sycophancy_answer": "Syco A", "sycophancy_feedback": "Syco B",
         "mask_factual": "MASK provided", "mask_known_facts": "MASK known", "mask_statistics": "MASK stats",
         "mask_continuations": "MASK cont.", "mask_disinformation": "MASK disinfo", "unanswerable": "Unanswerable",
         "capture_the_secret": "Capture secret", "alibi": "Alibi"}


def matrix(result: dict, variant: str) -> np.ndarray:
    """AUROC, rows = trained on, columns = tested on; NaN where a cell is not counted in a held-out mean."""
    names = list(CATEGORIES)
    m = np.full((len(names), len(names)), np.nan)
    per = result["in_category_reference_detail"][variant]["per_category"]
    for i, name in enumerate(names):
        own = per.get(name)
        if own and diversity.trusted(own) and name not in diversity.UNSCORED_HELD_OUT[variant]:
            m[i, i] = own["auroc"]
        record = next(r for r in result["records"] if r["variant"] == variant and r["line"] == "all_rows"
                      and r["k"] == 1 and r["subset"] == [name])
        for j, other in enumerate(names):
            cell = record.get("held_out", {}).get(other)
            if cell and diversity.trusted(cell) and other not in diversity.UNSCORED_HELD_OUT[variant]:
                m[i, j] = cell["auroc"]
    return m


def heatmap(target: str) -> Path:
    result = stored.diversity(target)
    with rc_context(STYLE):
        fig, axes = charts.figure(13, 7.0, 1, 2)
        for ax, variant in zip(axes[0], charts.READS):
            m = matrix(result, variant)
            ax.imshow(m, cmap=AUROC_SCALE, vmin=0, vmax=1)
            for i in range(len(m)):
                for j in range(len(m)):
                    if not np.isnan(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                                color=INK, fontweight="bold" if i == j else "normal")
                ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
            labels = [SHORT[n] for n in CATEGORIES]
            ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_xlabel("tested on")
            ax.set_ylabel("trained on" if variant == "none" else None)
            diagonal = np.nanmean(np.diag(m))
            off = np.nanmean([np.nanmean([m[i, j] for j in range(len(m)) if j != i]) for i in range(len(m))
                              if not np.all(np.isnan([m[i, j] for j in range(len(m)) if j != i]))])
            ax.set_title(charts.READS[variant] + f"\ndiagonal (same kind) {diagonal:.2f};"
                         f" off-diagonal (k = 1) {off:.2f}", loc="left", fontsize=9)
            print(f"  heatmap | {target} | {variant}: same kind (diagonal) mean {diagonal:.3f};"
                  f" k = 1 (off-diagonal, row means averaged) {off:.3f}", flush=True)
        fig.suptitle(f"{stored.LABELS[target]}: the k = 1 point averages the off-diagonal (new kinds);"
                     " the same-kind figure is the diagonal", fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Each box: one probe trained on every row of one kind (row) and tested on another kind (column),"
                    " as AUROC. Qwen3-235B's own conversations.",
                    "Diagonal (boxed): train and test on the same kind, 5-fold, so every row is scored by a probe"
                    " that never saw it. This is the same-kind figure.",
                    "k = 1 point: average each row's off-diagonal boxes, then average the rows."
                    " Blank boxes are not scored: over the reply, sandbagging (ASCII) is one letter.",
                    charts.DOUBLED_DOWN)
    path = HERE / f"heatmap_{target}.png"
    fig.savefig(path, dpi=200)
    return path


def bars() -> Path:
    with rc_context(STYLE):
        fig, axes = charts.figure(12, 5.4, 1, 2)
        for ax, variant in zip(axes[0], charts.READS):
            for i, target in enumerate(stored.TARGETS):
                result = stored.diversity(target)
                same = result["in_category_reference"][variant]
                k1 = {line: result["summary"][variant][line]["1"]["mean"] for line in ("size_matched", "all_rows")}
                ax.bar(i - 0.2, same, 0.38, color=charts.COLOURS[target])
                ax.bar(i + 0.2, k1["size_matched"], 0.38, color=charts.COLOURS[target], alpha=0.35)
                ax.text(i - 0.2, same + 0.01, f"{same:.2f}", ha="center", fontsize=7)
                ax.text(i + 0.2, k1["size_matched"] + 0.01, f"{k1['size_matched']:.2f}", ha="center", fontsize=7)
                print(f"  same kind vs k = 1 | {target} | {variant}: same kind {same:.3f}; k = 1 size-matched"
                      f" {k1['size_matched']:.3f}, all rows {k1['all_rows']:.3f}", flush=True)
            charts.chance(ax)
            charts.style(ax, "AUROC" if variant == "none" else None)
            ax.set_xticks(range(len(stored.TARGETS)), [stored.LABELS[t] for t in stored.TARGETS], fontsize=8)
            ax.set_ylim(0.3, 1.0)
            ax.set_title(charts.READS[variant], loc="left", fontsize=10)
        axes[0][0].text(0.01, 0.97, "solid: same kind (5-fold)   faded: k = 1 (train on one kind, test on the others)",
                        transform=axes[0][0].transAxes, fontsize=8, color=MUTED, va="top")
        fig.suptitle("Same kind scores far above k = 1 for every target: they are different settings",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Solid bar, same kind: each kind of lie gets its own probe, trained and tested inside that kind"
                    " (5-fold); the bar averages those AUROCs over the kinds.",
                    "Faded bar, k = 1: a probe trained on one kind (80 rows) and tested on every other kind; AUROC"
                    " averaged over the tested kinds, then over every training kind with at least 80 rows.",
                    "Each model reads its own conversations. Lie label: did the reply lie. " + charts.DOUBLED_DOWN)
    path = HERE / "same_kind_vs_k1.png"
    fig.savefig(path, dpi=200)
    return path


def main() -> None:
    shown = HERE / "same_kind_figure.png"
    shutil.copyfile(SHOWN, shown)
    for path in (heatmap("qwen3-235b"), bars(), shown):
        stored.push(path, HERE.name)
    print("wrote heatmap_qwen3-235b.png, same_kind_vs_k1.png, same_kind_figure.png", flush=True)


if __name__ == "__main__":
    main()
