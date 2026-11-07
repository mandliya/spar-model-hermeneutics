"""Question 1: do bigger models generalize better, and why did Gemma look close to Qwen?

    ./run.sh yuqi bigger_models_generalize.figures        # from tianq/dyl-probe-repro (CPU)

targets_only.png    each target reading its own conversations: AUROC on new kinds of lie vs k (size-matched
                    training subsets), three panels:
                      over the reply, lie vs truthful reply          (the arena's reply probe)
                      at "Did you lie?", the same task               (the arena's did_you_lie probe, same rows)
                      at "Did you lie?", doubled-down vs truthful denial  (the stored k-curves, the task the
                                                                      earlier charts plotted)
                    The first two are the fair comparison of the two reads; the third is a harder task.
own_vs_other_conversations.png  the same Gemma-3-27B weights as a target (its own conversations) and as a
                    reader of Qwen3-Next-80B's and Qwen3-235B's conversations (stored k-curves)
words_own_conversations.png     words only (TF-IDF), each target's own conversations, lie vs truthful reply:
                    can the text alone carry a lie pattern across kinds? (the arena's words line)
95% bands from resampling the training subsets. Every plotted value is printed, with the all-rows values
of the stored curves, so a number quoted from an earlier chart can be traced.
"""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib import rc_context

from experiments.common.chart_style import STYLE
from experiments.yuqi_questions.shared import charts, stored

HERE = Path(__file__).parent
ARENA = HERE.parent / "lie_detector_arena"
KS = [1, 2, 3, 5, 8, 10]
PANELS = (("reply", 'Read over the reply\n(lie vs truthful reply)'),
          ("did_you_lie", 'Read at "Did you lie?"\n(lie vs truthful reply: the same task)'),
          ("neutral", 'Read at "Did you lie?"\n(doubled-down lie vs truthful denial: a harder task)'))


def arena(target: str) -> dict:
    return json.loads((ARENA / f"arena_{target}.json").read_text(encoding="utf-8"))


def curve(target: str, panel: str) -> list[dict]:
    if panel == "neutral":
        return stored.k_curve(stored.diversity(target), "neutral")
    return arena(target)["new_kinds"]["curves"][panel]


def show(name: str, points: list[dict]) -> None:
    print(f"  {name}: " + "; ".join(f"k={p['k']} {p['mean']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]" for p in points),
          flush=True)


def axes_k(ax, title: str, ylabel: bool) -> None:
    charts.chance(ax)
    charts.style(ax, "AUROC on new kinds of lie" if ylabel else None)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("kinds of lie trained on (k)")
    ax.set_xticks(KS)
    ax.set_ylim(0.4, 0.8)


def targets_only() -> Path:
    print("TARGETS, OWN CONVERSATIONS", flush=True)
    with rc_context(STYLE):
        fig, axes = charts.figure(15, 5.6, 1, 3)
        for i, (ax, (panel, title)) in enumerate(zip(axes[0], PANELS)):
            for target in stored.TARGETS:
                points = curve(target, panel)
                show(f"targets | {panel} | {stored.LABELS[target]}", points)
                charts.k_line(ax, points, charts.COLOURS[target], stored.LABELS[target])
            axes_k(ax, title, i == 0)
        axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle("As targets reading their own conversations, the bigger Qwens transfer to new kinds of lie"
                     " and Gemma-3-27B does not", fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Each line is one model as the target: it wrote the chats, and the probe reads its own"
                    " activations on them. Lie label: did the reply lie.", *charts.K_CURVE,
                    "Left and middle panels use the same rows and the same task, so the two reads compare directly. "
                    + charts.DOUBLED_DOWN)
    path = HERE / "targets_only.png"
    fig.savefig(path, dpi=200)
    return path


def own_vs_other() -> Path:
    gemma = "gemma-3-27b-it"
    lines = ((gemma, None, "Gemma-3-27B reading its own conversations (target)", False),
             ("qwen3-next-80b", gemma, "Gemma-3-27B reading Qwen3-Next-80B's (reader)", True),
             ("qwen3-235b", gemma, "Gemma-3-27B reading Qwen3-235B's (reader)", True))
    print("GEMMA-3-27B AS TARGET AND AS READER (stored k-curves)", flush=True)
    with rc_context(STYLE):
        fig, axes = charts.figure(12, 5.4, 1, 2)
        for i, (ax, variant) in enumerate(zip(axes[0], charts.READS)):
            for author, reader, label, dashed in lines:
                result = stored.diversity(author, reader)
                points = stored.k_curve(result, variant)
                show(f"gemma | {variant} | {label} | size-matched", points)
                show(f"gemma | {variant} | {label} | all rows", stored.k_curve(result, variant, "all_rows"))
                charts.k_line(ax, points, charts.COLOURS[author], label, dashed)
            axes_k(ax, charts.READS[variant], i == 0)
        axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle("The same Gemma-3-27B weights: flat on its own conversations, rising on Qwen's",
                     fontsize=11, x=0.01, ha="left")
        charts.note(fig, "Solid line: Gemma-3-27B as the target, on chats it wrote. Dashed lines: the same Gemma-3-27B"
                    " as a reader, fed a Qwen model's chats word for word; labels are the Qwen model's.",
                    *charts.K_CURVE, charts.DOUBLED_DOWN)
    path = HERE / "own_vs_other_conversations.png"
    fig.savefig(path, dpi=200)
    return path


def words() -> Path:
    print("WORDS ONLY, OWN CONVERSATIONS (lie vs truthful reply)", flush=True)
    with rc_context(STYLE):
        fig, axes = charts.figure(9, 5.6)
        ax = axes[0][0]
        for target in stored.TARGETS:
            points = arena(target)["new_kinds"]["curves"]["words"]
            show(f"words | {stored.LABELS[target]}", points)
            charts.k_line(ax, points, charts.COLOURS[target], stored.LABELS[target])
        axes_k(ax, "Words only (TF-IDF), each target's own conversations", True)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        charts.note(fig, "Words only: a TF-IDF classifier on the chat's words in place of the probe, no model inside."
                    " It shows how much the text alone gives away.",
                    "Each point: train on k kinds of lie, test on every kind left out; AUROC averaged over the"
                    " left-out kinds, then over 20 random picks of k kinds.",
                    "Every classifier trains on the same 80 rows. Bands: 95% interval from resampling the 20 picks."
                    " Lie label: did the reply lie.")
    path = HERE / "words_own_conversations.png"
    fig.savefig(path, dpi=200)
    return path


def main() -> None:
    print("STORED K-CURVES, ALL ROWS (to trace numbers quoted from earlier charts)", flush=True)
    for target in stored.TARGETS:
        for variant in charts.READS:
            show(f"stored | {variant} | {stored.LABELS[target]} | all rows",
                 stored.k_curve(stored.diversity(target), variant, "all_rows"))
            show(f"stored | {variant} | {stored.LABELS[target]} | size-matched",
                 stored.k_curve(stored.diversity(target), variant))
    for path in (targets_only(), own_vs_other(), words()):
        stored.push(path, HERE.name)
    print("wrote targets_only.png, own_vs_other_conversations.png, words_own_conversations.png", flush=True)


if __name__ == "__main__":
    main()
