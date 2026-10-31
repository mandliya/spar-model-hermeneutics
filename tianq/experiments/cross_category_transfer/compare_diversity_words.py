"""CPU: the diversity experiment with the words alone in place of a probe (experiments/cross_category_transfer/diversity.py)

    DYL_MODEL=qwen3-235b python -m experiments.cross_category_transfer.compare_diversity_words

Does training on more kinds of lie help when there are no activations at all, only the conversation's
words (the faithfulness gap's words-only substitute: TF-IDF + logistic regression)? If the words rise
with k as the target's probe does, the probe's transfer can be read off the text.

Every model of DYL_MODEL's diversity run trains on the same rows: this builds each training set with
diversity.training_set itself, on the target's stored rows and labels (its faithfulness scores, in the
same uid order as its activations), so each words classifier sees exactly the rows of the matching
probe. Both reads see the same words (compare_faithfulness.conversation_texts); they differ in their
rows and labels. Every training set's size and class counts are checked against the stored diversity
result before anything is stored. Stored as gigi/results/diversity_words_<recipe>.json.
"""

from __future__ import annotations

import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from dyl_repro.core import training
from dyl_repro.core.model import CURRENT, IS_SELF_READ
from dyl_repro.core.store import SCRATCH, RowStore
from experiments.cross_category_transfer import diversity
from experiments.cross_category_transfer.compare_diversity import NAME as DIVERSITY
from experiments.cross_category_transfer.compare_generalization import fetch
from experiments.cross_category_transfer.generalization import CATEGORIES
from experiments.hermeneutics import faithfulness
from experiments.hermeneutics.compare_faithfulness import conversation_texts, stored_scores

NAME = DIVERSITY.replace("diversity_", "diversity_words_", 1)


def categories_of(scores: dict) -> dict[str, diversity.Category]:
    """The target's trainable rows of one read, in its uid (and cache) order. x holds each row's position,
    so diversity.training_set returns which rows it picked."""
    out = {}
    for name in CATEGORIES:
        s = scores[name]
        scored = dict(zip(s["uids"], s["label"]))
        uids = s["trainable_uids"]
        out[name] = diversity.Category(np.arange(len(uids))[:, None], np.array(s["trainable_labels"]),
                                       np.array([scored.get(u, diversity.NOT_SCORED) for u in uids]), tuple(uids))
    return out


def words_run(categories: dict[str, diversity.Category], texts: dict[str, str], variant: str, k: int, index: int,
              subset: tuple[str, ...], line: str) -> dict:
    """diversity.run with the words in place of the activations: the same training rows and weights."""
    record = {"variant": variant, "k": k, "index": index, "subset": list(subset), "line": line}
    built = diversity.training_set(categories, subset, line, np.random.default_rng([diversity.SEED, k, index]))
    if built is None:
        return record | {"skipped": f"fewer than {diversity.SIZE_MATCHED_ROWS} labelled rows in these categories"}
    positions, y, weights, rows = built
    record |= {"rows": rows, "n_lie": int((y == 1).sum()), "n_honest": int((y == 0).sum())}
    if min(record["n_lie"], record["n_honest"]) < training.MIN_ROWS_PER_CLASS:
        return record | {"skipped": f"fewer than {training.MIN_ROWS_PER_CLASS} rows in a class"}
    names = np.repeat(list(rows), list(rows.values()))
    words = faithfulness.words_vectorizer()
    x = words.fit_transform([texts[categories[n].uids[i]] for n, i in zip(names, positions[:, 0])])
    # The probe's weights sum to 1; scaled to a mean of 1 here, so the classifier's regularisation is unchanged.
    fit = faithfulness.words_classifier().fit(x, y, sample_weight=None if weights is None else weights * len(y) / weights.sum())
    record["held_out"] = {}
    for name in CATEGORIES:
        if name in subset:
            continue
        c = categories[name]
        scored = np.flatnonzero(c.test_label != diversity.NOT_SCORED)
        test = c.test_label[scored]
        s = fit.decision_function(words.transform([texts[c.uids[i]] for i in scored]))
        n_lie, n_honest = int((test == 1).sum()), int((test == 0).sum())
        record["held_out"][name] = {"auroc": float(roc_auc_score(test, s)) if n_lie and n_honest else None,
                                    "n_lie": n_lie, "n_honest": n_honest}
    return record | diversity.held_out_mean(record)


def same_training_sets(records: list[dict], probes: list[dict]) -> list[tuple]:
    """Every training set whose size or class counts differ from the stored probe's (none, if the rows match)."""
    stored = {(r["variant"], r["line"], r["k"], r["index"]): r for r in probes}
    differ = []
    for r in records:
        key = (r["variant"], r["line"], r["k"], r["index"])
        if any(r.get(f) != stored[key].get(f) for f in ("rows", "n_lie", "n_honest", "skipped")):
            differ.append(key)
    return differ


def main() -> None:
    if not IS_SELF_READ:
        raise SystemExit("the words are the target's: run with DYL_MODEL only, no DYL_READER")
    print(f"words only, on the conversations of {CURRENT.name}", flush=True)
    target = stored_scores(CURRENT.tag)["scores"]
    texts = {}
    for name in CATEGORIES:
        texts |= conversation_texts(name)
    plan = diversity.subsets()
    records = []
    for variant in diversity.VARIANTS:
        categories = categories_of(target[variant])
        for line in diversity.LINES:
            for k, subsets in plan.items():
                for index, subset in enumerate(subsets):
                    started = time.time()
                    record = words_run(categories, texts, variant, k, index, subset, line)
                    records.append(record)
                    mean = record.get("held_out_mean")
                    print(f"[{len(records)}] {variant} {line} k={k} #{index}: "
                          + (record.get("skipped") or ("-" if mean is None else f"held-out mean {mean:.3f}"))
                          + f" ({time.time() - started:.1f}s)", flush=True)

    differ = same_training_sets(records, fetch(f"gigi/results/{DIVERSITY}.json")["records"])
    print(f"\ntraining sets checked against the stored probes: {len(records)}, {len(differ)} differ", flush=True)
    if differ:
        raise SystemExit(f"the words did not train on the probes' rows: {differ[:3]}")

    summaries = {variant: diversity.summary([r for r in records if r["variant"] == variant])
                 for variant in diversity.VARIANTS}
    print("\nWORDS ONLY: HELD-OUT AUROC BY NUMBER OF TRAINING CATEGORIES (mean over subsets, min-max, subsets scored)\n")
    print(diversity.markdown(summaries))
    without_refusals = {variant: diversity.summary([r for r in records if r["variant"] == variant],
                                                   diversity.REFUSAL_CONFOUNDED)
                        for variant in diversity.VARIANTS}
    print(f"\nSENSITIVITY: THE SAME, ALSO LEAVING OUT {', '.join(diversity.REFUSAL_CONFOUNDED)} WHEN HELD OUT\n")
    print(diversity.markdown(without_refusals), flush=True)

    RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, "gigi").put_result(NAME, {
        "model": {"words_of": CURRENT.name, "classifier": "TF-IDF (1-2 grams, min_df 2, sublinear) + logistic regression C=1"},
        "same_rows_as": f"gigi/results/{DIVERSITY}.json",
        "summary": summaries,
        "markdown": diversity.markdown(summaries),
        "sensitivity_without_refusal_confounded": {"left_out": list(diversity.REFUSAL_CONFOUNDED),
                                                    "summary": without_refusals,
                                                    "markdown": diversity.markdown(without_refusals)},
        "records": records,
    })


if __name__ == "__main__":
    main()
