"""CPU: does training on more kinds of lie help a probe catch a kind it never saw? (datasets/gigi/analyses/diversity.py)

    python -m dyl_repro.cli.compare_diversity

Trains every probe of the experiment (both variants, both training sets, every subset; seconds
each) from the stored activations, scores each on its held-out categories, and stores the result
as gigi/results/diversity_<recipe>.json, drawn as gigi/charts/diversity_<recipe>.png.

Before anything is stored, the k = 1 probes are checked against the transfer matrices already on
the Hub (no_followup.py for none, generalization.py for neutral): the same probes on the same rows,
so every held-out AUROC must match its cell. A mismatch stops the run.

Resumable: each finished probe is appended to a local checkpoint (scratch, not the record), and a
rerun skips what is in it; the stored result is always built from a complete set.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from dyl_repro import datasets
from dyl_repro.cli.compare_generalization import RESULT, fetch
from dyl_repro.cli.compare_no_followup import NAME as NO_FOLLOWUP_RESULT
from dyl_repro.core.store import SCRATCH, ActivationCache, RowStore
from dyl_repro.datasets.gigi import transfer
from dyl_repro.datasets.gigi.analyses import diversity, diversity_charts, generalization, no_followup
from dyl_repro.datasets.gigi.analyses.generalization import CATEGORIES, INDEX

NAME = RESULT.replace("generalization_", "diversity_", 1)
CHECKPOINT = SCRATCH / "gigi" / "checkpoints" / f"{NAME}.jsonl"


def scored(label: int | None) -> int:
    return diversity.NOT_SCORED if label is None else int(label)


def stored_row_count(variant: str, dataset: str) -> int:
    """The rows the category's single-category probe was trained from, read off its stored training
    result: the count of a complete cache, recorded when it was one."""
    path = (no_followup.result_paths(diversity.VARIANTS["none"].position, dataset)[0] if variant == "none"
            else generalization.result_paths("neutral", dataset)[0])
    trained = fetch(path)
    return trained["n_lie"] + trained["n_honest"] + trained["n_excluded"]


def load(variant: str, dataset: str) -> diversity.Category:
    """A category's stored rows. Completeness is checked against stored_row_count rather than by
    rebuilding the conversations (as cli/evaluate.py's pull_run does): those are built from
    Noddybear/lies, read unpinned by Gigi's code, and its 2026-10-05 revision renamed the configs."""
    key = diversity.key(dataset, variant)
    name, followup = datasets.run(key)
    RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, key).pull()
    cache = ActivationCache(SCRATCH, key)
    if len(cache.records) != stored_row_count(variant, dataset):
        raise SystemExit(f"{key}: {len(cache.records)} rows stored, {stored_row_count(variant, dataset)} expected")
    _, _, x = cache.matrix(diversity.VARIANTS[variant].position, INDEX)
    train = np.array([scored(y) for y in datasets.labels(name, cache, followup)])
    if variant == "none":
        test = train
    else:
        # Held out, a neutral probe is scored on the denials only: doubled down vs truthful.
        rows, _ = transfer.label_rows(cache, datasets.get(name).FOLLOWUPS[followup], None)
        test = np.array([{"lied_denied": 1, "truthful_denied": 0}.get(r["outcome"], diversity.NOT_SCORED)
                         for r in rows])
    return diversity.Category(x, train, test)


def checkpointed() -> dict[tuple, dict]:
    if not CHECKPOINT.exists():
        return {}
    records = [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line]
    return {(r["variant"], r["line"], r["k"], r["index"]): r for r in records}


def reproduces_matrices(records: list[dict]) -> dict:
    """Every k = 1, all_rows held-out AUROC against its cell in the stored transfer matrix."""
    stored = {
        "none": next(m for m in fetch(f"gigi/results/{NO_FOLLOWUP_RESULT}.json")["matrices"]
                     if m["position"] == diversity.VARIANTS["none"].position),
        "neutral": next(m for m in fetch(f"gigi/results/{RESULT}.json")["matrices"] if m["wording"] == "neutral"),
    }
    measure = {"none": no_followup.MEASURE, "neutral": "denials"}
    compared, mismatches = 0, []
    for r in records:
        if r["k"] != 1 or r["line"] != "all_rows" or "skipped" in r:
            continue
        m = stored[r["variant"]]
        row = next(row for row in m["rows"] if row["train"] == r["subset"][0])
        for column, cell in zip(m["columns"], row["cells"]):
            if column not in r["held_out"]:
                continue
            ours, theirs = r["held_out"][column], cell[measure[r["variant"]]]
            same = (theirs is not None and (ours["n_lie"], ours["n_honest"]) == (theirs["n_lie"], theirs["n_honest"])
                    and (ours["auroc"] is None) == (theirs["auroc"] is None)
                    and (ours["auroc"] is None or abs(ours["auroc"] - theirs["auroc"]) <= 1e-9))
            compared += 1
            if not same:
                mismatches.append({"variant": r["variant"], "train": r["subset"][0], "test": column,
                                   "ours": ours, "stored": theirs})
    return {"cells_compared": compared, "mismatches": mismatches}


def main() -> None:
    plan = diversity.subsets()
    done = checkpointed()
    total = len(diversity.VARIANTS) * len(diversity.LINES) * sum(len(s) for s in plan.values())
    print(f"{total} probes in all, {len(done)} already in {CHECKPOINT.name}", flush=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in diversity.VARIANTS:
        categories = None
        for line in diversity.LINES:
            for k, subsets in plan.items():
                for index, subset in enumerate(subsets):
                    if (variant, line, k, index) in done:
                        # Scored again from its stored held-out AUROCs, so a change in what counts applies.
                        stored = done[(variant, line, k, index)]
                        records.append(stored | diversity.held_out_mean(stored))
                        continue
                    if categories is None:
                        categories = {name: load(variant, name) for name in CATEGORIES}
                    started = time.time()
                    record = diversity.run(categories, variant, k, index, subset, line)
                    with CHECKPOINT.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    records.append(record)
                    mean = record.get("held_out_mean")
                    print(f"[{len(records)}/{total}] {variant} {line} k={k} #{index} {'+'.join(subset)}: "
                          + (record["skipped"] if "skipped" in record else
                             f"{record['n_lie']}/{record['n_honest']} rows, held-out mean "
                             + ("-" if mean is None else f"{mean:.3f}") + f" over {record['held_out_trusted']}")
                          + f" ({time.time() - started:.1f}s)", flush=True)

    check = reproduces_matrices(records)
    print(f"\nk = 1 against the stored transfer matrices: {check['cells_compared']} cells compared,"
          f" {len(check['mismatches'])} differ", flush=True)
    if check["mismatches"]:
        raise SystemExit(f"k = 1 does not reproduce the stored matrices: {check['mismatches'][:3]}")

    summaries = {variant: diversity.summary([r for r in records if r["variant"] == variant])
                 for variant in diversity.VARIANTS}
    table = diversity.markdown(summaries)
    print("\nHELD-OUT AUROC BY NUMBER OF TRAINING CATEGORIES (mean over subsets, min-max, subsets scored)\n")
    print(table)

    # The in-category reference: each category's own probe, cross-validated, averaged as the matrices do.
    references = {
        "none": fetch(f"gigi/results/{NO_FOLLOWUP_RESULT}.json")["summary"][diversity.VARIANTS["none"].position]
        [no_followup.MEASURE]["mean_in_category"],
        "neutral": fetch(f"gigi/results/{RESULT}.json")["summary"]["neutral"]["denials"]["mean_in_category"],
    }
    print(f"\nIN-CATEGORY REFERENCE (each category's own probe, 5-fold): none {references['none']:.4f}"
          f" (gigi/results/{NO_FOLLOWUP_RESULT}.json), neutral {references['neutral']:.4f}"
          f" (gigi/results/{RESULT}.json)")

    breakdown = {}
    for variant in diversity.VARIANTS:
        mine = [r for r in records if r["variant"] == variant]
        rows = diversity.labelled_rows(mine)
        largest = max(rows, key=rows.get)
        print(f"\n{variant}: labelled training rows per category: {rows}; {largest} is {rows[largest]} of"
              f" {sum(rows.values())} ({rows[largest] / sum(rows.values()):.0%})")
        breakdown[variant] = {"labelled_rows": rows}
        for line in diversity.LINES:
            table = diversity.per_category(mine, line)
            print(f"\n{variant}, {line}: HELD-OUT AUROC PER CATEGORY (mean over probes at each k)\n")
            print(diversity.per_category_markdown(table))
            splits = {
                "mask_with_another_mask": diversity.split_by_training(mine, line, diversity.MASK, diversity.MASK),
                "mask_with_alibi": diversity.split_by_training(mine, line, diversity.MASK, ("alibi",)),
            }
            for name, split in splits.items():
                t, n = split["trained"], split["not_trained"]
                print(f"  MASK held out, k in {diversity.SPLIT_KS}, {name.removeprefix('mask_with_')} trained:"
                      f" {t['mean']:.3f} ({t['cells']} cells) vs not {n['mean']:.3f} ({n['cells']} cells)")
            breakdown[variant][line] = {"per_category": table, "splits": splits}

    store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, "gigi")
    store.put_chart(NAME, diversity_charts.draw(records, summaries, references))
    store.put_result(NAME, {
        "design": {"ks": list(diversity.KS), "subsets_per_k": diversity.SUBSETS_PER_K, "seed": diversity.SEED,
                   "size_matched_rows": diversity.SIZE_MATCHED_ROWS, "few": diversity.FEW,
                   "variants": {v: {"followup": s.followup, "position": s.position,
                                    "unscored_held_out": list(diversity.UNSCORED_HELD_OUT[v])}
                                for v, s in diversity.VARIANTS.items()}},
        "reproduces_transfer_matrices": check,
        "in_category_reference": references,
        "summary": summaries,
        "markdown": diversity.markdown(summaries),
        "breakdown": breakdown,
        "records": records,
    })


if __name__ == "__main__":
    main()
