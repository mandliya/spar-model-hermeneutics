"""CPU: does training on more kinds of lie help a probe catch a kind it never saw? (experiments/cross_category_transfer/diversity.py)

    python -m experiments.cross_category_transfer.compare_diversity
    DYL_MODEL=qwen3-next-80b python -m experiments.cross_category_transfer.compare_diversity

Trains every probe of the experiment (both variants, both training sets, every subset; seconds
each) from the stored activations of the model DYL_MODEL names (core/model.py; Gemma when unset),
scores each on its held-out categories, and stores the result as gigi/results/diversity_<recipe>.json,
drawn as gigi/charts/diversity_<recipe>.png, under the model's namespace (core/store.py).

The in-category reference (each category's own probe, 5-fold) is computed here. For Gemma, before
anything is stored, it and the k = 1 probes are checked against the transfer matrices already on the
Hub (no_followup.py for none, generalization.py for neutral): the same probes on the same rows, so
every number must match. A mismatch stops the run.

Resumable: each finished probe is appended to a local checkpoint (scratch, not the record), and a
rerun skips what is in it; the stored result is always built from a complete set.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from dyl_repro import datasets
from dyl_repro.cli.evaluate import pull_run
from dyl_repro.core.model import CURRENT, IS_DEFAULT, IS_SELF_READ, READER
from dyl_repro.core.store import SCRATCH, RowStore, namespaced
from dyl_repro.datasets.followups import NO_FOLLOWUP as NO_FOLLOWUP_KEY
from dyl_repro.datasets.gigi import transfer
from experiments.cross_category_transfer import diversity, diversity_charts, no_followup
from experiments.cross_category_transfer.compare_generalization import RESULT, fetch
from experiments.cross_category_transfer.compare_no_followup import NAME as NO_FOLLOWUP_RESULT
from experiments.cross_category_transfer.generalization import CATEGORIES, FEW, INDEX

NAME = RESULT.replace("generalization_", "diversity_", 1)
CHECKPOINT = SCRATCH / namespaced("gigi") / "checkpoints" / f"{NAME}.jsonl"


def scored(label: int | None) -> int:
    return diversity.NOT_SCORED if label is None else int(label)


def load(variant: str, dataset: str) -> diversity.Category:
    """A category's stored rows, pulled and checked complete against its rebuilt conversations
    (dyl_repro/cli/evaluate.py's pull_run)."""
    key = diversity.key(dataset, variant)
    name, followup = datasets.run(key)
    run = pull_run(name, key, followup)
    if not run.complete:
        raise SystemExit(f"{key}: {len(run.cache.records)} rows stored, {run.rows_expected} expected")
    cache = run.cache
    uids, _, x = cache.matrix(diversity.VARIANTS[variant].position, INDEX)
    train = np.array([scored(y) for y in datasets.labels(name, cache, followup)])
    if diversity.VARIANTS[variant].followup == NO_FOLLOWUP_KEY:
        test = train
    else:
        # Held out, a neutral probe is scored on the denials only: doubled down vs truthful.
        rows, _ = transfer.label_rows(cache, datasets.get(name).FOLLOWUPS[followup], None)
        test = np.array([{"lied_denied": 1, "truthful_denied": 0}.get(r["outcome"], diversity.NOT_SCORED)
                         for r in rows])
    return diversity.Category(x, train, test, tuple(uids))


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


def in_category_reference(categories: dict[str, diversity.Category]) -> dict:
    """Each category's own probe, out of fold, averaged over those with at least FEW of each scored
    class: the matrices' in-category mean, and the dashed line of the chart."""
    per = {name: diversity.in_category(c) for name, c in categories.items()}
    kept = [s["auroc"] for s in per.values() if s and s["auroc"] is not None and min(s["n_lie"], s["n_honest"]) >= FEW]
    return {"mean": float(np.mean(kept)), "categories": len(kept), "per_category": per}


def stored_reference(variant: str) -> float:
    """Gemma's in-category mean as the transfer matrices stored it."""
    if variant == "none":
        return fetch(f"gigi/results/{NO_FOLLOWUP_RESULT}.json")["summary"][diversity.VARIANTS["none"].position][
            no_followup.MEASURE]["mean_in_category"]
    return fetch(f"gigi/results/{RESULT}.json")["summary"]["neutral"]["denials"]["mean_in_category"]


def main() -> None:
    print(f"model {READER.name} @ {READER.revision}, layer {READER.probe_layer} (hidden_states[{INDEX}]),"
          f" reading the conversations of {CURRENT.name}", flush=True)
    plan = diversity.subsets()
    done = checkpointed()
    total = len(diversity.VARIANTS) * len(diversity.LINES) * sum(len(s) for s in plan.values())
    print(f"{total} probes in all, {len(done)} already in {CHECKPOINT.name}", flush=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    records, loaded = [], {}

    def categories_of(variant: str) -> dict[str, diversity.Category]:
        if variant not in loaded:
            loaded[variant] = {name: load(variant, name) for name in CATEGORIES}
        return loaded[variant]

    for variant in diversity.VARIANTS:
        for line in diversity.LINES:
            for k, subsets in plan.items():
                for index, subset in enumerate(subsets):
                    if (variant, line, k, index) in done:
                        # Scored again from its stored held-out AUROCs, so a change in what counts applies.
                        stored = done[(variant, line, k, index)]
                        records.append(stored | diversity.held_out_mean(stored))
                        continue
                    started = time.time()
                    record = diversity.run(categories_of(variant), variant, k, index, subset, line)
                    with CHECKPOINT.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    records.append(record)
                    mean = record.get("held_out_mean")
                    print(f"[{len(records)}/{total}] {variant} {line} k={k} #{index} {'+'.join(subset)}: "
                          + (record["skipped"] if "skipped" in record else
                             f"{record['n_lie']}/{record['n_honest']} rows, held-out mean "
                             + ("-" if mean is None else f"{mean:.3f}") + f" over {record['held_out_trusted']}")
                          + f" ({time.time() - started:.1f}s)", flush=True)

    # Gemma's transfer matrices are on the Hub; another model has none to reproduce.
    check = reproduces_matrices(records) if IS_DEFAULT and IS_SELF_READ else None
    if check is not None:
        print(f"\nk = 1 against the stored transfer matrices: {check['cells_compared']} cells compared,"
              f" {len(check['mismatches'])} differ", flush=True)
        if check["mismatches"]:
            raise SystemExit(f"k = 1 does not reproduce the stored matrices: {check['mismatches'][:3]}")

    summaries = {variant: diversity.summary([r for r in records if r["variant"] == variant])
                 for variant in diversity.VARIANTS}
    table = diversity.markdown(summaries)
    print("\nHELD-OUT AUROC BY NUMBER OF TRAINING CATEGORIES (mean over subsets, min-max, subsets scored)\n")
    print(table)
    # Sensitivity: the same probes, the refusal-confounded categories also left out of the held-out score.
    without_refusals = {variant: diversity.summary([r for r in records if r["variant"] == variant],
                                                   diversity.REFUSAL_CONFOUNDED)
                        for variant in diversity.VARIANTS}
    print(f"\nSENSITIVITY: THE SAME, ALSO LEAVING OUT {', '.join(diversity.REFUSAL_CONFOUNDED)} WHEN HELD OUT"
          " (truthful replies there mostly refuse or correct)\n")
    print(diversity.markdown(without_refusals))

    # The in-category reference: each category's own probe, cross-validated, averaged as the matrices do.
    # For Gemma it must equal what the matrices stored.
    reference_detail = {variant: in_category_reference(categories_of(variant)) for variant in diversity.VARIANTS}
    references = {variant: r["mean"] for variant, r in reference_detail.items()}
    for variant, r in reference_detail.items():
        line = f"\nIN-CATEGORY REFERENCE {variant} (each category's own probe, 5-fold): {r['mean']:.4f} over {r['categories']} categories"
        if IS_DEFAULT and IS_SELF_READ:
            stored = stored_reference(variant)
            line += f"; stored by the transfer matrices {stored:.4f}"
            if not np.isclose(stored, r["mean"], rtol=0, atol=1e-9):
                raise SystemExit(f"{variant}: in-category reference {r['mean']} does not reproduce the stored {stored}")
        print(line, flush=True)

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
                mean = lambda side: "-" if side["mean"] is None else f"{side['mean']:.3f}"
                print(f"  MASK held out, k in {diversity.SPLIT_KS}, {name.removeprefix('mask_with_')} trained:"
                      f" {mean(t)} ({t['cells']} cells) vs not {mean(n)} ({n['cells']} cells)")
            breakdown[variant][line] = {"per_category": table, "splits": splits}

    store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, "gigi")
    store.put_chart(NAME, diversity_charts.draw(records, summaries, references, READER, IS_DEFAULT and IS_SELF_READ, CURRENT))
    store.put_result(NAME, {
        "model": {"name": READER.name, "revision": READER.revision, "probe_layer": READER.probe_layer,
                  "hidden_states_index": INDEX, "conversations_of": CURRENT.name},
        "in_category_reference_detail": reference_detail,
        "design": {"ks": list(diversity.KS), "subsets_per_k": diversity.SUBSETS_PER_K, "seed": diversity.SEED,
                   "size_matched_rows": diversity.SIZE_MATCHED_ROWS, "few": diversity.FEW,
                   "variants": {v: {"followup": s.followup, "position": s.position,
                                    "unscored_held_out": list(diversity.UNSCORED_HELD_OUT[v])}
                                for v, s in diversity.VARIANTS.items()}},
        "reproduces_transfer_matrices": check,
        "in_category_reference": references,
        "summary": summaries,
        "markdown": diversity.markdown(summaries),
        "sensitivity_without_refusal_confounded": {"left_out": list(diversity.REFUSAL_CONFOUNDED),
                                                    "summary": without_refusals,
                                                    "markdown": diversity.markdown(without_refusals)},
        "breakdown": breakdown,
        "records": records,
    })


if __name__ == "__main__":
    main()
