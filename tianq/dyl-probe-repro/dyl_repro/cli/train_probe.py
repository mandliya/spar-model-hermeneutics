"""CPU: train a DYL probe the way AISI trained theirs on one dataset, and test it on others.

    python -m dyl_repro.cli.train_probe --train aisi_city_train --test aisi_varied_deception aisi_alpaca
    python -m dyl_repro.cli.train_probe --train ascii_sandbagging/dyl --test sycophancy_answer/dyl aisi_varied_deception

Datasets are named by storage key, "<dataset>" or "<dataset>/<follow-up>", so any
dataset with any follow-up can be trained or tested on. The probe
(core/training.py) learns the training dataset's own labels (datasets.labels) at
the released probe's read position, its threshold is set on AISI's honest Alpaca
rows as the released probe's was, and each test dataset scores it with its own
evaluate, exactly as it scores the released probe (cli/evaluate.py).

Stored under the training key: the probe, probes/<probe>.pt, and its training
result, results/<probe>_hs<index>_training.json (row counts, threshold, the
cross-validated AUROC that is the in-distribution reference for every transfer
number, the cosine with the released probe's weights), and the per-row scores that
reference comes from, results/<probe>_hs<index>_out_of_fold.json (so it can be
read on a subset, as the transfer cells are). Under each test key:
results/<probe>_hs<index>.json. A training set with fewer than
training.MIN_ROWS_PER_CLASS rows in either class is not trained on: its training
result records why, and nothing is tested.

Resumable, and never overwrites: a probe already on the Hub is loaded rather than
retrained, and a test whose result is already there is skipped, so a rerun only
adds what is missing. Every cache it reads is pulled and checked first: an
incomplete one stops the run, since a probe or score built on part of a dataset
is not a real result. Testing on the training key is refused; its score is the
cross-validated one.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from dyl_repro import datasets
from dyl_repro.cli.evaluate import StoredRun, evaluate_run, pull_run
from dyl_repro.core import training
from dyl_repro.core.metrics import labelled
from dyl_repro.core.probe import CALIBRATION_FPR, PROBE_POSITION, DylProbe, hidden_states_index_for
from dyl_repro.core.store import SCRATCH, RowStore
from dyl_repro.datasets.aisi import alpaca

CALIBRATION_KEY = "aisi_alpaca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="storage key to train on, e.g. ascii_sandbagging/dyl")
    parser.add_argument("--test", nargs="+", required=True, help="storage keys to test on")
    parser.add_argument("--layer", type=int, default=training.DEFAULT_LAYER)
    parser.add_argument("--l2-lambda", type=float, default=training.DEFAULT_L2_LAMBDA)
    args = parser.parse_args()
    if args.train in args.test:
        parser.error(f"{args.train} is the training data; its in-distribution score is the cross-validated one")
    return args


def pull_all(keys: list[str]) -> dict[str, StoredRun]:
    """Every run this invocation reads, pulled; stops if any is incomplete."""
    runs = {}
    for key in dict.fromkeys(keys):
        name, followup = datasets.run(key)
        runs[key] = pull_run(name, key, followup)
    incomplete = [key for key, run in runs.items() if not run.complete]
    if incomplete:
        raise SystemExit(f"incomplete caches, nothing trained or tested: {', '.join(incomplete)}")
    return runs


def training_rows(run: StoredRun, hidden_states_index: int) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """(activations, labels, records) of the labelled rows at the probe's read position, and how many were excluded."""
    _, _, activations = run.cache.matrix(PROBE_POSITION, hidden_states_index)
    mask, y = labelled(datasets.labels(run.dataset, run.cache, run.followup))
    records = [r for r, keep in zip(run.cache.records, mask) if keep]
    return activations[mask], y, records, int((~mask).sum())


def out_of_fold(name: str, train_key: str, x: np.ndarray, y: np.ndarray, records: list[dict], l2_lambda: float) -> dict:
    """Every labelled training row scored by a probe trained without it, kept per row so the
    in-category reference can be read on any subset (e.g. the denials only, as transfer
    cells are). turn_lie is the dataset's label for the original answer, where it has one."""
    scores = training.out_of_fold_scores(x, y, l2_lambda=l2_lambda)
    return {
        "probe": name,
        "trained_on": train_key,
        "cross_validated": training.cross_validated_auroc(y, scores),
        "rows": [{"uid": r["uid"], "label": int(label), "turn_lie": r["provenance"].get("turn_lie"), "score": float(s)}
                 for r, label, s in zip(records, y, scores)],
    }


def train_and_describe(train: StoredRun, calibration: StoredRun, name: str, layer: int,
                       l2_lambda: float) -> tuple[DylProbe | None, dict, dict | None]:
    """The calibrated probe, its training result and its out-of-fold rows; no probe (and no
    rows) when a class is too small to train on."""
    index = hidden_states_index_for(layer)
    x, y, records, n_excluded = training_rows(train, index)
    counts = {"probe": name, "trained_on": train.key, "layer": layer, "hidden_states_index": index,
              "position": PROBE_POSITION, "l2_lambda": l2_lambda,
              "n_lie": int((y == 1).sum()), "n_honest": int((y == 0).sum()), "n_excluded": n_excluded}
    if min(counts["n_lie"], counts["n_honest"]) < training.MIN_ROWS_PER_CLASS:
        return None, counts | {"skipped": f"fewer than {training.MIN_ROWS_PER_CLASS} rows in a class"}, None
    probe = training.train(x, y, layer=layer, l2_lambda=l2_lambda, trained_on=train.key)
    threshold = training.calibrate(probe, alpaca.honest_activations(calibration.cache, index))
    released = DylProbe.load()
    weight = probe.module.linear.weight.detach().double().numpy().ravel()
    rows = out_of_fold(name, train.key, x, y, records, l2_lambda)
    return probe, counts | {
        "threshold": threshold,
        "calibrated_on": calibration.key,
        "calibration_fpr": CALIBRATION_FPR,
        "cross_validated": rows["cross_validated"],
        "weight_norm": float(np.linalg.norm(weight)),
        "bias": float(probe.module.linear.bias),
        # Weights are comparable only between probes of the same layer.
        "released_probe": released.checkpoint,
        "weight_cosine_with_released": training.weight_cosine(probe, released) if layer == released.layer else None,
    }


def report(description: dict) -> None:
    print(f"\nTRAINED {description['probe']}")
    print(f"  {description['n_lie']} lie / {description['n_honest']} honest ({description['n_excluded']} excluded)"
          f" at {PROBE_POSITION}, hidden_states[{description['hidden_states_index']}]")
    if "skipped" in description:
        print(f"  not trained: {description['skipped']}")
        return
    cv = description["cross_validated"]
    print(f"  in-distribution AUROC ({cv['folds']}-fold cross-validated): {cv['auroc']:.4f}")
    print(f"  threshold {description['threshold']:.4f} (flags {CALIBRATION_FPR:.0%} of {CALIBRATION_KEY}'s honest rows)")
    cosine = description["weight_cosine_with_released"]
    print(f"  |w| {description['weight_norm']:.4f}, bias {description['bias']:.4f}; weight cosine with"
          f" {description['released_probe']}: " + ("other layer" if cosine is None else f"{cosine:.4f}"))


def main() -> None:
    args = parse_args()
    name = training.probe_name(args.train, args.layer, args.l2_lambda)
    index = hidden_states_index_for(args.layer)
    training_result, rows_result = f"{name}_hs{index}_training", f"{name}_hs{index}_out_of_fold"
    store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, args.train)
    stored = store.stored()
    pending = [key for key in args.test if f"{key}/results/{name}_hs{index}.json" not in stored]

    if f"{args.train}/results/{training_result}.json" in stored:
        description = store.result(training_result)
        trained = "skipped" not in description
        # Probes trained before out-of-fold rows were kept get them now, without retraining.
        missing_rows = trained and f"{args.train}/results/{rows_result}.json" not in stored
        print(f"{name}: already trained, loaded from the Hub; {len(pending)} of {len(args.test)} tests to run"
              + ("; out-of-fold rows to add" if missing_rows else ""))
        if not trained or not (pending or missing_rows):
            report(description)
            return
        runs = pull_all(([args.train] if missing_rows else []) + pending)
        probe = store.probe(name)
        if missing_rows:
            x, y, records, _ = training_rows(runs[args.train], index)
            rows = out_of_fold(name, args.train, x, y, records, args.l2_lambda)
            stored_auroc, now = description["cross_validated"]["auroc"], rows["cross_validated"]["auroc"]
            if not np.isclose(stored_auroc, now, rtol=0, atol=1e-9):
                raise SystemExit(f"out-of-fold AUROC {now} does not reproduce the stored {stored_auroc}")
            store.put_result(rows_result, rows)
    else:
        runs = pull_all([args.train, CALIBRATION_KEY, *pending])
        probe, description, rows = train_and_describe(runs[args.train], runs[CALIBRATION_KEY], name, args.layer,
                                                      args.l2_lambda)
        if probe is not None:
            store.put_probe(probe)
            store.put_result(rows_result, rows)
        store.put_result(training_result, description)
    report(description)
    if probe is None:
        return
    for key in pending:
        evaluate_run(runs[key], probe)


if __name__ == "__main__":
    main()
