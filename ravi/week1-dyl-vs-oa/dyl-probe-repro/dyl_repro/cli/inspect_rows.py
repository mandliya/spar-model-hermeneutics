"""Print what stored rows contain, for eyeballing or auditing a run.

    python -m dyl_repro.cli.inspect_rows --dataset aisi_city_train             # one line per row
    python -m dyl_repro.cli.inspect_rows --dataset aisi_city_train --summary   # audit counts only

Per row: label, answer token, prefill fidelity, the model's top predictions,
and the hidden-state tensor. The summary checks every row holds both positions
at every layer, finite, and reports the fidelity spread.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from dyl_repro import datasets
from dyl_repro.core.prefill import POSITIONS
from dyl_repro.core.store import SCRATCH, ActivationCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=datasets.NAMES)
    parser.add_argument("--followup", help="for datasets with swappable follow-ups: which one")
    parser.add_argument("--summary", action="store_true", help="audit counts only, no per-row lines")
    args = parser.parse_args()

    cache = ActivationCache(SCRATCH, f"{args.dataset}/{args.followup}" if args.followup else args.dataset)
    logprobs, shapes, position_lists, finite = [], Counter(), Counter(), 0
    for record, hidden_states in cache.iter_rows():
        f = record["fidelity"]
        logprobs.append(f["answer_first_logprob"])
        shapes[hidden_states.shape] += 1
        position_lists[tuple(record["hidden_states"]["positions"])] += 1
        finite += bool(np.isfinite(hidden_states).all())
        if not args.summary:
            top = ", ".join(f"{t!r} {lp:.2f}" for t, lp in zip(f["top_token_texts"][:3], f["top_logprobs"][:3]))
            print(
                f"{record['uid']}  label={record['label']}  answer={record['answer_first_text']!r}"
                f"  logprob={f['answer_first_logprob']:.4f}  top3=[{top}]"
                f"  hs={hidden_states.shape} finite={bool(np.isfinite(hidden_states).all())}"
            )

    n = len(logprobs)
    print(f"\n{args.dataset}: {n} rows stored")
    print(f"  tensor shapes:   {dict(shapes)}")
    print(f"  positions saved: {dict(position_lists)}   (expected {POSITIONS})")
    print(f"  all finite:      {finite} of {n}")
    print(
        f"  answer logprob:  min {min(logprobs):.4f}  median {np.median(logprobs):.4f}"
        f"  share above -1: {np.mean(np.array(logprobs) > -1):.4f}"
    )
    if not args.summary:
        first = cache.records[0]
        print(f"\nrendered_text of {first['uid']}:\n{first['rendered_text']}")


if __name__ == "__main__":
    main()
