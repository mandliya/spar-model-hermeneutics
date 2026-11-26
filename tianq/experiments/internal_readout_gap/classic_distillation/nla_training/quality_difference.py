"""Is one pair's rebuild quality really different from another's on the same activations, or within sampling noise?

    python3 quality_difference.py --before <released pair's json> --after <trained pair's json>

Both files come from rebuild_quality.py on the same rows of the same activations file. For each activation the
explanations with a closed tag are averaged to one MSE per pair. The difference in fve_nrm is
(mean MSE before - mean MSE after) / baseline. Its 95% range comes from resampling the activations with replacement
10,000 times (seed 0): what would change if a different set of activations had been drawn. Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import json
import random


def per_row(path: str) -> tuple[dict[int, float], float]:
    record = json.load(open(path))
    by_row = collections.defaultdict(list)
    for s in record["samples"]:
        if s["closed"]:
            by_row[s["row"]].append(s["mse"])
    return {row: sum(v) / len(v) for row, v in by_row.items()}, record["summary"]["baseline_mse"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    before, baseline = per_row(args.before)
    after, _ = per_row(args.after)
    rows = sorted(set(before) & set(after))
    gains = [(before[r] - after[r]) / baseline for r in rows]
    rng = random.Random(0)
    resampled = sorted(sum(rng.choices(gains, k=len(gains))) / len(gains) for _ in range(10_000))
    print(f"{len(rows)} activations, each pair's closed explanations averaged per activation")
    print(f"fve_nrm before {1 - sum(before[r] for r in rows) / len(rows) / baseline:.4f}, "
          f"after {1 - sum(after[r] for r in rows) / len(rows) / baseline:.4f}")
    print(f"difference {sum(gains) / len(gains):+.4f}, 95% range {resampled[250]:+.4f} to {resampled[9750]:+.4f}; "
          f"the trained pair is better on {sum(g > 0 for g in gains) / len(gains):.0%} of the activations")


if __name__ == "__main__":
    main()
