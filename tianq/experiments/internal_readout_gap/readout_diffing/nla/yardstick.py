"""The yardstick for NLA scoring: plain word counts against rarity-weighted word counts (18 Nov, exploratory).

    READOUT_OUT=<folder from shared/fetch.py> python yardstick.py

../jspace/similarity_methods.py ranks ways of scoring J-space overlap by how often the same restart point scores
above a different restart point of the same task, for base gemma against Qwen3.6. This applies the same yardstick to
the two ways the NLA sentences have been scored with words: the cosine of word counts (similarity.py), and the same
with each word weighted by rarity, log(texts / texts that have the word), pooled over Qwen3.6, base gemma and the
final checkpoint (root_cause.py). Both are the shared methods in ../methods/. A tie counts as half. Restart points
where a model's five sentences are all empty are left out. The meaning-vector and judge scores of the NLA are not
ranked here.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: evaluation, methods
import evaluation  # noqa: E402
from methods import rare_words, value_weighted  # noqa: E402
from readouts import BASE, FINAL, ORIGINAL, compare, load_counts  # noqa: E402
from restart_points import tasks  # noqa: E402


def main() -> None:
    _, counts, names = load_counts()
    names = [x for x in names if all(counts[t][x] for t in counts)]
    n = len(names)
    vocabulary = {w for t in counts for x in names for w in counts[t][x]}
    mask = evaluation.others(n) & evaluation.same_task(tasks(names))
    print(f"{n} restart points with NLA sentences from all three models; {len(vocabulary)} words", flush=True)
    for label, method, options in (("word counts, every word the same (the reported score)", value_weighted.cosine, {}),
                                   ("word counts, rare words count more", rare_words.rarity_cosine, {"top": None, "value": "p"})):
        for version in (BASE, FINAL):
            table = compare(method, counts, ORIGINAL, version, names, **options)
            strict, ties = evaluation.same_moment_rates(table, mask)
            print(f"{label} | {version} against {ORIGINAL}: same restart point scores above a different one of the same task "
                  f"{strict + ties / 2:.3f} of the time (ties {ties:.3f}); same {np.diag(table).mean():.4f}, "
                  f"chance {evaluation.chance(table).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
