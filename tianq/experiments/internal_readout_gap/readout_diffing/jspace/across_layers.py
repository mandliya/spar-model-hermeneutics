"""Does reading the J-lens at several depths at once beat reading it at one? (26 Nov; gender_secret_rerun PLAN, step 0)

    READOUT_DISTRIBUTIONS=<folder of write_jlens_lists.py's files> python across_layers.py

The J space paper averages its concept score "across the workspace layers", and the tuned-lens paper uses a
prediction's path through the layers as the features. Here the two best single-depth methods (rare words count more,
top 5,000; the concept score) are run at the stored depths from half way to 85% of the model, and combined two ways:
  averaged   the mean of the depths' tables
  joined     one description per restart point: every depth's words, kept apart by depth, compared at once (the
             rare-word method only: the concept score's baseline belongs to one depth)
Same test as similarity_methods.py: for base gemma against Qwen3.6, published lenses, special tokens and non-words
removed, how often the same restart point scores above a different one of the same task. The single depths are
printed beside them. Writes <folder>/across_layers.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: method_menu, restart_points
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods, shared
from readout_methods.likeness import concept_score, evaluation, rare_words  # noqa: E402
from readout_methods.likeness.pair import Pair  # noqa: E402
from readouts import BASE, FINAL, FOLDER, QWEN, load_shared  # noqa: E402
from restart_points import tasks  # noqa: E402

DEPTHS = (0.5, 2 / 3, 0.75, 0.85)
TOP = 5000
METHODS = {"rare words count more, top 5,000": lambda pair: rare_words.rarity_cosine(pair, TOP),
           "concept score, top 5,000": concept_score.above_control_cosine}


def joined(by_depth: list[list]) -> list:
    """One readout per restart point from its readouts at several depths: each depth's first TOP words, tagged by depth."""
    return [([f"{w} @{k}" for k, (ws, _) in enumerate(point) for w in ws[:TOP]],
             np.concatenate([ps[:TOP] for _, ps in point])) for point in zip(*by_depth)]


def main() -> None:
    models = (QWEN, BASE, FINAL)   # the final checkpoint only as part of the pool the rare-word weights are counted over
    reads, names = load_shared([(m, "published") for m in models])
    mask = evaluation.others(len(names)) & evaluation.same_task(tasks(names))
    lists = {m: [reads[(m, "published")].lists(d, masked=True) for d in DEPTHS] for m in models}
    result = {}
    for label, method in METHODS.items():
        tables = [method(Pair(q, g, pool=[q, g, f])) for q, g, f in zip(*(lists[m] for m in models))]
        rates = {f"depth {d:.3f}": evaluation.same_moment_rates(t, mask)[0] for d, t in zip(DEPTHS, tables)}
        rates["averaged over the depths"] = evaluation.same_moment_rates(np.mean(tables, axis=0), mask)[0]
        if method is not concept_score.above_control_cosine:
            q, g, f = (joined(lists[m]) for m in models)
            rates["the depths joined"] = evaluation.same_moment_rates(rare_words.rarity_cosine(Pair(q, g, pool=[q, g, f]), None), mask)[0]
        result[label] = rates
        print(f"{label}: " + "; ".join(f"{k} {v:.3f}" for k, v in rates.items()), flush=True)
    (FOLDER / "across_layers.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {FOLDER / 'across_layers.json'}", flush=True)


if __name__ == "__main__":
    main()
