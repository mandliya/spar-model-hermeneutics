"""Question 2 again, by meaning instead of identical words: does the gap shrink when readouts are meaning vectors?

    cd tianq && python -m experiments.internal_readout_gap.lie_detection_gap.meaning_gap --target gemma-3-27b-it --substitute qwen3.6-27b
    (as gap.py: the dyl-probe-repro venv, bundles from HF or READOUT_BUNDLES; HF_HOME holds the embedding model)

gap.py's word classifiers only see identical words, so a classifier trained on the target's words can miss the same
content in the substitute's different words (and languages). Here each readout becomes one meaning vector
(shared.embedding):
  J-lens meaning   the J-lens top-200 words' vectors, averaged
  NLA meaning      the NLA explanations' vectors, averaged (a row with no explanation gets the zero vector)
and the classifier is the arena's probe recipe on that vector (gap.fit). Everything else is gap.py's: the rows, the
positions, the labels, the splits, "target" / "substitute through the target's classifier" / "substitute with its own
classifier", the gap and its intervals. Added on 13 Nov after gap.py's results, to test whether the gap between two
models' readouts is a gap in wording. Prints every number it writes; writes
scratch/internal-readout-gap/meaning_gap_<target>__<substitute>.json.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.internal_readout_gap.lie_detection_gap.gap import (LABELS, LOCAL, MEANING_METHODS, POSITIONS, SPLITS,
                                                                    load_pair, report, score)
from experiments.internal_readout_gap.shared.embedding import EMBEDDER, Embedder

SOURCE = {"J-lens meaning": "J-lens words", "NLA meaning": "NLA sentences"}   # the texts each vector is averaged from


def vectors(emb: Embedder, read: dict, rows: list[dict], source: str, position: str) -> list[np.ndarray]:
    return [np.nan_to_num(emb.mean(read[r["uid"]][source][position])) for r in rows]


def main() -> None:
    target, substitute, pair, rows, t, s = load_pair()
    emb = Embedder()
    result: dict = {"target": target, "substitute": substitute, "rows": len(rows), "embedder": EMBEDDER, "gaps": {}}
    for method in MEANING_METHODS:
        source = SOURCE[method]
        if not all(source in m[r["uid"]] for m in (t, s) for r in rows):
            print(f"gap | {pair} | {method}: not computed, a model has no readout of this kind", flush=True)
            continue
        emb.add({text for m in (t, s) for r in rows for p in POSITIONS for text in m[r["uid"]][source][p]})
        for position in POSITIONS:
            ft, fs = (vectors(emb, m, rows, source, position) for m in (t, s))
            for label_name, labels in LABELS.items():
                for split in SPLITS:
                    tag = f"{pair} | {method} | {position} | {label_name} | {split}"
                    result["gaps"][tag] = report(tag, score(rows, ft, fs, method, labels, split))
    path = LOCAL / f"meaning_gap_{target}__{substitute}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
