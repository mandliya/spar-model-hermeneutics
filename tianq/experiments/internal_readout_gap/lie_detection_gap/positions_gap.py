"""Where to read for the lie detector (step 3b): the faithfulness gap at each of the 11 token positions.

    cd tianq && python -m experiments.internal_readout_gap.lie_detection_gap.positions_gap --target gemma-3-27b-it --substitute qwen3.6-27b
    (the dyl-probe-repro venv; reads the bundles of read_reply_positions.py's reads, from HF or READOUT_BUNDLES)

gap.py's own functions, run at every position read_reply_positions.py kept (8 through the tested reply, the reply's
end marker, before the answer to "Did you lie?", the answer token) instead of gap.py's two. Per position, label and
split: the J-lens words classifier (target, substitute through the target's classifier, substitute's own) and the
probe at the arena's probe layer (target, substitute's own), beside the words-only baseline, which has no position.
Every position is reported; none is chosen here. J space only: these reads have no NLA sentences.
Prints every number it writes; writes scratch/internal-readout-gap/positions_gap_<target>__<substitute>.json.
"""

from __future__ import annotations

import json

from experiments.internal_readout_gap.lie_detection_gap import gap

gap.EXPERIMENT = "lie_detection_positions"
METHODS = ("J-lens words", "probe")


def main() -> None:
    import gzip
    import sys
    target = sys.argv[sys.argv.index("--target") + 1]
    with gzip.open(gap.bundle(f"{target}__read_by_{target}", "words.json.gz"), "rt", encoding="utf-8") as f:
        gap.POSITIONS = tuple(json.load(f)["positions"])
    target, substitute, pair, rows, t, s = gap.load_pair()
    result: dict = {"target": target, "substitute": substitute, "rows": len(rows), "positions": list(gap.POSITIONS), "gaps": {}}
    text = ["\n".join(f"{m['role']}: {m['content']}" for m in r["messages"][:-2]) for r in rows]
    for label_name, labels in gap.LABELS.items():
        for split in gap.SPLITS:
            tag = f"{pair} | words only | {label_name} | {split}"
            result["gaps"][tag] = gap.report(tag, gap.score(rows, text, None, "words only", labels, split))
            for position in gap.POSITIONS:
                for method in METHODS:
                    ft, fs = ([m[r["uid"]][method][position] for r in rows] for m in (t, s))
                    tag = f"{pair} | {method} | {position} | {label_name} | {split}"
                    result["gaps"][tag] = gap.report(tag, gap.score(rows, ft, fs, method, labels, split))
    path = gap.LOCAL / f"positions_gap_{target}__{substitute}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
