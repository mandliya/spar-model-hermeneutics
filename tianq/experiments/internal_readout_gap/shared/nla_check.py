"""How often an NLA returned nothing, per read of an experiment.

    python nla_check.py --experiment lie_detection_gap

For every OUT/<experiment>/<read>/nla/: the number of explanations, how many are empty, how many end with a closing
tag, and how many activations have no explanation at all among their samples (those contribute no words).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    for d in sorted((models.OUT / args.experiment).glob("*/nla")):
        n = empty = closed = activations = wordless = 0
        for path in d.glob("*.json"):
            for samples in json.loads(path.read_text())["positions"].values():
                blank = [not s["explanation"].strip() for s in samples]
                n += len(blank); empty += sum(blank); closed += sum(s["closed"] for s in samples)
                activations += 1; wordless += all(blank)
        print(f"nla check | {d.parent.name}: {n} explanations, {empty} empty ({empty / n:.4f}), {closed} with a closing "
              f"tag ({closed / n:.4f}); {wordless} of {activations} activations have no explanation at all", flush=True)


if __name__ == "__main__":
    main()
