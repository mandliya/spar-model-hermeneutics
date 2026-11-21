"""Rank the similarity methods on the yardstick, then show what each says about the distillation (17 Nov).

    READOUT_DISTRIBUTIONS=<folder holding similarity_methods.json> python rank_methods.py

For each depth and lens: the methods ordered by how often base gemma's same-restart-point score beats a different
restart point of the same task (the harder yardstick; the all-points one is printed beside it), with the final
checkpoint's change from base gemma in units of its own spread. Reads similarity_methods.json only.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib

FOLDER = pathlib.Path(os.environ["READOUT_DISTRIBUTIONS"])


def main() -> None:
    methods = json.loads((FOLDER / "similarity_methods.json").read_text())["methods"]
    groups = collections.defaultdict(list)
    for tag, r in methods.items():
        depth, lens, masked, method = tag.split(" | ")
        groups[(depth, lens)].append((r["beats_same_task"], r["beats_other"], f"{method.strip()} ({masked})", r))
    for (depth, lens), rows in groups.items():
        print(f"== {depth}, {lens}")
        for same_task, other, name, r in sorted(rows, key=lambda x: x[0], reverse=True):
            star = "*" if r["ci"][0] > 0 or r["ci"][1] < 0 else " "
            print(f"   same task {same_task:.3f} | all {other:.3f} | change {r['change']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] {star} "
                  f"{r['change_in_sd']:+.2f} sd | {name}")


if __name__ == "__main__":
    main()
