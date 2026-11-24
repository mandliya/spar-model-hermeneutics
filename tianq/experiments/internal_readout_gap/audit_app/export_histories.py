"""On a machine that has Gigi's transcripts: write every restart point's history, and each model's next turn, as one file.

    python export_histories.py --to /workspace/histories.json.gz

For each point of the hero swap: the messages before assistant turn K+1 (the history every model was given,
spots._prefix, from the first continuation that wrote a new turn), and the first new turn Qwen3.6 and gemma each
wrote from there (their first continuation in Gigi's hero9_swap_per_run.csv that has one). Messages are kept as they
are in the transcript. The audit page (build_histories.py) lays them out to read.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "distillation_similarity"))
import spots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    args = ap.parse_args()
    runs: dict = {}
    for row in csv.DictReader(open(spots.FORENSICS / "results" / "hero9_swap_per_run.csv", encoding="utf-8")):
        runs.setdefault(row["point"], {}).setdefault(row["who"], []).append(pathlib.PurePosixPath(row["run"]).relative_to("results/raw").as_posix())
    out = {}
    for s in spots.spots():
        entry = {"history": None, "history_from": None, "next": {}}
        for who, paths in runs.get(s.name, {}).items():
            for rel in paths:
                path = spots.TRANSCRIPTS / rel / "messages.json"
                if not path.exists():
                    continue
                messages = json.loads(path.read_text())
                turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
                if len(turns) > s.K + 1:
                    if entry["history"] is None:
                        entry["history"], entry["history_from"] = messages[: turns[s.K + 1]], rel
                    entry["next"][who] = {"from": rel, "turn": messages[turns[s.K + 1]]}
                    break
        out[s.name] = entry
    with gzip.open(args.to, "wt", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{len(out)} points; {sum(1 for e in out.values() if e['history'])} with a history; wrote {args.to}", flush=True)


if __name__ == "__main__":
    main()
