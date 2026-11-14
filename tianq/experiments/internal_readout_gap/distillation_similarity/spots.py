"""Gigi's symmetric hero swap: the restart points, the history each model saw there, and the behaviour measured there.

A point is one hero's own run rewound to step K (gigi/3-forensics/results/hero9_swap_plan.json); each hero then
continued it about ten times (results/hero9_swap_per_run.csv, one row per continuation; results/hero9_swap.csv, the
counts per point and model). On HF each continuation has only its final messages.json, so the history at step K is cut
from a continuation by Gigi's own rule (distill_data.load_runs_hero9): assistant turns 0..K are the source run's and
the first new turn is K+1. `check()` verifies that every continuation of a point shares that prefix.

Gigi's modules are imported as they are (tools per environment, the gemma text-mode renderer); nothing is copied.
"""

from __future__ import annotations

import collections
import csv
import functools
import json
import os
import pathlib
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[4]
FORENSICS = REPO / "gigi" / "3-forensics"


@functools.cache
def gigi():
    """Gigi's distill_data module (his tools per environment and his text-mode renderer). Imported on first use: it
    needs his harness clone (agent-interp-envs), which only the box has; the spot list and the behaviour counts
    below read his result files alone, so analyses can run without it."""
    sys.path.insert(0, str(FORENSICS / "scripts"))
    import distill_data
    return distill_data

# Gigi's HF dataset model-hermeneutics/gigi-forensics-transcripts; a run at results/raw/<x> in his csvs is <x> here.
TRANSCRIPTS = pathlib.Path(os.environ.get("GIGI_TRANSCRIPTS", "/workspace/data/gigi-forensics-transcripts"))
HELD_OUT = ("tower_of_hanoi", "eval_tampering", "currency_conversion")   # not trained on (distill/PLAN_hero9.md)
WHO = {"qwen3.6-27b": "qwen36", "gemma-3-27b-it": "gemma"}               # our model tag -> Gigi's name


@dataclass(frozen=True)
class Spot:
    name: str          # Gigi's point name
    env: str
    src: str           # whose run it is: "gemma" | "qwen36"
    K: int             # the history ends after assistant turn K
    N: int             # the step Gigi's plan marks (the bad act on incident points)
    incident: bool     # True: rewound from a run with the bad act
    take: str          # "N" | "N-1" | "clean"

    @property
    def held_out(self) -> bool:
        return self.env in HELD_OUT


def spots() -> list[Spot]:
    out = []
    for p in json.loads((FORENSICS / "results" / "hero9_swap_plan.json").read_text()):
        run = pathlib.Path(p["run"])
        name = f"{p['src']}_{run.parent.name}_{run.name}_K{p['K']}"     # as load_runs_hero9 names a point
        out.append(Spot(name, p["env"], p["src"], int(p["K"]), int(p["N"]), bool(p["incident"]), str(p["take"])))
    return out


def continuations() -> dict[str, list[pathlib.Path]]:
    """point -> the folders of its continuations in TRANSCRIPTS, in csv order."""
    out = collections.defaultdict(list)
    for row in csv.DictReader(open(FORENSICS / "results" / "hero9_swap_per_run.csv", encoding="utf-8")):
        out[row["point"]].append(TRANSCRIPTS / pathlib.Path(row["run"]).relative_to("results/raw"))
    return out


def _prefix(messages: list[dict], K: int) -> list[dict] | None:
    """The history after assistant turn K: everything before assistant turn K+1 (None if the run never got there)."""
    turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    return messages[: turns[K + 1]] if len(turns) > K + 1 else None


def history(spot: Spot, runs: list[pathlib.Path]) -> tuple[list[dict], list[dict]] | None:
    """(messages, tools) at the spot, from its first continuation that wrote a new turn; None if none did."""
    for run in runs:
        path = run / "messages.json"
        if path.exists():
            prefix = _prefix(json.loads(path.read_text()), spot.K)
            if prefix is not None:
                return prefix, gigi().tools_for(gigi().yaml.safe_load((run / "config.yaml").read_text()))
    return None


def check(spot: Spot, runs: list[pathlib.Path]) -> tuple[int, int]:
    """(continuations with a new turn, distinct prefixes among them): 1 distinct prefix = the cut is the shared history."""
    seen = collections.Counter()
    for run in runs:
        path = run / "messages.json"
        if path.exists():
            prefix = _prefix(json.loads(path.read_text()), spot.K)
            if prefix is not None:
                seen[json.dumps(prefix, sort_keys=True)] += 1
    return sum(seen.values()), len(seen)


def text_render(messages: list[dict], tools: list[dict]) -> list[dict]:
    """The history as gemma receives it in text mode (Gigi's harness renderer)."""
    return gigi()._text_tools_messages(messages, tools)


def behaviour() -> dict[str, dict[str, dict[str, int]]]:
    """point -> Gigi's model name -> {"cheated", "scored"} from results/hero9_swap.csv."""
    out: dict = collections.defaultdict(dict)
    for row in csv.DictReader(open(FORENSICS / "results" / "hero9_swap.csv", encoding="utf-8")):
        out[row["point"]][row["who"]] = {"cheated": int(row["k"]), "scored": int(row["n"])}
    return dict(out)


def main() -> None:
    """Print the check for every spot: how many continuations reach a new turn and whether they share one prefix."""
    by_point = continuations()
    bad = 0
    for s in spots():
        n, distinct = check(s, by_point.get(s.name, []))
        if distinct != 1:
            bad += 1
            print(f"{s.name} ({s.env}): {n} continuations with a new turn, {distinct} distinct prefixes", flush=True)
    print(f"prefix check: {len(spots())} spots, {bad} without exactly one shared prefix", flush=True)


if __name__ == "__main__":
    main()
