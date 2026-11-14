"""Question 1, sample level: at a few restart points, what happened in the run and what did each readout say?

    READOUT_OUT=<folder from shared/fetch.py> GIGI_TRANSCRIPTS=<empty folder> python example_points.py <point> [<point> ...]

For each named restart point of Gigi's hero swap this prints, in full and from saved files only:
  the run       the task, the last messages of the shared history, and the actual next turn each hero wrote from there
                in Gigi's continuations (the first CONTINUATIONS of each hero), with Gigi's bad-act label for that run
  the readouts  for Qwen3.6, base gemma and the final checkpoint: the model's own top next tokens, the J-lens words
                two-thirds of the way in and at 0.95, and every NLA description
The transcripts are downloaded one file at a time from Gigi's HF dataset at REVISION into GIGI_TRANSCRIPTS; the
readouts are the files similarity.py reads. Nothing is rerun or rebuilt. Added on 14 Nov with where_it_moved.py.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import spots  # noqa: E402
from shared import models, readout, words  # noqa: E402
from similarity import EXPERIMENT, ORIGINAL, VERSIONS  # noqa: E402

REPO = "model-hermeneutics/gigi-forensics-transcripts"
REVISION = "1aac2780007e0dc34553d0096142e2c3f3d98e57"
CONTINUATIONS = 2
TAIL = 3
WORDS = 30


def turn(m: dict) -> str:
    calls = "".join(f"\n    TOOL CALL {c['function']['name']} {c['function']['arguments']}" for c in m.get("tool_calls") or [])
    return (m.get("content") or "") + calls


def main() -> None:
    rows = list(csv.DictReader(open(spots.FORENSICS / "results" / "hero9_swap_per_run.csv", encoding="utf-8")))
    info = {s.name: s for s in spots.spots()}
    counts = spots.behaviour()
    for name in sys.argv[1:]:
        s = info[name]
        print(f"\n{'=' * 100}\npoint | {name}: task {s.env}; rewound from {s.src}'s own run after assistant turn {s.K}; Gigi's "
              f"counts of the bad act from here: {counts[name]}", flush=True)
        shown: dict[str, int] = {}
        for r in (r for r in rows if r["point"] == name):
            if shown.get(r["who"], 0) >= CONTINUATIONS:
                continue
            rel = (pathlib.Path(r["run"]).relative_to("results/raw") / "messages.json").as_posix()
            path = hf_hub_download(REPO, rel, repo_type="dataset", revision=REVISION, token=os.environ["HF_TOKEN"],
                                   local_dir=spots.TRANSCRIPTS)
            messages = json.loads(pathlib.Path(path).read_text())
            history = spots._prefix(messages, s.K)
            if history is None:
                continue
            if not shown:
                print(f"run | history: {len(history)} messages, {sum(m['role'] == 'assistant' for m in history)} assistant turns", flush=True)
                print(f"run | task, the first user message:\n{next(m['content'] for m in history if m['role'] == 'user')}", flush=True)
                for m in history[-TAIL:]:
                    print(f"run | history, one of the last {TAIL} messages [{m['role']}]:\n{turn(m)}", flush=True)
            shown[r["who"]] = shown.get(r["who"], 0) + 1
            print(f"run | next turn written by {r['who']} (Gigi's label for this run: bad act = {r['bad']}; {r['ending_or_loss']}) "
                  f"| {REPO}@{REVISION[:8]}:{rel}\n{turn(messages[len(history)])}", flush=True)
        for t in (ORIGINAL, VERSIONS[0], VERSIONS[-1]):
            r = json.loads((models.OUT / EXPERIMENT / t / "lens" / f"{name}.json").read_text())
            layer = {d: str(readout.layer_at(r["n_layers"], [int(k) for k in r["layers"]], d)) for d in (readout.HEADLINE, 0.95)}
            print(f"readout | {t} | own top next tokens: {words.lens_words(r['model'])[:WORDS]}", flush=True)
            for d, l in layer.items():
                print(f"readout | {t} | J-lens at depth {d:.3f} (layer {l} of {r['n_layers']}): {words.lens_words(r['layers'][l])[:WORDS]}", flush=True)
            n = json.loads((models.OUT / EXPERIMENT / t / "nla" / f"{name}.json").read_text())
            for i, x in enumerate(n["positions"]["before_action"], 1):
                print(f"readout | {t} | NLA description {i} of {len(n['positions']['before_action'])}:\n{x['explanation'].strip() or '(empty)'}", flush=True)


if __name__ == "__main__":
    main()
