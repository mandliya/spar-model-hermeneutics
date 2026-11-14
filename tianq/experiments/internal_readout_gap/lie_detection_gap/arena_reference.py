"""The arena's own stored results for the targets of question 2, printed so the report can set them beside the gap.

    cd tianq && python -m experiments.internal_readout_gap.lie_detection_gap.arena_reference

experiments/yuqi_questions/lie_detector_arena scored every lie detector on "did the target's reply lie" and stored
arena_<target>.json in the repo: the probes (read over the reply, and just before the "did you lie" answer), the
monitors (another model rates the transcript), the conversation's words, and just asking. Nothing is recomputed here.
Same kind = trained and tested inside a kind of lie; new kinds = trained on the most kinds the arena used, tested on
kinds left out. The arena's kinds are its own selection (9 per target), so these sit beside gap.py's numbers, not on
exactly the same rows.
"""

from __future__ import annotations

import json
import pathlib

ARENA = pathlib.Path(__file__).resolve().parents[2] / "yuqi_questions" / "lie_detector_arena"
TARGETS = ("gemma-3-27b-it", "qwen3-32b")
NAMES = {"reply": "probe over the reply", "did_you_lie": "probe before the 'did you lie' answer", "words": "words only",
         "asked": "just asking", "monitor_gemini-2.5-flash": "monitor, Gemini 2.5 Flash", "monitor_gpt-5-mini": "monitor, GPT-5 mini"}


def main() -> None:
    for target in TARGETS:
        arena = json.loads((ARENA / f"arena_{target}.json").read_text(encoding="utf-8"))
        kinds = len(arena["same_kind"]["kinds"])
        for method, name in NAMES.items():
            s = arena["same_kind"]["summary"][method]
            last = arena["new_kinds"]["curves"][method][-1]
            print(f"arena | {target} | {name}: same kind {s['mean']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] over {kinds} kinds; "
                  f"new kinds {last['mean']:.3f} [{last['lo']:.3f}, {last['hi']:.3f}] (trained on {last['k']} kinds)", flush=True)


if __name__ == "__main__":
    main()
