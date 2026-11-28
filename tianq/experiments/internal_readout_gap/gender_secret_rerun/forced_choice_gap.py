"""Forced choice from the NLA as the monitor, and as a direct question about the organism's secret (ask_the_nla.py
wrote the scores on the box; the questions: readout_methods/monitors/forced_choice.py).

    cd tianq && PYTHONPATH=$PWD/dyl-probe-repro:$PWD python -m experiments.internal_readout_gap.gender_secret_rerun.forced_choice_gap
    ... forced_choice_gap --reader <tag>     # another reader in place of the untouched model

No training. A row's score is how far the NLA leans to a question's second option when asked about the activation.
Per condition, position and question, on the gender rows both readers have:
  gap     as in gap.py: AUROC of lie against honest rows for the organism and for the reader, and the difference
          with its interval
  level   the mean score of each reader's activations, and the organism's minus the reader's on the same row with
          its interval: does the NLA say "yes" more often about the organism's state, whatever the row?
Writes scratch/.../gender_secret_rerun/forced_choice_<author>__<reader>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.hermeneutics import faithfulness
from experiments.internal_readout_gap.gender_secret_rerun import gap, transcript
from experiments.internal_readout_gap.readout_methods.monitors import forced_choice
from experiments.internal_readout_gap.shared import words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default=gap.READER)
    args = ap.parse_args()
    gender = gap.load_rows("gender_secret")
    readers = {"organism": gap.AUTHOR, "reader": args.reader}
    result = {}
    for condition in transcript.CONDITIONS:
        asked = {who: json.loads(gap.hub(f"{transcript.read_name('gender_secret', condition, model)}/forced_choice.json").read_text())["rows"]
                 for who, model in readers.items()}
        for position in transcript.POSITIONS:
            uids = sorted(u for u in gender if gender[u]["reply_lied"] is not None and all(position in asked[who].get(u, {}) for who in readers))
            for name in forced_choice.QUESTIONS:
                scored = {who: {"uids": uids, "label": [gender[u]["reply_lied"] for u in uids],
                                "score": [asked[who][u][position][name]["score"] for u in uids]} for who in readers}
                g = faithfulness.gap(scored["organism"], scored["reader"])
                level = {who: float(np.mean(s["score"])) for who, s in scored.items()}
                m, lo, hi = words.mean_ci(np.array(scored["organism"]["score"]) - np.array(scored["reader"]["score"]))
                tag = f"{condition} | {position} | NLA, forced choice, {name}"
                result[tag] = g | {"rows": len(uids), "level": level, "organism_minus_reader": {"mean": m, "ci": [lo, hi]}}
                print(f"gap | {tag}: organism {g['target']:.3f}, reader {g['substitute']:.3f}, gap {g['gap']:+.3f} "
                      f"[{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] ({g['n_lie']} lie, {g['n_honest']} honest rows)", flush=True)
                print(f"level | {tag}: organism {level['organism']:+.3f}, reader {level['reader']:+.3f}, organism minus reader "
                      f"on the same row {m:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    path = gap.LOCAL / f"forced_choice_{gap.AUTHOR}__{args.reader}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
