"""Exploration: the faithfulness gap with the probe at a third position, before the reader has written anything
(read_before_reasoning.py wrote the activations on the box and says why).

    cd tianq && PYTHONPATH=$PWD/dyl-probe-repro:$PWD python -m experiments.internal_readout_gap.gender_secret_rerun.gap_before_reasoning
    ... gap_before_reasoning --reader <tag>      # another reader in place of the untouched model

As in gap.py: for each reader the probe (dyl_repro's fit) is trained on its own city lies and applied unchanged to
the gender rows in both conditions; AUROC of lie against honest rows, and the organism's minus the reader's with its
interval. The reader has not answered yet at this token, so the probe is trained on each of the two city labels:
  reply_lied     the tested reply was a lie (the label of the gender rows too)
  followup_lie   AISI's: the yes or no the reader went on to give was a lie
Each probe's cross-validated AUROC on its city rows is printed first.
Writes scratch/.../gender_secret_rerun/gap_before_reasoning_<author>__<reader>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.hermeneutics import faithfulness
from experiments.internal_readout_gap.gender_secret_rerun import gap, transcript

CITY_LABELS = ("reply_lied", "followup_lie")


def load(rows_name: str, condition: str, model: str) -> tuple[dict[str, int], list[int], np.ndarray]:
    """(uid -> row of x, layers, x [rows, layers, d_model]) of one read."""
    saved = np.load(gap.hub(f"before_reasoning/{transcript.read_name(rows_name, condition, model)}.npz"))
    return {u: i for i, u in enumerate(saved["uids"].tolist())}, saved["layers"].tolist(), saved["x"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default=gap.READER)
    args = ap.parse_args()
    gender = gap.load_rows("gender_secret")
    uids = sorted(u for u in gender if gender[u]["reply_lied"] is not None)
    labels = [gender[u]["reply_lied"] for u in uids]
    scored: dict[str, dict[str, dict]] = {}
    result: dict = {"on_city_rows": {}, "gaps": {}}
    for who, model in {"organism": gap.AUTHOR, "reader": args.reader}.items():
        city_rows = gap.load_rows(f"city_lies_of_{model}")
        at, layers, x = load(f"city_lies_of_{model}", "answer_only", model)
        tests = {c: load("gender_secret", c, model) for c in transcript.CONDITIONS}
        for label in CITY_LABELS:
            train = sorted(u for u in at if city_rows.get(u, {}).get(label) is not None)
            y = np.array([city_rows[u][label] for u in train])
            for k, layer in enumerate(layers):
                monitor = f"probe at layer {layer}, trained on {label}"
                feats = [x[at[u], k] for u in train]
                cv = gap.cross_validated(monitor, feats, y)
                result["on_city_rows"][f"{who} | {monitor}"] = cv
                print(f"city rows | {who} ({model}) | {monitor}: cross-validated AUROC {cv:.3f} on {len(y)} rows ({int(y.sum())} lie)", flush=True)
                scorer = gap.fit(monitor, feats, y)
                for condition, (test_at, _, test_x) in tests.items():
                    scored.setdefault(f"{condition} | before_reasoning | {monitor}", {})[who] = {
                        "uids": uids, "label": labels, "score": np.asarray(scorer([test_x[test_at[u], k] for u in uids]), dtype=float).tolist()}
    for tag, by_who in scored.items():
        g = faithfulness.gap(by_who["organism"], by_who["reader"])
        result["gaps"][tag] = g | {"rows": len(uids)}
        print(f"gap | {tag}: organism {g['target']:.3f}, reader {g['substitute']:.3f}, gap {g['gap']:+.3f} "
              f"[{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] ({g['n_lie']} lie, {g['n_honest']} honest rows)", flush=True)
    path = gap.LOCAL / f"gap_before_reasoning_{gap.AUTHOR}__{args.reader}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
