"""Control for step 3: is the distilled reader's rise only because its monitors are fitted on fewer city rows?

    cd tianq && PYTHONPATH=$PWD/dyl-probe-repro:$PWD python -m experiments.internal_readout_gap.gender_secret_rerun.same_prompts_control

The distilled reader's own city lies can be labelled only for the cities whose true country is recoverable
(label_own_lies.py), about 2,000 rows where the untouched reader has about 4,700. Here the untouched reader's probe
and J space are refitted on exactly the prompts the distilled reader's labelled rows come from (with AISI's labels),
and the gender rows (answer only) are scored again beside the fit on all its rows. Prints; writes nothing.
"""

from __future__ import annotations

import numpy as np

from experiments.hermeneutics import faithfulness
from experiments.internal_readout_gap.gender_secret_rerun import gap
from experiments.internal_readout_gap.gender_secret_rerun.rows import DISTILLED

MONITORS = ("probe at layer 36", "probe at layer 42", "J space")


def main() -> None:
    gender = gap.load_rows("gender_secret")
    own = gap.load_rows(f"city_lies_of_{DISTILLED}")
    city_rows = gap.load_rows(f"city_lies_of_{gap.READER}")
    city = gap.load_read(f"city_lies_of_{gap.READER}", "answer_only", gap.READER)
    test = gap.load_read("gender_secret", "answer_only", gap.READER)
    for position, label in gap.TRAIN_LABEL.items():
        same = {r["prompts_from"] for r in own.values() if r[label] is not None}
        for monitor in MONITORS:
            uids = sorted(u for u in test[position][monitor] if gender[u]["reply_lied"] is not None)
            y_test = np.array([gender[u]["reply_lied"] for u in uids])
            for name, keep in (("all its labelled city rows", None), ("only the prompts the distilled reader's labelled rows use", same)):
                train = sorted(u for u in city[position][monitor] if city_rows[u][label] is not None and (keep is None or u in keep))
                y = np.array([city_rows[u][label] for u in train])
                scorer = gap.fit(monitor, [city[position][monitor][u] for u in train], y)
                auroc = faithfulness.auroc(y_test, np.asarray(scorer([test[position][monitor][u] for u in uids]), dtype=float))
                print(f"untouched reader | answer_only | {position} | {monitor} | fitted on {name} ({len(y)} rows, {int(y.sum())} lie): "
                      f"AUROC {auroc:.3f}", flush=True)


if __name__ == "__main__":
    main()
