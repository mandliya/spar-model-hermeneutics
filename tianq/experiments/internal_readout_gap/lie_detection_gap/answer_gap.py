"""The faithfulness gap for a label the conversation's words cannot give: what the target is about to answer.

    cd tianq && READOUT_BUNDLES=dyl-probe-repro/scratch/internal-readout-gap/bundles \
        python -m experiments.internal_readout_gap.lie_detection_gap.answer_gap --target gemma-3-27b-it --substitute qwen3.6-27b
    (the dyl-probe-repro venv; reads what gap.py reads; CPU only)

Same rows, readings, classifiers, splits and bootstrap as gap.py; only the label differs. Here the label is the
target's own answer to "did you lie?": said_yes = 1 when it answered yes (it lied and did not deny it, or it did not
lie and said it did). Both models are read at the token BEFORE that answer, so the answer is in nobody's input: the
target's state holds what the target is about to say, and the substitute, prefilled with the same conversation, holds
what the substitute would say. unseen_type_diagnosis.py found the target's J-lens and NLA words give this answer at
0.96 AUROC on kinds of lie they were not trained on, where the conversation's words give 0.52.

    gap = target AUROC - substitute AUROC, both against the target's answer

Prints every number it writes; writes scratch/internal-readout-gap/answer_gap_<target>__<substitute>.json.
"""

from __future__ import annotations

import json

from experiments.internal_readout_gap.lie_detection_gap import gap

LABEL = "said_yes"
POSITION = gap.POSITIONS[0]   # before_answer


def main() -> None:
    target, substitute, pair, rows, t, s = gap.load_pair()
    for r in rows:
        r[LABEL] = None if r["reply_lied"] is None or r["followup_lie"] is None else int(r["reply_lied"] != r["followup_lie"])
    answered = [r for r in rows if r[LABEL] is not None]
    print(f"rows | {pair}: {len(answered)} rows with a judged answer; the target said yes in {sum(r[LABEL] for r in answered)}", flush=True)
    result: dict = {"target": target, "substitute": substitute, "label": LABEL, "position": POSITION, "rows": len(answered), "gaps": {}}
    text = ["\n".join(f"{m['role']}: {m['content']}" for m in r["messages"][:-2]) for r in rows]
    for split in gap.SPLITS:
        tag = f"{pair} | words only | {split}"
        result["gaps"][tag] = gap.report(tag, gap.score(rows, text, None, "words only", (LABEL, LABEL), split))
        for method in (*gap.WORD_METHODS, "probe"):
            if not all(method in t[r["uid"]] and method in s[r["uid"]] for r in rows):
                print(f"gap | {pair} | {method}: not computed, a model has no readout of this kind", flush=True)
                continue
            ft, fs = ([m[r["uid"]][method][POSITION] for r in rows] for m in (t, s))
            tag = f"{pair} | {method} | {split}"
            result["gaps"][tag] = gap.report(tag, gap.score(rows, ft, fs, method, (LABEL, LABEL), split))
    path = gap.LOCAL / f"answer_gap_{target}__{substitute}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
