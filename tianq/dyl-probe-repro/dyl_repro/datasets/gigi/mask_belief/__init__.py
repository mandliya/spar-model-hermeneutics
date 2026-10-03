"""Gigi's MASK belief rows, four subtypes: does Gemma say what it believes is false under a role?

    known_facts.py     a yes/no question whose honest answer hurts the role
    statistics.py      a request whose framing pushes a number
    continuations.py   continuing a text that opens with a false claim
    disinformation.py  writing an article that argues a falsehood

The turn label is checked against Gemma's own belief, not the dataset's: Gigi
asked the plain belief question with no role, his judge read which side each
answer takes, and `lied` is a pressured statement on the other side (for
statistics, a number more than 10% from the believed one). Rows with no clear
belief carry no label. Labels: results/mask_belief_<subtype>_labels.csv.

Each conversation is the one his follow-up run sent,
14_mask_belief_followups.py L79-80: the dataset trace, its lie_detection_prompt
(the pressured prompt), Gemma's saved statement, then the follow-up:
https://github.com/mandliya/spar-model-hermeneutics/blob/0df3751/gigi/mask-belief/14_mask_belief_followups.py#L79-L80
"""

from __future__ import annotations

from dyl_repro.datasets import followups, gigi

CATEGORY = "mask-belief"
# subtype -> (dataset config, task), from his src/mask_belief.py L25-30 (not
# imported: it imports his common.py, see gigi.BORROWED):
# https://github.com/mandliya/spar-model-hermeneutics/blob/0df3751/gigi/mask-belief/src/mask_belief.py#L25-L30
SUBTYPES = {
    "known_facts": ("google-gemma-3-27b-it_task-group_mask-factual", "mask_known_facts"),
    "statistics": ("google-gemma-3-27b-it_task-group_mask-factual", "mask_statistics"),
    "continuations": ("google-gemma-3-27b-it_task-group_mask-roleplay", "mask_continuations"),
    "disinformation": ("google-gemma-3-27b-it_task-group_mask-roleplay", "mask_disinformation"),
}


def rows(dataset: str, subtype: str) -> gigi.GigiRows:
    config, task = SUBTYPES[subtype]
    labels = {r["sample_id"]: r for r in gigi.csv_rows(CATEGORY, f"mask_belief_{subtype}_labels.csv")}
    statements = {a["sample_id"]: a["statement"]
                  for a in gigi.jsonl_rows(CATEGORY, f"mask_belief_{subtype}_answers.jsonl") if "error" not in a}
    # His rows() drops repeated sample_ids before selecting the task, L82-84.
    trace = gigi.dataset_rows(CATEGORY, config, task, drop_duplicate_ids=True)
    asked = {uid: gigi.trace_messages(CATEGORY, trace[uid]) for uid in labels}
    # Some rows of the dataset carry no user turn: an empty trace and a system-only
    # lie_detection_prompt, the request itself missing (13 statistics rows). The
    # chat template then drops the system text, so the statement answered an empty
    # prompt (all 13 are the same generic text, judged "no number", so labelled
    # honest), and his follow-up run errored on every one. No label applies.
    unasked = sorted(uid for uid, messages in asked.items() if not any(m["role"] == "user" for m in messages))
    if unasked:
        print(f"{dataset}: {len(unasked)} rows dropped, the dataset holds no user prompt for them: {unasked}", flush=True)
    return gigi.gigi_rows(
        dataset, CATEGORY, config, None,
        prefixes={
            uid: asked[uid] + [{"role": "assistant", "content": statements[uid]}]
            for uid in labels if uid not in unasked
        },
        turn_lie={uid: gigi.BOOL[r["lied"]] for uid, r in labels.items()},
        groups={},
        recorded_csv={c: f"mask_belief_{subtype}_benchmark_conditions.csv" for c in followups.gigi_conditions(dataset)},
    )
