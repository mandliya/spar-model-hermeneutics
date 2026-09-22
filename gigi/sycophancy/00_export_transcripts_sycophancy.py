"""Export full, readable transcripts for sycophancy track A (answer-conformity).

Same motivation as 00_export_transcripts.py, but for a different dataset
config and a different conversation shape — sycophancy_answer_conditions.csv
only saves gemma's raw answers, not the questions or the trace they sat in.

Track A isn't one chained follow-up like the sandbagging conditions are —
it's three INDEPENDENT calls per row (neutral_1, neutral_2, pressured), each
its own fresh conversation starting from the same system message. So this
writes three separate transcripts per row, not one growing one.

Run:
    uv run python 00_export_transcripts_sycophancy.py

Cost: $0 — no API calls, just re-reading the dataset and joining against
results already on disk.
"""

import json

import pandas as pd

from src.common import RESULTS, load_config, to_messages

CONFIG = "google-gemma-3-27b-it_task-group_sycophancy"

df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"].str.startswith("sycophancy_answer")].set_index("sample_id")

res = pd.read_csv(RESULTS / "sycophancy_answer_conditions.csv")
out_path = RESULTS / "transcripts_sycophancy_answers.jsonl"
written = 0
with out_path.open("w") as f:
    for _, r in res.iterrows():
        row = df.loc[r["uid"]]
        msgs = to_messages(row["trace"])
        probe = dict(row["lie_detection_prompt"])

        for condition, question, answer in [
            ("neutral_1", r["question"], r["neutral_1"]),
            ("neutral_2", r["question"], r["neutral_2"]),
            ("pressured", probe["content"], r["pressured"]),
        ]:
            transcript = msgs + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(answer)},
            ]
            f.write(json.dumps({
                "uid": f"{r['uid']}::{condition}",
                "sample_id": r["uid"],
                "condition": condition,
                "task": r["task"],
                "user_belief": r["user_belief"],
                "neutral_stable": bool(r["neutral_stable"]),
                "flipped": None if pd.isna(r["flipped"]) else bool(r["flipped"]),
                "aligned_with_user": (None if pd.isna(r["aligned_with_user"])
                                       else bool(r["aligned_with_user"])),
                "fold": None if pd.isna(r["fold"]) else bool(r["fold"]),
                "transcript": transcript,
            }, ensure_ascii=False) + "\n")
            written += 1

print(f"{out_path}: {written} transcripts ({written // 3} rows x 3 conditions each)")

# ------------------------------------------------------ track B: feedback --
feedback_path = RESULTS / "sycophancy_feedback_conditions.csv"
if feedback_path.exists():
    fb_df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
    fb_df = fb_df[fb_df["task"].str.startswith("sycophancy_feedback")].set_index("sample_id")

    fb = pd.read_csv(feedback_path)
    out_path2 = RESULTS / "transcripts_sycophancy_feedback.jsonl"
    written2 = 0
    with out_path2.open("w") as f:
        for _, r in fb.iterrows():
            row = fb_df.loc[r["sample_id"]]
            msgs = to_messages(row["trace"])
            probe = dict(row["lie_detection_prompt"])
            transcript = msgs + [probe, {"role": "assistant", "content": str(r["critique"])}]
            f.write(json.dumps({
                "uid": r["uid"],
                "sample_id": r["sample_id"],
                "kind": r["kind"],
                "task": r["task"],
                "argument_group_id": r["argument_group_id"],
                "sentiment_score": (None if pd.isna(r["sentiment_score"])
                                    else int(r["sentiment_score"])),
                "transcript": transcript,
            }, ensure_ascii=False) + "\n")
            written2 += 1
    print(f"{out_path2}: {written2} track-B (feedback) transcripts")
else:
    print(f"\n{feedback_path} not found yet — skipping transcripts_sycophancy_feedback.jsonl "
          f"(run 00_sycophancy_feedback.py first)")
