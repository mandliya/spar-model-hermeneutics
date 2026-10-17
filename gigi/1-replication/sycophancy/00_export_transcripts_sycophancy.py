"""Export full, readable transcripts for sycophancy, both tracks.

Track A: two INDEPENDENT calls per row (neutral, pressured), each its own
fresh conversation starting from the same system message — so this writes
two separate transcripts per row, not one growing one. (An earlier,
temperature-1.0 version of the underlying script asked the neutral
question twice; post temperature-0, there's only one neutral ask.)

Track B: one transcript per row (the framed prompt, already in the
dataset, plus gemma's critique).

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
            ("neutral", r["question"], r["neutral"]),
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
                "flipped": None if pd.isna(r["flipped"]) else bool(r["flipped"]),
                "aligned_with_user": (None if pd.isna(r["aligned_with_user"])
                                       else bool(r["aligned_with_user"])),
                "fold": None if pd.isna(r["fold"]) else bool(r["fold"]),
                "transcript": transcript,
            }, ensure_ascii=False) + "\n")
            written += 1

print(f"{out_path}: {written} transcripts ({written // 2} rows x 2 conditions each)")

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
            row = fb_df.loc[r["uid"]]
            msgs = to_messages(row["trace"])
            probe = dict(row["lie_detection_prompt"])
            transcript = msgs + [probe, {"role": "assistant", "content": str(r["critique"])}]
            f.write(json.dumps({
                "uid": r["uid"],
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
