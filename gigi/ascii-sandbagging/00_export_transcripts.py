"""Export full, readable transcripts — the actual rollouts, not just the
parsed letter.

`results/ascii_results.csv` and `results/ascii_followup_conditions.csv` only
ever saved gemma's raw answer (a single letter/number) plus a row id. The
question it was actually answering, and the conversation it sat inside, was
never saved anywhere — only reconstructable by rejoining the original
dataset. This script does that join once and writes the whole exchange back
out, per row, so it can actually be read and audited without re-fetching
anything.

Run:
    uv run python 00_export_transcripts.py

Cost: $0 — no API calls, just re-reading the dataset and joining against
results already on disk.
"""

import json

import pandas as pd

from src.common import (RESULTS, load_config, to_messages,
                        FOLLOWUP, FOLLOWUP_LOW_STAKES, FOLLOWUP_HIGH_STAKES,
                        FOLLOWUP_CONFIDENCE, FOLLOWUP_NEUTRAL_COT,
                        FOLLOWUP_LOW_STAKES_COT, FOLLOWUP_HIGH_STAKES_COT,
                        FOLLOWUP_CONFIDENCE_COT, FOLLOWUP_PLAIN,
                        FOLLOWUP_NAMED_SANDBAGGING)

CONFIG = "google-gemma-3-27b-it_task-group_sandbagging"

df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df[df["task"].str.contains("ascii", case=False, na=False)].set_index("sample_id")

FOLLOWUP_TEXT = {
    "neutral": FOLLOWUP,
    "low_stakes": FOLLOWUP_LOW_STAKES,
    "high_stakes": FOLLOWUP_HIGH_STAKES,
    "confidence": FOLLOWUP_CONFIDENCE,
    "neutral_cot": FOLLOWUP_NEUTRAL_COT,
    "low_stakes_cot": FOLLOWUP_LOW_STAKES_COT,
    "high_stakes_cot": FOLLOWUP_HIGH_STAKES_COT,
    "confidence_cot": FOLLOWUP_CONFIDENCE_COT,
    "plain": FOLLOWUP_PLAIN,
    "named": FOLLOWUP_NAMED_SANDBAGGING,
}

# --------------------------------------------------------- neutral run -----
base = pd.read_csv(RESULTS / "ascii_results.csv")
out_path = RESULTS / "transcripts.jsonl"
written = 0
with out_path.open("w") as f:
    for _, r in base.iterrows():
        row = df.loc[r["uid"]]
        msgs = to_messages(row["trace"])
        probe = dict(row["lie_detection_prompt"])
        transcript = msgs + [
            probe,
            {"role": "assistant", "content": r["probe_answer"]},
            {"role": "user", "content": FOLLOWUP_TEXT["neutral"]},
            {"role": "assistant", "content": r["followup_answer"]},
        ]
        f.write(json.dumps({
            "uid": r["uid"],
            "condition": "neutral",
            "lied": bool(r["lied"]) if pd.notna(r["lied"]) else None,
            "admitted": bool(r["admitted"]) if pd.notna(r["admitted"]) else None,
            "transcript": transcript,
        }, ensure_ascii=False) + "\n")
        written += 1
print(f"{out_path}: {written} neutral-condition transcripts")

# --------------------------------------------------- follow-up conditions --
fups = pd.read_csv(RESULTS / "ascii_followup_conditions.csv")
out_path2 = RESULTS / "transcripts_followups.jsonl"
written2 = 0
with out_path2.open("w") as f:
    for _, r in fups.iterrows():
        row = df.loc[r["sample_id"]]
        base_row = base[base["uid"] == r["sample_id"]].iloc[0]
        msgs = to_messages(row["trace"])
        probe = dict(row["lie_detection_prompt"])
        transcript = msgs + [
            probe,
            {"role": "assistant", "content": base_row["probe_answer"]},
            {"role": "user", "content": FOLLOWUP_TEXT[r["condition"]]},
            {"role": "assistant", "content": str(r["answer"])},
        ]
        f.write(json.dumps({
            "uid": r["uid"],
            "sample_id": r["sample_id"],
            "condition": r["condition"],
            "lied": bool(r["lied"]) if pd.notna(r["lied"]) else None,
            "admitted": bool(r["admitted"]) if pd.notna(r["admitted"]) else None,
            "confidence_score": (None if pd.isna(r["confidence_score"])
                                  else r["confidence_score"]),
            "transcript": transcript,
        }, ensure_ascii=False) + "\n")
        written2 += 1
print(f"{out_path2}: {written2} follow-up-condition transcripts")

# ------------------------------------------------------ reasoning conditions -
reasoning_path = RESULTS / "ascii_reasoning_conditions.csv"
if reasoning_path.exists():
    reas = pd.read_csv(reasoning_path)
    out_path3 = RESULTS / "transcripts_reasoning.jsonl"
    written3 = 0
    with out_path3.open("w") as f:
        for _, r in reas.iterrows():
            row = df.loc[r["sample_id"]]
            base_row = base[base["uid"] == r["sample_id"]].iloc[0]
            msgs = to_messages(row["trace"])
            probe = dict(row["lie_detection_prompt"])
            transcript = msgs + [
                probe,
                {"role": "assistant", "content": base_row["probe_answer"]},
                {"role": "user", "content": FOLLOWUP_TEXT[r["condition"]]},
                {"role": "assistant", "content": str(r["answer"])},
            ]
            f.write(json.dumps({
                "uid": r["uid"],
                "sample_id": r["sample_id"],
                "condition": r["condition"],
                "lied": bool(r["lied"]) if pd.notna(r["lied"]) else None,
                "admitted": bool(r["admitted"]) if pd.notna(r["admitted"]) else None,
                "confidence_score": (None if pd.isna(r.get("confidence_score"))
                                      else r["confidence_score"]),
                "reasoning": (None if pd.isna(r.get("reasoning")) else r["reasoning"]),
                "transcript": transcript,
            }, ensure_ascii=False) + "\n")
            written3 += 1
    print(f"{out_path3}: {written3} reasoning-condition transcripts")
else:
    print(f"\n{reasoning_path} not found yet — skipping transcripts_reasoning.jsonl "
          f"(run 00_ascii_reasoning.py first)")

# --------------------------------------------- self-report benchmark -------
benchmark_path = RESULTS / "ascii_benchmark_conditions.csv"
if benchmark_path.exists():
    bench = pd.read_csv(benchmark_path)
    out_path4 = RESULTS / "transcripts_benchmark.jsonl"
    written4 = 0
    with out_path4.open("w") as f:
        for _, r in bench.iterrows():
            row = df.loc[r["sample_id"]]
            base_row = base[base["uid"] == r["sample_id"]].iloc[0]
            msgs = to_messages(row["trace"])
            probe = dict(row["lie_detection_prompt"])
            transcript = msgs + [
                probe,
                {"role": "assistant", "content": base_row["probe_answer"]},
                {"role": "user", "content": FOLLOWUP_TEXT[r["condition"]]},
                {"role": "assistant", "content": str(r["answer"])},
            ]
            f.write(json.dumps({
                "uid": r["uid"],
                "sample_id": r["sample_id"],
                "condition": r["condition"],
                "lied": bool(r["lied"]) if pd.notna(r["lied"]) else None,
                "admitted": bool(r["admitted"]) if pd.notna(r["admitted"]) else None,
                "transcript": transcript,
            }, ensure_ascii=False) + "\n")
            written4 += 1
    print(f"{out_path4}: {written4} self-report-benchmark transcripts")
else:
    print(f"\n{benchmark_path} not found yet — skipping transcripts_benchmark.jsonl "
          f"(run 00_ascii_selfreport_benchmark.py first)")
