"""Put each row's activation filenames into the results CSV that row lives in.

capture-the-secret and unanswerable already write both paths per row, because
their capture happened inside the generation script. ascii-sandbagging and
sycophancy were captured later, by separate recapture scripts, which wrote the
row -> filename mapping into their own JSONL files instead. The tensors exist
either way; the mapping just wasn't where anyone reading the results would
look for it.

That matters now that these files carry `dyl_is_lie`: anyone training a probe
needs label and tensor together, and having to discover a second file to join
against is exactly the kind of friction that produces a silently wrong join.

Adds `predecision_activations_file` and `answer_token_activations_file` to the
ascii and sycophancy benchmark CSVs. Nothing existing is modified, and rows
with no captured tensor are left blank rather than pointed at a file that
isn't there.

Note the sycophancy maps key on "<track>::<sample_id>::<condition>" while the
CSVs key on "<sample_id>::<condition>", so the track prefix is stripped before
joining - a straight merge on uid silently matches nothing.

Run from the repo root:
    uv run --with pandas python gigi/join_activation_paths.py
"""

import json
import pathlib

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent

JOBS = [
    # (csv, [map jsonl, ...])
    ("ascii-sandbagging/results/ascii_benchmark_conditions.csv",
     ["ascii-sandbagging/results/dyl_answer_token_ascii.jsonl",
      "ascii-sandbagging/results/predecision_ascii_benchmark.jsonl"]),
    ("ascii-sandbagging/results/ascii_reasoning_conditions.csv",
     ["ascii-sandbagging/results/dyl_answer_token_ascii.jsonl"]),
    ("ascii-sandbagging/results/ascii_followup_conditions.csv",
     ["ascii-sandbagging/results/dyl_answer_token_ascii.jsonl"]),
    ("sycophancy/results/sycophancy_benchmark_conditions.csv",
     ["sycophancy/results/dyl_answer_token_sycophancy.jsonl",
      "sycophancy/results/predecision_sycophancy_battery.jsonl"]),
    ("sycophancy/results/sycophancy_feedback_benchmark_conditions.csv",
     ["sycophancy/results/dyl_answer_token_sycophancy.jsonl",
      "sycophancy/results/predecision_sycophancy_battery.jsonl"]),
    ("mask-factual/results/mask_factual_benchmark_conditions.csv",
     ["mask-factual/results/mask_factual_activations.jsonl"]),
    ("mask-factual/results/mask_factual_named_rerun.csv",
     ["mask-factual/results/mask_factual_activations.jsonl"]),
]

COLS = ["answer_token_activations_file", "predecision_activations_file"]


def load_map(path):
    """uid -> {column: filename}. The sycophancy maps prefix uid with the
    track ("A::" / "B::"); the CSVs don't, so strip it."""
    out = {}
    for line in pathlib.Path(path).open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "error" in r:
            continue
        uid = str(r.get("uid", ""))
        if uid[:3] in ("A::", "B::"):
            uid = uid[3:]
        vals = {c: r[c] for c in COLS if r.get(c)}
        if uid and vals:
            out.setdefault(uid, {}).update(vals)
    return out


for rel_csv, rel_maps in JOBS:
    csv = REPO / rel_csv
    if not csv.exists():
        print(f"!! missing csv, skipped: {rel_csv}")
        continue
    df = pd.read_csv(csv, low_memory=False)

    mapping = {}
    for m in rel_maps:
        p = REPO / m
        if not p.exists():
            print(f"!! missing map, skipped: {m}")
            continue
        for uid, vals in load_map(p).items():
            mapping.setdefault(uid, {}).update(vals)

    for col in COLS:
        df[col] = df["uid"].map(lambda u: mapping.get(str(u), {}).get(col))

    # ascii's followup/reasoning files predate the two-column convention and
    # store their pre-decision capture in a legacy `activations_file` column.
    # It is the same capture point - every layer's hidden state at the last
    # prompt token - so copy it across rather than leaving the column blank
    # and implying no tensor exists.
    if "activations_file" in df.columns:
        blank = df["predecision_activations_file"].isna()
        df.loc[blank, "predecision_activations_file"] = df.loc[blank, "activations_file"]

    # mask-factual's original benchmark file still holds the superseded
    # named/eval_named rows, which sent the `plain` wording by mistake. Their
    # tensors were captured from the RE-RUN answers (the real vocabulary), so
    # attaching them here would pair a label with a tensor from a different
    # prompt. Those two conditions live authoritatively in
    # mask_factual_named_rerun.csv; blank them here.
    if rel_csv.endswith("mask_factual_benchmark_conditions.csv"):
        superseded = df["condition"].isin(["named", "eval_named"])
        df.loc[superseded, COLS] = None

    df.to_csv(csv, index=False)
    at = int(df["answer_token_activations_file"].notna().sum())
    pdz = int(df["predecision_activations_file"].notna().sum())
    print(f"{rel_csv.split('/')[-1]:48s} n={len(df):5d}  "
          f"answer_token={at:5d}  predecision={pdz:5d}")

print("\nBlank means no tensor was captured for that row - not a broken path.")
