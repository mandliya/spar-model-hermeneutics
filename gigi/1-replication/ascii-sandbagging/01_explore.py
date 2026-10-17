"""STEP 1 — See what's in the dataset before spending a cent.

Lists every gemma-3-27b config in the repo, loads one, checks the labels are
sane, and dumps three full traces to a text file so you can read what a lie
actually looks like in this task group.

Run:  uv run python 01_explore.py
      uv run python 01_explore.py --config google-gemma-3-27b-it_task-group_sandbagging
      uv run python 01_explore.py --list        # just show available configs
Cost: $0. No API calls.
"""

import argparse
import json

from src.common import (DATA, CONFIG, list_configs, load_config,
                        sanity_check_labels, to_messages)

p = argparse.ArgumentParser()
p.add_argument("--config", default=CONFIG)
p.add_argument("--list", action="store_true")
args = p.parse_args()

print("Available gemma-3-27b configs in Noddybear/lies:\n")
for c in list_configs("gemma-3-27b"):
    print(f"  {c}")
print("\nThe task-group configs are small (tens to hundreds of rows) and map")
print("onto the paper's per-category numbers. The megafolds are the big mixed")
print("sets. Start with a megafold for volume; use task groups to compare")
print("against a specific figure in the paper.\n")

if args.list:
    raise SystemExit(0)

# Both splits. They exist because the paper TRAINED lie detectors on train and
# checked them on validation. You aren't training a detector — but you do tune
# a judge prompt, which is fitting. So: tune the judge on train, run the
# experiment on validation, and the number you report is honestly held out.
df = load_config(args.config, "train")
sanity_check_labels(df, f"{args.config} · train")

val = load_config(args.config, "validation")
sanity_check_labels(val, f"{args.config} · validation")

print(f"\nColumns: {list(df.columns)}")

for col in ["task", "task_id", "category", "model"]:
    if col not in df.columns:
        continue
    n = df[col].nunique()
    print(f"\n{col}: {n} distinct")
    if 1 < n <= 30:
        print(df.groupby(col)["did_lie"].agg(["mean", "count"])
                .sort_values("count", ascending=False).to_string())

# ---- read three actual traces ----------------------------------------------
out = DATA / f"traces_{args.config}.txt"
sample = df.sample(min(3, len(df)), random_state=0)
with out.open("w") as f:
    for _, row in sample.iterrows():
        msgs = to_messages(row["trace"])
        f.write(f"\n{'='*70}\nsample_id: {row['sample_id']}\n")
        f.write(f"did_lie: {row['did_lie']}   model: {row['model']}\n")
        f.write(f"task: {row.get('task')}  task_id: {row.get('task_id')}\n")
        f.write(f"{'='*70}\n")
        for m in msgs:
            f.write(f"\n--- {m['role'].upper()} ---\n{m['content']}\n")
        f.write("\n--- LIE DETECTION PROMPT ---\n"
                f"{json.dumps(dict(row['lie_detection_prompt']), indent=2)}\n")

print(f"\nWrote 3 full traces to {out}")
print("READ THAT FILE. You need to know what counts as a lie in this task")
print("group before the judge prompt in step 3 can be anything but a guess.")

df.to_parquet(DATA / "train.parquet")
val.to_parquet(DATA / "validation.parquet")
print(f"\nCached both splits to {DATA}/")
print("  train.parquet       -> judge tuning only (step 3)")
print("  validation.parquet  -> the experiment (steps 4-6)")
