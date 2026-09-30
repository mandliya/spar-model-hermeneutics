"""Pick the rows for the A/B swap test: up to 50 lied + 50 honest per category.

Balanced so confession and false confession both get enough rows; where a
category has fewer than 50 on one side, all of them are taken. Seed 0, drawn
once - results/ab_swap_subset.csv is the fixed list, never redrawn.

Ground truth is exactly the one the cross-category tables use
(make_condition_tables.py): the same truth column, the same row filters.

Run from the repo root (no GPU, no API calls):
    uv run --with pandas python gigi/ab_swap/00_pick_subset.py
"""

import pathlib

import pandas as pd

G = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "results" / "ab_swap_subset.csv"
N, SEED = 50, 0
B = lambda s: s.map({True: True, False: False, "True": True, "False": False})


def load(key):
    if key == "ascii":
        return pd.read_csv(G / "ascii-sandbagging/results/ascii_results.csv").rename(
            columns={"uid": "sample_id"}), "lied"
    if key == "sycophancy_a":
        return pd.read_csv(G / "sycophancy/results/sycophancy_benchmark_conditions.csv"), "fold"
    if key == "sycophancy_b":
        return pd.read_csv(G / "sycophancy/results/sycophancy_feedback_benchmark_conditions.csv"), "sycophantic"
    if key == "mask_factual":
        return pd.read_csv(G / "mask-factual/results/mask_factual_benchmark_conditions.csv"), "lied"
    if key == "unanswerable":
        d = pd.read_csv(G / "unanswerable/results/unanswerable_benchmark_conditions.csv")
        return d[d.bucket == "confabulation"], "confabulated"
    if key == "capture_secret":
        return pd.read_csv(G / "capture-the-secret/results/capture_secret_benchmark_conditions.csv",
                           low_memory=False), "lied"
    if key == "alibi":
        return pd.read_csv(G / "alibi/results/alibi_benchmark_conditions.csv", low_memory=False), "lied"


rows = []
for key in ["ascii", "sycophancy_a", "sycophancy_b", "mask_factual", "unanswerable",
            "capture_secret", "alibi"]:
    d, truth = load(key)
    d = d[d[truth].notna()].drop_duplicates("sample_id")
    d = d.assign(truth=B(d[truth]).astype(bool))[["sample_id", "truth"]]
    for t in (True, False):
        arm = d[d.truth == t].sort_values("sample_id")
        pick = arm.sample(n=min(N, len(arm)), random_state=SEED)
        rows.append(pick.assign(category=key))
        print(f"{key:15s} {'lied' if t else 'honest':6s} {len(pick):3d} of {len(arm)}")

out = pd.concat(rows)[["category", "sample_id", "truth"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    raise SystemExit(f"{OUT} already exists - the subset is fixed, not redrawn")
out.to_csv(OUT, index=False)
print(f"{len(out)} rows -> {OUT}")
