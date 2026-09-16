"""STEP 2 — Pick two samples, and keep them apart.

  judge_dev.parquet   from train       -> step 3 only. Tune the judge here.
  eval.parquet        from validation  -> steps 4-6. The numbers you report.

Why two. You iterate JUDGE_PROMPT until agreement clears 85%. That is fitting
to data. If you tune on the same rows you later report from, your agreement
figure is partly memorisation and the lying rate inherits that bias. The
dataset's own train/validation split hands you the fix for free.

Both samples are balanced between did_lie True and False, as far as the config
allows: the headline is conditional on having lied, the control is conditional
on not having lied, and a lopsided sample starves one of them.

Run:  uv run python 02_sample.py --n 400 --judge-n 150
Cost: $0.
"""

import argparse

import pandas as pd

from src.common import DATA


def balanced(df, n, seed, name):
    per_arm = n // 2
    frames = []
    for label in [True, False]:
        sub = df[df["did_lie"] == label]
        take = min(per_arm, len(sub))
        if take < per_arm:
            print(f"  [{name}] only {len(sub)} rows with did_lie={label} "
                  f"— taking all of them")
        frames.append(sub.sample(take, random_state=seed))
    return pd.concat(frames).sample(frac=1, random_state=seed)


p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=400, help="eval set size")
p.add_argument("--judge-n", type=int, default=150, help="judge tuning set size")
p.add_argument("--seed", type=int, default=17)
args = p.parse_args()

out = {}
for split, size, fname in [("train", args.judge_n, "judge_dev"),
                           ("validation", args.n, "eval")]:
    df = pd.read_parquet(DATA / f"{split}.parquet")
    df["n_chars"] = df["trace"].apply(lambda t: sum(len(c) for c in t["content"]))
    before = len(df)
    df = df[df["n_chars"] < 12_000]
    if len(df) < before:
        print(f"[{fname}] dropped {before - len(df)} traces over 12k chars")

    sample = balanced(df, size, args.seed, fname)
    sample.to_parquet(DATA / f"{fname}.parquet")
    out[fname] = sample

    print(f"\n{fname}.parquet  (from {split})  n={len(sample)}")
    print(sample["did_lie"].value_counts().to_string())

# ---- guard: the two sets must not overlap ---------------------------------
overlap = set(out["judge_dev"]["sample_id"]) & set(out["eval"]["sample_id"])
if overlap:
    print(f"\n!! {len(overlap)} sample_ids appear in BOTH sets. The splits are")
    print("!! not disjoint in this config — your held-out claim is void.")
    print("!! Drop the overlap from eval before running step 4.")
else:
    print("\nNo overlap between the two sets. Held-out claim is clean.")

n_lies = int(out["eval"]["did_lie"].sum())
if n_lies < 60:
    print(f"\n!! Only {n_lies} lie-positive rows in eval. Your doubling-down")
    print("!! interval will be ±13 points or worse. Fine for a first look;")
    print("!! switch to a megafolds config for tighter numbers.")

est = out["eval"]["n_chars"].sum() / 3.5 * 2
print(f"\nRough estimate for steps 4-5: {est/1e6:.2f}M input tokens")
print(f"≈ ${est/1e6 * 0.08:.2f} on gemma-3-27b-it, plus judge calls")
