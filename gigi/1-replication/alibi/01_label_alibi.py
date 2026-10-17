"""Step 1 of games/alibi: pick the 500-row subset and label each suspect's
guilt from the persona prompt alone.

No GPU, no target-model calls. The judge reads two sections of the role-play
brief - the story the suspect is told to claim, and what actually happened -
and says whether they contradict. It never sees gemma's behaviour.

Why a judge rather than the dataset's tags: sample_ids carry "_lie_" /
"_truth_", but those disagree with a direct text comparison in ~290 of 1,281
rows, so they are not trusted - the same reason did_lie isn't.

The subset is 500 rows, drawn with a fixed seed and STRATIFIED by scenario
(fraud / theft / assault / workplace / breaking_entering) and by an
approximate guilty-vs-innocent flag, so it keeps the full set's proportions.
The approximate flag is a crude text match used only for stratifying; the
judge's label is what gets used downstream.

Run:
    python 01_label_alibi.py --limit 5     # smoke test + cost measurement
    python 01_label_alibi.py               # all 500
    python 01_label_alibi.py --rest        # the other 781 rows (results/alibi_rest.csv)

Resumable per sample_id; per-row failures are logged and skipped.
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.common import (RESULTS, MAX_WORKERS, load_config, load_done, append,
                        read_jsonl, alibi_claim_and_truth)
from src.judge import classify_alibi_guilt, ALIBI_GUILT_PROMPT

CONFIG = "google-gemma-3-27b-it_task-group_games"
N_SUBSET = 500
SEED = 0
OUT_PATH = RESULTS / "alibi_guilt_labels.jsonl"
SUBSET_PATH = RESULTS / "alibi_subset.csv"

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
p.add_argument("--rest", action="store_true",
               help="label every row NOT in the 500-row subset (added 1 Oct 2026)")
args = p.parse_args()

# ---------------------------------------------------------------- subset ---
df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df = df.drop_duplicates(subset=["sample_id"])
df = df[df["task"].str.startswith("alibi")].reset_index(drop=True)
df["system_prompt"] = df["trace"].apply(
    lambda t: next((c for r, c in zip(t["role"], t["content"])
                    if r == "system"), ""))
pairs = df["system_prompt"].apply(alibi_claim_and_truth)
df["claim"] = pairs.str[0]
df["truth"] = pairs.str[1]
df = df[df["claim"].notna() & df["truth"].notna()].copy()
df["scenario"] = df["task"].str.replace(r"^alibi(-multi)?_", "", regex=True)


def _core(x):
    x = x.lower().replace("actually ", "")
    x = re.split(r"witness:|evidence:|if pressed", x)[0]
    return re.sub(r"[^a-z0-9]", "", x)


df["approx_innocent"] = [_core(c) == _core(t) for c, t in zip(df.claim, df.truth)]

if SUBSET_PATH.exists():
    # Never redraw: the subset is fixed once chosen, so every later stage and
    # every rerun refers to exactly the same 500 rows.
    subset_ids = set(pd.read_csv(SUBSET_PATH)["sample_id"])
    sub = df[df["sample_id"].isin(subset_ids)].copy()
    print(f"using existing subset: {len(sub)} rows")
else:
    frac = N_SUBSET / len(df)
    # explicit per-stratum draw (groupby().apply drops the grouping columns
    # in current pandas, which silently breaks everything downstream)
    parts = [g.sample(frac=frac, random_state=SEED)
             for _, g in df.groupby(["scenario", "approx_innocent"])]
    sub = pd.concat(parts)
    if len(sub) > N_SUBSET:
        sub = sub.sample(n=N_SUBSET, random_state=SEED)
    elif len(sub) < N_SUBSET:
        extra = df[~df["sample_id"].isin(sub["sample_id"])].sample(
            n=N_SUBSET - len(sub), random_state=SEED)
        sub = pd.concat([sub, extra])
    sub[["sample_id", "task", "scenario", "approx_innocent"]].to_csv(
        SUBSET_PATH, index=False)
    print(f"drew stratified subset: {len(sub)} of {len(df)} rows (seed {SEED})")

if args.rest:
    sub = df[~df["sample_id"].isin(set(pd.read_csv(SUBSET_PATH)["sample_id"]))].copy()
    sub[["sample_id", "task", "scenario", "approx_innocent"]].to_csv(
        RESULTS / "alibi_rest.csv", index=False)
    print(f"--rest: the {len(sub)} rows outside the subset -> results/alibi_rest.csv")

print("\nsubset vs full set, by scenario:")
print(pd.concat([df.scenario.value_counts(normalize=True).rename("full"),
                 sub.scenario.value_counts(normalize=True).rename("subset")],
                axis=1).round(3).to_string())
print(f"approx innocent share: full {df.approx_innocent.mean():.3f}  "
      f"subset {sub.approx_innocent.mean():.3f}")

# ----------------------------------------------------------------- label ---
todo = sub.copy()
if args.limit:
    todo = todo.head(args.limit)
done = load_done(OUT_PATH, key="sample_id")
todo = todo[~todo["sample_id"].isin(done)]
print(f"\n{len(done)} already labelled, {len(todo)} to go")


def work(row):
    try:
        label, raw = classify_alibi_guilt(row["claim"], row["truth"])
    except Exception as e:
        append(OUT_PATH, {"sample_id": row["sample_id"], "error": str(e)})
        return
    append(OUT_PATH, {"sample_id": row["sample_id"], "scenario": row["scenario"],
                      "claim": row["claim"], "truth": row["truth"],
                      "guilt": label, "approx_innocent": bool(row["approx_innocent"]),
                      "judge_raw": raw})


if len(todo):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(tqdm(ex.map(work, [r for _, r in todo.iterrows()]), total=len(todo)))

# --------------------------------------------------------------- summary ---
out = pd.DataFrame(read_jsonl(OUT_PATH))
ok = out[out["guilt"].notna()] if "guilt" in out.columns else out.iloc[0:0]
print(f"\n{len(ok)} labelled, "
      f"{int(out['error'].notna().sum()) if 'error' in out.columns else 0} errors")
if len(ok):
    print(ok["guilt"].value_counts().to_string())
    print("\njudge vs crude text match (disagreement is expected on complex alibis):")
    print(pd.crosstab(ok["guilt"], ok["approx_innocent"]).to_string())

    # cost, measured from the saved text rather than estimated
    tin = sum(len(ALIBI_GUILT_PROMPT.format(claim=c, truth=t))
              for c, t in zip(ok.claim, ok.truth)) / 4
    tout = ok["judge_raw"].str.len().sum() / 4
    per = (tin / 1e6 * 3 + tout / 1e6 * 15) / len(ok)
    print(f"\nmeasured cost: ${per:.4f}/row -> ${per * N_SUBSET:.2f} for all "
          f"{N_SUBSET}")

    ok.drop(columns=["judge_raw"]).to_csv(RESULTS / "alibi_guilt_labels.csv",
                                          index=False)
