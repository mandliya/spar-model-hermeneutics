"""Stage 1 of unanswerable-claims: label each generated QUESTION.

No GPU, no generation, no target-model calls at all — this only reads the
questions gemma already generated (recorded in the dataset's trace, the one
config in this release that has a real assistant turn) and asks the JUDGE to
characterise each one.

Why this stage exists: a spot check of 14 real rows showed gemma did NOT
reliably do what it was asked. Roughly 70% of the questions are genuinely
undecidable, but ~25% are plainly answerable (the infinite nested radical is
phi; the Dogon question is Sirius B; a California non-compete is void under
B&P Code 16600), and a few are answerable-in-principle with no correct option
listed. Those are three different situations and mixing them into one lying
rate would average two different phenomena, so they get split here, before
any GPU time is spent.

The judge NEVER judges the lie in this category — the lie is behavioural
(claimed it couldn't answer, then answered). The judge only labels questions.

Run:
    python 01_label_unanswerable_questions.py
    python 01_label_unanswerable_questions.py --limit 20   # smoke test

Resumable: appends to results/unanswerable_question_labels.jsonl and skips
sample_ids already in it. Per-row failures are logged and skipped, never
fatal — one bad row must not take down a 604-row run.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.common import (RESULTS, MAX_WORKERS, load_config, load_done, append,
                        read_jsonl, generated_question)
from src.judge import classify_question_defensibility

CONFIG = "google-gemma-3-27b-it_task-group_unanswerable"
OUT_PATH = RESULTS / "unanswerable_question_labels.jsonl"

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()

# --------------------------------------------------------------- inputs ----
df = pd.concat([load_config(CONFIG, "train"), load_config(CONFIG, "validation")])
df["domain"] = df["task"].str.replace("unanswerable-claims_", "", regex=False)
df["question"] = df["trace"].apply(generated_question)

# Drop rows with no recorded question rather than guessing what was asked.
n_empty = int((df["question"].str.len() == 0).sum())
if n_empty:
    print(f"!! {n_empty} rows have no recorded assistant turn (no question to "
          f"label) — dropping them, not guessing")
    df = df[df["question"].str.len() > 0].copy()

df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)
print(f"{len(df)} questions to label  ({df['domain'].nunique()} domains)")

done = load_done(OUT_PATH, key="sample_id")
todo = df[~df["sample_id"].isin(done)]
print(f"{len(done)} already done, {len(todo)} to go")


def work(row):
    """One judge call. Any failure is logged as a row and swallowed — a single
    bad response must never abort the run."""
    try:
        label, correct, raw = classify_question_defensibility(row["question"])
    except Exception as e:
        append(OUT_PATH, {"sample_id": row["sample_id"],
                          "domain": row["domain"], "error": str(e)})
        return
    append(OUT_PATH, {
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "task": row["task"],
        "question": row["question"],
        "label": label,              # None means unparseable — excluded, not guessed
        "correct_option": correct,   # only ever set when label == DEFENSIBLE
        "judge_raw": raw,
    })


if len(todo):
    rows = [r for _, r in todo.iterrows()]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(tqdm(ex.map(work, rows), total=len(rows)))

# --------------------------------------------------------------- summary ---
recs = read_jsonl(OUT_PATH)
out = pd.DataFrame(recs)
n_err = int(out["error"].notna().sum()) if "error" in out.columns else 0
ok = out[out["label"].notna()] if "label" in out.columns else out.iloc[0:0]

print(f"\n{len(ok)} labelled, {n_err} errors, "
      f"{len(out) - len(ok) - n_err} unparseable")

print("\nLabel distribution:")
print(ok["label"].value_counts().to_string())
print("\nBy domain:")
print(pd.crosstab(ok["domain"], ok["label"]).to_string())

ok.drop(columns=["judge_raw"]).to_csv(
    RESULTS / "unanswerable_question_labels.csv", index=False)
print(f"\nWrote results/unanswerable_question_labels.csv")
print("\nBuckets for the main run:")
print("  DEFENSIBLE        -> falsifiable_inability  (ascii-comparable lie)")
print("  UNDECIDABLE       -> undecidable_confabulation (new deception type)")
print("  NO_CORRECT_OPTION -> reported separately, in neither headline bucket")
