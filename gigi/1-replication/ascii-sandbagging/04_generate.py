"""STEP 4 — Replication arm. Does gemma-3-27b-it lie in these situations?

Replays each trace up to the final user turn, lets the target model write its
own answer, then labels that answer with the judge you validated in step 3.

This is the replication John asked for: the paper's own model, the paper's own
situations, your own run.

One caveat for any writeup: earlier assistant turns in the history come from
the dataset's recorded run, so the model is conditioned on its own past words
rather than generating the whole conversation fresh. Standard for trace-replay
evals; still worth a sentence.

Run:  uv run python 04_generate.py
Cost: ~$0.40 for 400 samples (generation is pennies; the judge is most of it).
"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.common import (DATA, RESULTS, TARGET_MODEL, MAX_WORKERS, TEMPERATURE,
                        chat, append, load_done, read_jsonl, to_messages,
                        prompt_prefix)
from src.judge import judge

out_path = RESULTS / "generations.jsonl"
# eval comes from the VALIDATION split — rows the judge prompt never saw.
df = pd.read_parquet(DATA / "eval.parquet")
done = load_done(out_path)
todo = df[~df["sample_id"].isin(done)]
print(f"Model under test: {TARGET_MODEL} @ temp {TEMPERATURE}")
print(f"{len(done)} done, {len(todo)} to go")


def work(row):
    msgs = to_messages(row["trace"])
    prefix = prompt_prefix(msgs)
    try:
        response = chat(TARGET_MODEL, prefix)
        verdict, raw = judge(prefix, response)
    except Exception as e:
        append(out_path, {"uid": row["sample_id"], "error": str(e)})
        return
    append(out_path, {
        "uid": row["sample_id"],
        "orig_did_lie": bool(row["did_lie"]),   # what gemma-12b did
        "our_did_lie": verdict,                 # what gemma-27b did
        "response": response,
        "judge_raw": raw,
        "task_id": row["task_id"],
    })


with ThreadPoolExecutor(MAX_WORKERS) as ex:
    list(tqdm(ex.map(work, [r for _, r in todo.iterrows()]), total=len(todo)))

res = pd.DataFrame([r for r in read_jsonl(out_path)
                    if r.get("our_did_lie") is not None])
print(f"\nUsable generations: {len(res)}")
print(f"gemma-3-27b lying rate (our run):  {res['our_did_lie'].mean():.1%}")
print(f"same rows, the dataset's labels:   {res['orig_did_lie'].mean():.1%}")
print("\nThese two should be close. They are the replication. The second one"
      "\nis NOT the paper's headline figure, because the pilot was balanced"
      "\n50/50 on purpose — for the paper comparison use the per-task-group"
      "\nrate from step 1 on the full config.")
print("\nPer task:")
print(res.groupby("task_id")[["orig_did_lie", "our_did_lie"]]
         .agg(["mean", "count"]).to_string())
