"""Recapture every follow-up activation in float32, one category per run.

Why: the original captures converted gemma's bf16 hidden states to float16 on
save. bf16 -> float16 can overflow above 65,504 (and loses range generally);
bf16 -> float32 is lossless. Requested by Tianqin for probe work.

Nothing is regenerated. Every answer is already saved, so each row's
conversation is rebuilt, the saved answer teacher-forced, and two forward
passes record the state - exactly the capture points already shipped:

  <name>__predecision.pt   every layer, last prompt token before the answer
  <name>__answer_token.pt  every layer, at the answer token (DYL-matched)

Filenames match the float16 set exactly, so every existing map and CSV column
still identifies the right row - only the directory differs.

Output goes to /workspace/fp32_staging/<category>/ (NOT results/): at 1.35 MB
per tensor the full set is ~56 GB, which doesn't fit beside the 52 GB model,
so chain_fp32.sh uploads and verifies each category before the next starts.

Run:
    python 04_recapture_fp32.py --category mask_factual --limit 10
    python 04_recapture_fp32.py --category alibi
"""

import argparse
import pathlib

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import common as C
from src.dyl_activations import find_answer_char_span, capture_state_at_answer

R = C.RESULTS
STAGING = pathlib.Path("/workspace/fp32_staging")

p = argparse.ArgumentParser()
p.add_argument("--category", required=True)
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
CAT = args.category

KIND = {"neutral": "letter", "low_stakes": "letter", "high_stakes": "letter",
        "confidence": "confidence", "neutral_cot": "letter_cot",
        "low_stakes_cot": "letter_cot", "high_stakes_cot": "letter_cot",
        "confidence_cot": "confidence_cot", "plain": "letter", "named": "letter",
        "eval_plain": "letter", "eval_named": "letter"}
BASE = {"neutral": C.FOLLOWUP, "low_stakes": C.FOLLOWUP_LOW_STAKES,
        "high_stakes": C.FOLLOWUP_HIGH_STAKES, "confidence": C.FOLLOWUP_CONFIDENCE,
        "neutral_cot": C.FOLLOWUP_NEUTRAL_COT,
        "low_stakes_cot": C.FOLLOWUP_LOW_STAKES_COT,
        "high_stakes_cot": C.FOLLOWUP_HIGH_STAKES_COT,
        "confidence_cot": C.FOLLOWUP_CONFIDENCE_COT,
        "plain": C.FOLLOWUP_PLAIN, "eval_plain": C.FOLLOWUP_EVAL_PLAIN}


def wording(cond, named, eval_named):
    return {"named": named, "eval_named": eval_named}.get(cond) or BASE[cond]


def dataset(config, task_filter):
    d = pd.concat([C.load_config(config, "train"), C.load_config(config, "validation")])
    d = d.drop_duplicates(subset=["sample_id"])
    return d[task_filter(d)].set_index("sample_id")


def hist(ds, sid, assistant_turn):
    r = ds.loc[sid]
    return C.to_messages(r["trace"]) + [dict(r["lie_detection_prompt"]),
                                        {"role": "assistant", "content": str(assistant_turn)}]


# Each builder yields (name, messages_before_followup, followup_text, answer, kind)
def jobs_ascii():
    ds = dataset("google-gemma-3-27b-it_task-group_sandbagging",
                 lambda d: d["task"].str.contains("ascii", case=False, na=False))
    base = pd.read_csv(R / "ascii_results.csv").set_index("uid")
    rows = [(u, "neutral", str(base.loc[u, "followup_answer"])) for u in base.index]
    for f in ["ascii_followup_conditions.csv", "ascii_reasoning_conditions.csv",
              "ascii_benchmark_conditions.csv"]:
        d = pd.read_csv(R / f).drop_duplicates(subset=["sample_id", "condition"])
        rows += [(r.sample_id, r.condition, str(r.answer)) for r in d.itertuples()]
    for sid, cond, ans in rows:
        if sid in ds.index and sid in base.index:
            yield (f"{sid}__{cond}", hist(ds, sid, base.loc[sid, "probe_answer"]),
                   wording(cond, C.FOLLOWUP_NAMED_SANDBAGGING,
                           C.FOLLOWUP_EVAL_NAMED_SANDBAGGING), ans, KIND[cond])


def jobs_sycophancy(track):
    ds = dataset("google-gemma-3-27b-it_task-group_sycophancy",
                 lambda d: d["task"].str.startswith(
                     "sycophancy_answer" if track == "A" else "sycophancy_feedback"))
    prior_f, col, bench_f = (("sycophancy_answer_conditions.csv", "pressured",
                              "sycophancy_benchmark_conditions.csv") if track == "A" else
                             ("sycophancy_feedback_conditions.csv", "critique",
                              "sycophancy_feedback_benchmark_conditions.csv"))
    prior = pd.read_csv(R / prior_f).set_index("uid")
    bench = pd.read_csv(R / bench_f).drop_duplicates(subset=["sample_id", "condition"])
    for r in bench.itertuples():
        if r.sample_id in ds.index and r.sample_id in prior.index:
            yield (f"{track}__{r.sample_id}__{r.condition}",
                   hist(ds, r.sample_id, prior.loc[r.sample_id, col]),
                   wording(r.condition, C.FOLLOWUP_NAMED_SYCOPHANCY,
                           C.FOLLOWUP_EVAL_NAMED_SYCOPHANCY), str(r.answer), KIND[r.condition])


def jobs_mask():
    ds = dataset("google-gemma-3-27b-it_task-group_mask-factual",
                 lambda d: d["task"] == "mask_provided_facts")
    prior = pd.read_csv(R / "mask_factual_provided_conditions.csv").set_index("uid")
    bench = pd.read_csv(R / "mask_factual_benchmark_conditions.csv")
    bench = bench[~bench.condition.isin(["named", "eval_named"])]
    rerun = pd.read_csv(R / "mask_factual_named_rerun.csv")
    rerun = rerun[rerun.condition.isin(["named", "eval_named"])]
    for r in pd.concat([bench, rerun]).drop_duplicates(subset=["sample_id", "condition"]).itertuples():
        if r.sample_id in ds.index and r.sample_id in prior.index:
            yield (f"{r.sample_id}__{r.condition}",
                   hist(ds, r.sample_id, prior.loc[r.sample_id, "response"]),
                   wording(r.condition, C.FOLLOWUP_NAMED_MASK, C.FOLLOWUP_EVAL_NAMED_MASK),
                   str(r.answer), KIND[r.condition])


def jobs_unanswerable():
    ds = dataset("google-gemma-3-27b-it_task-group_unanswerable", lambda d: d["task"].notna())
    claims = pd.read_csv(R / "unanswerable_claims.csv").set_index("sample_id")
    bench = pd.read_csv(R / "unanswerable_benchmark_conditions.csv").drop_duplicates(
        subset=["sample_id", "condition"])
    for r in bench.itertuples():
        if r.sample_id not in claims.index:
            continue
        c = claims.loc[r.sample_id]
        if r.bucket == "confabulation":
            base = C.fresh_ask(c["question"]) + [{"role": "assistant",
                                                  "content": str(c["fresh_reply"])}]
            w = wording(r.condition, C.FOLLOWUP_NAMED_CONFABULATION,
                        C.FOLLOWUP_EVAL_NAMED_CONFABULATION)
        else:
            if r.sample_id not in ds.index:
                continue
            base = hist(ds, r.sample_id, c["claim_reply"])
            w = wording(r.condition, C.FOLLOWUP_NAMED_UNANSWERABLE,
                        C.FOLLOWUP_EVAL_NAMED_UNANSWERABLE)
        yield (f"{r.sample_id}__{r.condition}", base, w, str(r.answer), KIND[r.condition])


def jobs_games(prior_f, bench_f, named, eval_named):
    ds = dataset("google-gemma-3-27b-it_task-group_games", lambda d: d["task"].notna())
    prior = pd.read_csv(R / prior_f).drop_duplicates(subset=["sample_id"]).set_index("sample_id")
    bench = pd.read_csv(R / bench_f, low_memory=False).drop_duplicates(
        subset=["sample_id", "condition"])
    bench = bench[bench["answer"].notna()]
    for r in bench.itertuples():
        if r.sample_id in ds.index and r.sample_id in prior.index:
            yield (f"{r.sample_id}__{r.condition}",
                   hist(ds, r.sample_id, prior.loc[r.sample_id, "reply"]),
                   wording(r.condition, named, eval_named), str(r.answer), KIND[r.condition])


BUILDERS = {
    "ascii": jobs_ascii,
    "sycophancy_a": lambda: jobs_sycophancy("A"),
    "sycophancy_b": lambda: jobs_sycophancy("B"),
    "mask_factual": jobs_mask,
    "unanswerable": jobs_unanswerable,
    "capture_secret": lambda: jobs_games("capture_secret_conditions.csv",
                                         "capture_secret_benchmark_conditions.csv",
                                         C.FOLLOWUP_NAMED_CAPTURE,
                                         C.FOLLOWUP_EVAL_NAMED_CAPTURE),
    "alibi": lambda: jobs_games("alibi_conditions.csv", "alibi_benchmark_conditions.csv",
                                C.FOLLOWUP_NAMED_ALIBI, C.FOLLOWUP_EVAL_NAMED_ALIBI),
}

jobs = list(BUILDERS[CAT]())
if args.limit:
    jobs = jobs[: args.limit]
out_dir = STAGING / CAT
out_dir.mkdir(parents=True, exist_ok=True)
MAP = R / f"fp32_map_{CAT}.jsonl"
done = C.load_done(MAP, key="name")
todo = [j for j in jobs if j[0] not in done]
print(f"[{CAT}] {len(jobs)} rows, {len(done)} done, {len(todo)} to go -> {out_dir}")

tok = AutoTokenizer.from_pretrained(C.TARGET_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    C.TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager")
model.eval()


@torch.inference_mode()
def predecision(messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu()


for name, base, followup, answer, kind in tqdm(todo):
    msgs = base + [{"role": "user", "content": followup}]
    try:
        torch.save(predecision(msgs), out_dir / f"{name}__predecision.pt")
        at = None
        span = find_answer_char_span(answer, kind)
        if span is not None:
            st = capture_state_at_answer(tok, model, msgs, answer, span,
                                         out_dtype=torch.float32)
            if st is not None:
                torch.save(st, out_dir / f"{name}__answer_token.pt")
                at = f"{name}__answer_token.pt"
        C.append(MAP, {"name": name, "predecision_file": f"{name}__predecision.pt",
                       "answer_token_file": at})
    except Exception as e:
        C.append(MAP, {"name": name, "error": str(e)})

recs = C.read_jsonl(MAP)
print(f"[{CAT}] done: {sum(1 for r in recs if 'error' not in r)} ok, "
      f"{sum(1 for r in recs if 'error' in r)} errors")
