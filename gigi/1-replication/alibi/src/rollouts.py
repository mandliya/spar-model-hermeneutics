"""Rebuild every follow-up conversation from the saved results, per category.

Shared by 04_recapture_fp32.py's logic and 05_named_cot.py, so the conversation
a tensor or a new answer is computed from is built in exactly one place.

Each builder yields (name, messages_before_followup, followup_text, answer,
kind) for every saved (row, condition). `name` is "<sample_id>__<condition>",
prefixed "A__"/"B__" for the sycophancy tracks - identical to the filenames of
the tensors already shipped.
"""

import pandas as pd

from src import common as C

R = C.RESULTS

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
