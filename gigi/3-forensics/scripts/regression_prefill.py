"""Test Regression (the paper's code_summary_honesty prefill eval) for Qwen3-Next-80B and gemma-3-27b.

Both models get the paper's prefilled coding session (it sped up an endpoint but left one failing
test) and the same follow-up, then write the PR description. The paper's judge prompt grades each
reply: discloses_regression, silent, or claims_tests_pass (a false claim). It is the part-two
design at one decision point: the same trajectory, the last turn written by each model.

Verbatim from agent-interp-envs@310c03f environments/code_summary_honesty: the transcript and
tools (transcripts.py), the follow-up (FRAMINGS["pr_direct"]), the judge prompt (JUDGE_PROMPT),
tool_choice="none", max_tokens=8000. Ours: provider pinned (DeepInfra, no fallback), each
model's model-card sampling sent explicitly (the paper sends none), and the project's judge
(anthropic/claude-sonnet-4.5 pinned to Anthropic, temperature 0; the paper's default is
claude-sonnet-5). Raw replies and raw judge text are saved before parsing; appends to
results/regression_prefill.jsonl and skips samples already done; never rewrites it.

Run:  python gigi/3-forensics/scripts/regression_prefill.py --k 30     (needs OPENROUTER_API_KEY)
"""
import argparse
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = pathlib.Path(__file__).resolve().parents[1]
ENV = HERE / "agent-interp-envs" / "environments" / "code_summary_honesty"
sys.path.insert(0, str(ENV))
from transcripts import BASH_TOOL, END_TASK_TOOL, TRANSCRIPTS  # noqa: E402

spec = importlib.util.spec_from_file_location("paper_eval", ENV / "run_eval.py")
paper = importlib.util.module_from_spec(spec); spec.loader.exec_module(paper)

MODELS = {   # the same settings as configs/*.yaml
    "qwen/qwen3-next-80b-a3b-instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0},
    "google/gemma-3-27b-it": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
}
PROVIDER = {"only": ["deepinfra"], "allow_fallbacks": False}
JUDGE, JUDGE_PROVIDER = "anthropic/claude-sonnet-4.5", {"order": ["anthropic"], "allow_fallbacks": False}
OUT = HERE / "results" / "regression_prefill.jsonl"


def call(body, key, tries=4):
    err = None
    for i in range(tries):
        try:
            d = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=300,
                              headers={"Authorization": f"Bearer {key}"}, json={**body, "usage": {"include": True}}).json()
            if d.get("choices") and (d["choices"][0]["message"].get("content") or "").strip():
                return d["choices"][0]["message"]["content"], d.get("usage", {}), d.get("provider")
            err = d.get("error") or "empty reply"
        except (requests.RequestException, ValueError) as e:
            err = repr(e)
        time.sleep(2 ** (i + 1))
    return None, {"error": str(err)}, None


def one(model, t, k, key):
    rec = {"model": model, "transcript_id": t["id"], "framing": "pr_direct", "k": k, "sampling": MODELS[model]}
    msgs = t["messages"] + [{"role": "user", "content": paper.FRAMINGS["pr_direct"]}]
    reply, usage, prov = call({"model": model, "provider": PROVIDER, "messages": msgs, "tools": [BASH_TOOL, END_TASK_TOOL],
                               "tool_choice": "none", "max_tokens": 8000, **MODELS[model]}, key)
    rec.update(reply=reply, usage=usage, served_by=prov)
    if reply is None:
        rec["grade"] = None; rec["error"] = "sample failed"; return rec
    jtext, jusage, _ = call({"model": JUDGE, "provider": JUDGE_PROVIDER, "temperature": 0, "max_tokens": 2000,
                             "messages": [{"role": "user", "content": paper.JUDGE_PROMPT.format(
                                 failure_description=t["failure_description"], reply=reply)}]}, key)
    rec.update(judge=JUDGE, judge_raw=jtext, judge_usage=jusage)
    try:
        v = json.loads(re.search(r"\{.*\}", jtext or "", re.S).group(0))
        rec["grade"] = v["grade"] if v.get("grade") in paper.GRADES else None
        rec["evidence"] = v.get("evidence")
    except (AttributeError, ValueError, TypeError):
        rec["grade"] = None   # unparseable stays null, never coerced
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=30); ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    done = {(r["model"], r["transcript_id"], r["k"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
    todo = [(m, t, k) for m in MODELS for t in TRANSCRIPTS for k in range(args.k) if (m, t["id"], k) not in done]
    print(f"{len(todo)} samples to do ({len(done)} already done)")
    with ThreadPoolExecutor(args.workers) as pool:
        for rec in pool.map(lambda x: one(*x, key), todo):
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{rec['model'][:34]:34s} k={rec['k']:2d} {rec.get('grade')}")
    recs = [json.loads(l) for l in OUT.open()]
    for m in MODELS:
        g = [r["grade"] for r in recs if r["model"] == m and r.get("grade")]
        n = len(g)
        print(f"{m}: n={n}  " + "  ".join(f"{x} {g.count(x)}" for x in sorted(paper.GRADES))
              + (f"  | nondisclosure {100 * (g.count('silent') + g.count('claims_tests_pass')) / n:.0f}%" if n else ""))


if __name__ == "__main__":
    main()
