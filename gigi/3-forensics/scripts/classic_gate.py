"""Gate for the classic round's Qwen3.6 server (distill/PLAN_classic.md section 1, gates 1-3; 17 Nov 2026).

Real hero9 Qwen histories (one run per task, cut before an assistant turn at 20k-100k characters), sent to our vLLM
server the way the harness sends them (raw message history, the task's own tools, no parallel_tool_calls field,
chat_template_kwargs enable_thinking false):
  A  one at a time, temperature 0, 64 tokens
  B  all at once (batched), same requests
  C  one at a time again (prefix-cache hits), same requests
  D  hero9's sampling (temperature 0.7, top-p 0.8, top-k 20, min-p 0), 800 tokens: the reply checks
Passes when: every reply in D has no thinking (reasoning null, no <think>), no unparsed tool-call text (<tool_call>,
<function=), no Chinese/Japanese/Korean or other-script gibberish, and a parsed tool call or plain text; and A, B and C
agree on their first 64 tokens for at least 10 of 12 histories (cached and batched decoding of this hybrid model is not
guaranteed bit-identical; a garbling server disagrees everywhere). Prints every reply's start for reading by eye.

  uv run python ../scripts/classic_gate.py --base-url http://135.135.24.114:41528/v1   (from agent-interp-envs/)
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import sys

import yaml
from openai import OpenAI

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import distill_data as D   # noqa: E402  tools_for, transcript

ENVS = ["coding", "tower_of_hanoi", "currency_conversion", "eval_tampering", "lazy_investigation", "precommit_hook",
        "dashboard_costly_failure", "grade_your_own", "chess", "funding_email", "leaked_scorer", "puppeteer"]
ODD = re.compile(r"[぀-ヿ㐀-鿿가-힯ঀ-৿஀-௿ఀ-౿ഀ-ൿ฀-๿Ⴀ-ჿ]")
RAW_TOOL = re.compile(r"<tool_call>|<function=|</function>|<think>|</think>")


def cases():
    out = []
    for f in D.HERO9_FULL:
        import csv
        for row in csv.DictReader(open(HERE / "results" / f)):
            env = row["environment"]
            if env not in ENVS or env in {c[0] for c in out} or row["status"] != "finished":
                continue
            run = HERE / row["run"]
            if not (run.parent / "config.yaml").exists():
                continue
            m = D.transcript(run)
            if not m:
                continue
            cut = None
            for i, msg in enumerate(m):   # the last assistant turn whose history is 20k-100k characters (else the longest under 100k)
                if msg.get("role") == "assistant" and i > 1:
                    n = len(json.dumps(m[:i]))
                    if n <= 100_000:
                        cut = i
            if cut is None:
                continue
            tools = D.tools_for(yaml.safe_load((run.parent / "config.yaml").read_text()))
            out.append((env, m[:cut], tools))
    return out


def ask(client, msgs, tools, **kw):
    r = client.chat.completions.create(model="qwen/qwen3.6-27b", messages=msgs, tools=tools,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}, **kw.pop("extra", {})}, **kw)
    return r.choices[0].message.model_dump()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    args = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY") or next(l.split("=", 1)[1].strip() for l in open(HERE.parent.parent / ".env") if l.startswith("VLLM_API_KEY="))
    client = OpenAI(base_url=args.base_url, api_key=key, timeout=900, max_retries=0)
    cs = cases()
    print(f"{len(cs)} histories: " + ", ".join(f"{e} ({len(json.dumps(m)) // 1000}k chars)" for e, m, _ in cs))
    greedy = dict(temperature=0, max_tokens=64)
    A = [ask(client, m, t, **greedy) for _, m, t in cs]
    with cf.ThreadPoolExecutor(len(cs)) as ex:
        B = list(ex.map(lambda c: ask(client, c[1], c[2], **greedy), cs))
    C = [ask(client, m, t, **greedy) for _, m, t in cs]
    with cf.ThreadPoolExecutor(len(cs)) as ex:
        Dr = list(ex.map(lambda c: ask(client, c[1], c[2], temperature=0.7, top_p=0.8, max_tokens=800,
                                       extra={"top_k": 20, "min_p": 0}), cs))
    sig = lambda r: json.dumps([r.get("content"), [(t["function"]["name"], t["function"]["arguments"]) for t in r.get("tool_calls") or []]])
    agree, bad = 0, []
    for (env, m, _), a, b, c, d in zip(cs, A, B, C, Dr):
        same = sig(a) == sig(b) == sig(c); agree += same
        text = (d.get("content") or "") + json.dumps(d.get("tool_calls") or [])
        probs = [p for p, hit in [("reasoning", bool(d.get("reasoning") or d.get("reasoning_content"))),
                                  ("raw tool/think text", bool(RAW_TOOL.search(d.get("content") or ""))),
                                  ("odd script", bool(ODD.search(text))),
                                  ("empty", not (d.get("content") or "").strip() and not d.get("tool_calls"))] if hit]
        if probs:
            bad.append((env, probs))
        calls = [t["function"]["name"] for t in d.get("tool_calls") or []]
        print(f"\n== {env}: A=B=C {same} | D tool calls {calls} | problems {probs or 'none'}")
        print("   A:", sig(a)[:160])
        if not same:
            print("   B:", sig(b)[:160]); print("   C:", sig(c)[:160])
        print("   D:", ((d.get("content") or "")[:200] + " " + json.dumps(d.get("tool_calls") or [])[:200]).replace("\n", " "))
    ok = not bad and agree >= min(10, len(cs) - 2)
    print(f"\nA=B=C agreement {agree} of {len(cs)}; reply problems {bad or 'none'}  ->  GATE {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
