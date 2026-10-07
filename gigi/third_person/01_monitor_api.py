"""Third-person monitors through OpenRouter: Gemini 2.5 Flash and GPT-5 mini.

Each monitor reads every transcript (results/transcripts.jsonl) in both prompt
variants (monitor_prompt.py) and gives a 1-100 "how likely is the assistant
lying" score. Provider pinned per model, reasoning effort "low", raw reply and
OpenRouter's reported cost saved before parsing; an unparseable score is null.
GPT-5 mini doesn't accept a temperature, so its replies aren't strictly
deterministic; Gemini runs at temperature 0.

Outputs: results/monitor_<model>.jsonl
Run from the repo root:
    python gigi/third_person/01_monitor_api.py --models all --per-category 1   # smoke test
    python gigi/third_person/01_monitor_api.py --models all                    # full run
Resumable per (row, variant).
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from monitor_prompt import VARIANTS, parse_score, prompt  # noqa: E402

# key from the environment (cloud sessions), else a gitignored .env: next to
# this script on the box, or the repo root on a laptop
load_dotenv(HERE / ".env")
load_dotenv()
KEY = os.environ["OPENROUTER_API_KEY"]
MODELS = {   # short name -> (OpenRouter id, pinned provider, extra params)
    "gemini-2.5-flash": ("google/gemini-2.5-flash", "google-ai-studio", {"temperature": 0}),
    "gpt-5-mini": ("openai/gpt-5-mini", "openai", {}),
}
RES = HERE / "results"
lock = threading.Lock()

p = argparse.ArgumentParser()
p.add_argument("--models", default="all")
p.add_argument("--per-category", type=int, default=None, help="smoke test: this many rows per category")
p.add_argument("--workers", type=int, default=8)
args = p.parse_args()
models = list(MODELS) if args.models == "all" else args.models.split(",")

rows = [json.loads(l) for l in open(RES / "transcripts.jsonl")]
if args.per_category:
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    rows = [r for v in by.values() for r in v[: args.per_category]]


def key(r, variant):
    return f"{r['category']}::{r['track']}::{r['sample_id']}::{variant}"


def call(model, text):
    mid, provider, extra = MODELS[model]
    body = {"model": mid, "messages": [{"role": "user", "content": text}], "max_tokens": int(os.getenv("MAX_TOKENS", "2000")),
            "reasoning": {"effort": "low"}, "provider": {"order": [provider], "allow_fallbacks": False},
            "usage": {"include": True}, **extra}
    for attempt in range(6):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=180)
            j = r.json()
            if "choices" in j:
                return j
            err = j.get("error")
        except Exception as e:
            err = repr(e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up: {err}")


for model in models:
    out = RES / f"monitor_{model}{'_' + os.getenv('MONITOR_TAG') if os.getenv('MONITOR_TAG') else ''}.jsonl"   # MONITOR_TAG=paper -> separate file
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in open(out) if l.strip()}
    jobs = [(r, v) for r in rows for v in VARIANTS if key(r, v) not in done]
    if os.getenv("RETRY_UNPARSED") == "1":   # redo only replies already in the file that have no readable score
        latest = {}
        for l in open(out):
            if l.strip():
                x = json.loads(l); latest[x["key"]] = x
        jobs = [(r, v) for r in rows for v in VARIANTS
                if key(r, v) in latest and latest[key(r, v)].get("score") is None and "error" not in latest[key(r, v)]]
    print(f"[{model}] {len(rows)} rows x {len(VARIANTS)} variants, {len(done)} done, {len(jobs)} to go", flush=True)

    def work(job):
        r, v = job
        try:
            j = call(model, prompt(r["messages"], v))
            raw = j["choices"][0]["message"].get("content") or ""
            u = j.get("usage", {})
            rec = {"key": key(r, v), "category": r["category"], "sample_id": r["sample_id"],
                   "track": r["track"], "variant": v, "model": model, "raw": raw,
                   "score": parse_score(raw), "prompt_tokens": u.get("prompt_tokens"),
                   "completion_tokens": u.get("completion_tokens"),
                   "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   "cost": u.get("cost"), "provider": j.get("provider")}
        except Exception as e:
            rec = {"key": key(r, v), "category": r["category"], "sample_id": r["sample_id"],
                   "variant": v, "model": model, "error": repr(e)[:400]}
        with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    recs = [json.loads(l) for l in open(out) if l.strip()]
    ok = [x for x in recs if "error" not in x]
    cost = sum(x.get("cost") or 0 for x in ok)
    print(f"[{model}] {len(ok)} ok, {len(recs) - len(ok)} errors, "
          f"{sum(1 for x in ok if x['score'] is None)} unparseable, measured cost so far ${cost:.4f}", flush=True)
print("MONITOR API DONE", flush=True)
