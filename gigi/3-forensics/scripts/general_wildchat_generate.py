"""FAR rung of the general training set (24 Nov 2026, phase2/PLAN_next_24nov.md, direction 2): DeepSeek V4 Pro answers
the WildChat questions of general/wildchat/questions.jsonl, one question per call, with its full reasoning.

THIS SPENDS MONEY (OpenRouter). Ask GG first, with the estimate, every time. --dry-run sends nothing.

  python3 gigi/3-forensics/scripts/general_wildchat_generate.py --dry-run
  python3 gigi/3-forensics/scripts/general_wildchat_generate.py --limit 20 --cap 2 --concurrency 4      # smoke
  python3 gigi/3-forensics/scripts/general_wildchat_generate.py --cap 100 --concurrency 16              # all 2,000
  (plain python 3.10+, standard library only; run it detached with nohup for the full pass)

The model settings are DeepSeek's settings in our agent runs, read from scripts/make_target_configs.py (TARGETS), the
script that wrote configs/*__deepseek-v4-pro.yaml:
  - model deepseek/deepseek-v4-pro through OpenRouter, provider pinned to SiliconFlow fp8, allow_fallbacks false
  - reasoning effort xhigh, temperature 1.0, top_p 1.0, both sent explicitly
  - no max_tokens (the harness sends none), no streaming, request timeout 1,800 s
  - usage accounting on, so every reply carries its cost
The request body is built as agent-interp-envs' openrouter_provider.py builds it. Differences from an agent run, all
forced by "a plain question": no system prompt, no tools (so no parallel_tool_calls either), one user message.

What is saved: every HTTP reply, as the raw text OpenRouter sent, is appended to <out>/raw.jsonl BEFORE anything reads
it (one line per attempt: id, attempt, time, HTTP status, raw body). The reply's message keeps OpenRouter's own fields
(content, reasoning, reasoning_details); nothing is rewritten. A reply that cannot be read stays as it is; nothing is
coerced. Resumable: a question with a usable reply in raw.jsonl is skipped; the file is only ever appended to.

A usable reply: HTTP 200, one choice, served by the pinned provider, finish_reason "stop", and a non-empty text. An
empty text (the model spent the reply on reasoning) is retried like the harness does (3 attempts); 429, 5xx, timeouts
and network errors wait and retry (--retries, exponential backoff up to 120 s). Every billed attempt is counted.

Spend cap (--cap, dollars, required for a paid pass): the sum of usage.cost over EVERY line of raw.jsonl (earlier
passes included). No new call starts once the sum reaches the cap; calls already in flight finish, so the overshoot
is at most --concurrency calls. A 200 reply without usage.cost stops the pass (the cap could not be enforced). Before
the first call the key's remaining limit is read (free) and must cover the cap.

Key: OPENROUTER_API_KEY from the environment or the gitignored .env at the repo root. Never printed.
"""
import argparse, concurrent.futures as cf, json, os, pathlib, sys, threading, time, types, urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]           # gigi/3-forensics
ROOT = HERE.parents[1]                                       # repo root
sys.path.insert(0, str(HERE / "scripts"))
try:
    import yaml   # noqa: F401  make_target_configs imports it to write configs; we only read its TARGETS table
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
from make_target_configs import TARGETS   # noqa: E402

TAG = "deepseek-v4-pro"
URL = "https://openrouter.ai/api/v1/chat/completions"
EMPTY_ATTEMPTS = 3      # as openrouter_provider.invoke: an empty reply is retried, 3 attempts in all
PROVIDER_NAMES = {"siliconflow": "SiliconFlow"}   # slug we pin -> the name OpenRouter reports in `provider`


def settings(tag=TAG):
    t = TARGETS[tag]
    if not t["checked"]:
        sys.exit(f"{tag}: settings not checked in make_target_configs.TARGETS")
    return t


def build_request(question: str, tag=TAG) -> dict:
    """The request body, field for field what OpenRouterProvider sends for configs/*__deepseek-v4-pro.yaml, without
    tools: model, messages, temperature, top_p, reasoning effort, the pinned provider, usage accounting."""
    t = settings(tag); pin = t["pin"]
    return {
        "model": t["model"],
        "messages": [{"role": "user", "content": question}],
        "temperature": t["temperature"],
        "top_p": t["top_p"],
        "reasoning": {"effort": t["reasoning_effort"]},
        "provider": {"only": [pin], "quantizations": [t["pins"][pin]], "allow_fallbacks": False},
        "usage": {"include": True},
    }


def api_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("'\"")
    if not k:
        sys.exit("no OPENROUTER_API_KEY in the environment or in the repo's .env")
    return k


def parse(raw):
    """The raw body as JSON, or None if it is not JSON (never guessed)."""
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def cost_of(body):
    """usage.cost of a reply in dollars, or None if the reply has none."""
    u = (body or {}).get("usage")
    c = u.get("cost") if isinstance(u, dict) else None
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None


def verdict(status, body, tag=TAG):
    """(usable, why). Usable = a complete answer from the pinned provider (see the module text)."""
    if status != 200:
        return False, f"http {status}"
    if body is None:
        return False, "body is not JSON"
    ch = body.get("choices")
    if not isinstance(ch, list) or len(ch) != 1 or not isinstance(ch[0], dict):
        return False, "no choice" + (": " + str((body.get("error") or {}).get("code")) if isinstance(body.get("error"), dict) else "")
    want = PROVIDER_NAMES[settings(tag)["pin"]]
    if str(body.get("provider") or "").lower().replace(" ", "") != want.lower():
        return False, f"served by {body.get('provider')!r}, not {want}"
    msg = ch[0].get("message")
    if not isinstance(msg, dict):
        return False, "no message"
    if ch[0].get("finish_reason") != "stop":
        return False, f"finish_reason {ch[0].get('finish_reason')!r}"
    if not isinstance(msg.get("content"), str) or not msg["content"].strip():
        return False, "empty text"
    return True, "ok"


def read_raw(path):
    """(ids with a usable reply, dollars billed so far, lines, attempts per id) from an existing raw.jsonl."""
    done, spent, n, tries = set(), 0.0, 0, {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                try:
                    r = json.loads(line)
                except ValueError:
                    continue   # a line cut by a crash: it is not a usable reply and carries no readable cost
                body = parse(r.get("raw"))
                spent += cost_of(body) or 0.0
                tries[r.get("id")] = tries.get(r.get("id"), 0) + 1
                if verdict(r.get("status"), body)[0]:
                    done.add(r["id"])
    return done, spent, n, tries


class Run:
    def __init__(self, a, key, out, spent, tries):
        self.a, self.key, self.out = a, key, out
        self.lock = threading.Lock(); self.spent = spent; self.tries = dict(tries)
        self.stop = None; self.ok = self.failed = self.calls = 0; self.t0 = time.time()

    def call(self, body):
        """One HTTP call -> (status or None, raw text or None, error text or None)."""
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.a.timeout) as r:
                return r.status, r.read().decode("utf-8", "replace"), None
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace"), None
        except Exception as e:   # timeouts, connection errors: no reply
            return None, None, f"{type(e).__name__}: {str(e)[:200]}"

    def save(self, rec):
        with self.lock:
            with open(self.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

    def one(self, q):
        body = build_request(q["question"]); empties = 0
        for attempt in range(1, self.a.retries + 1):
            with self.lock:
                if self.stop:
                    return
                if self.spent >= self.a.cap:
                    self.stop = f"spend cap reached: ${self.spent:.2f} of ${self.a.cap:.2f}"; return
                self.tries[q["id"]] = n = self.tries.get(q["id"], 0) + 1; self.calls += 1
            t = time.time()
            status, raw, err = self.call(body)
            # raw reply to disk first, then read it
            self.save({"id": q["id"], "attempt": n, "t": round(t, 1), "seconds": round(time.time() - t, 1),
                       "status": status, "error": err, "request": {k: v for k, v in body.items() if k != "messages"},
                       "raw": raw})
            parsed = parse(raw); c = cost_of(parsed); good, why = verdict(status, parsed)
            with self.lock:
                self.spent += c or 0.0
                if status == 200 and parsed is not None and parsed.get("choices") and c is None:
                    self.stop = f"a reply came back without usage.cost ({q['id']}): the cap cannot be enforced"
                if why.startswith("served by"):
                    self.stop = f"{q['id']}: {why}; the provider pin did not hold"
                if status in (401, 402):
                    self.stop = f"http {status} from OpenRouter (key, credit or limit); nothing more is sent"
                if good:
                    self.ok += 1
                    if self.ok % 25 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] {self.ok} answered, {self.failed} failed, {self.calls} calls, "
                              f"${self.spent:.2f} of cap ${self.a.cap:.2f}, {time.time() - self.t0:.0f}s", flush=True)
                    return
                if self.stop:
                    return
            if why == "empty text" or why.startswith("finish_reason"):
                empties += 1
                if empties >= EMPTY_ATTEMPTS:
                    break
                continue
            if status is not None and status not in (408, 429) and status < 500 and status != 200:
                break   # a 4xx other than a rate limit will not get better by waiting
            time.sleep(min(120, 2 * 2 ** (attempt - 1)))
        with self.lock:
            self.failed += 1
        print(f"[{time.strftime('%H:%M:%S')}] gave up on {q['id']} after {attempt} attempts ({why})", flush=True)


def key_limit(key):
    """(remaining limit in dollars or None if unlimited, ok). One free call to OpenRouter's key endpoint."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode()).get("data") or {}
    except Exception as e:
        return None, f"could not read the key's limit ({type(e).__name__})"
    return d.get("limit_remaining"), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(HERE / "general/wildchat/questions.jsonl"))
    ap.add_argument("--out", default=str(HERE / "results/general_wildchat"), help="directory for raw.jsonl (appended to)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions of the file (0 = all)")
    ap.add_argument("--cap", type=float, default=None, help="hard spend cap in dollars over the whole raw.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=8, help="attempts per question in one pass")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per call (default: the agent runs' 1800)")
    ap.add_argument("--dry-run", action="store_true", help="build every request, print one, send nothing")
    a = ap.parse_args()
    t = settings()
    a.timeout = a.timeout or float(t["request_timeout"])
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    if len({q["id"] for q in qs}) != len(qs):
        sys.exit("duplicate ids in the questions file")
    if a.limit:
        qs = qs[:a.limit]
    out = pathlib.Path(a.out) / "raw.jsonl"
    done, spent, lines, tries = read_raw(out)
    todo = [q for q in qs if q["id"] not in done]
    reqs = [build_request(q["question"]) for q in todo]   # every request is built, dry run or not
    chars = sum(len(q["question"]) for q in todo)
    print(f"questions: {len(qs)} in {a.questions}; already answered in {out}: {len(qs) - len(todo)}; to send: {len(todo)} "
          f"({chars:,} characters of questions)")
    print(f"settings: {t['model']} on {t['pin']} {t['pins'][t['pin']]}, allow_fallbacks false, reasoning {t['reasoning_effort']}, "
          f"temperature {t['temperature']}, top_p {t['top_p']}, no max_tokens, timeout {a.timeout:.0f}s")
    print(f"raw.jsonl so far: {lines} lines, ${spent:.4f} billed")
    if a.dry_run:
        if reqs:
            print("DRY RUN, nothing sent. First request (POST " + URL + ", Authorization: Bearer <key not shown>):")
            print(json.dumps(reqs[0], indent=1, ensure_ascii=False))
        return
    if a.cap is None:
        sys.exit("a paid pass needs --cap <dollars>")
    if not todo:
        print("nothing to send"); return
    if spent >= a.cap:
        sys.exit(f"the cap (${a.cap:.2f}) is already reached by earlier passes (${spent:.2f})")
    key = api_key()
    left, err = key_limit(key)
    if err:
        sys.exit(err)
    if left is not None and left < a.cap - spent:
        sys.exit(f"the key has ${left:.2f} left, less than what the cap still allows (${a.cap - spent:.2f}); "
                 "raise the key's limit or lower --cap. Check the workspace budget too.")
    print(f"key limit left: {'no limit set' if left is None else f'${left:.2f}'}; cap ${a.cap:.2f}; "
          f"concurrency {a.concurrency}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    run = Run(a, key, out, spent, tries)
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(run.one, todo))
    done2, spent2, lines2, _ = read_raw(out)
    print(f"DONE: {len([q for q in qs if q['id'] in done2])} of {len(qs)} questions answered; this pass {run.ok} answered, "
          f"{run.failed} gave up, {run.calls} calls; raw.jsonl {lines2} lines, ${spent2:.4f} billed in all "
          f"(cap ${a.cap:.2f})")
    if run.stop:
        print("STOPPED EARLY:", run.stop); sys.exit(2)


if __name__ == "__main__":
    main()
