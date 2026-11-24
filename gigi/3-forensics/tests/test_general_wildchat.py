"""Checks for the general training set's far rung (24 Nov 2026). No paid call: the generator talks to a fake server on
this machine. Run with the environments' venv (distill_data.py needs it); check 4 also needs jinja2 on the path:

  agent-interp-envs/.venv/bin/python tests/test_general_wildchat.py [--template <Qwen3.6 chat_template.jinja>]

  1. the request equals DeepSeek's settings in configs/*__deepseek-v4-pro.yaml (model, pin, reasoning, sampling)
  2. generator against a fake OpenRouter: a 429 is retried; an empty text is retried 3 times then given up; every
     attempt is in raw.jsonl; a second pass skips what is answered; the cap stops new calls; a reply from another
     provider or without a cost stops the pass; a reply that is not JSON stays unusable
  3. converter: the p2ds keys, reasoning in reasoning_content, null text left out, 80/20 split, seed 0, repeatable
  4. with --template: Qwen3.6's own chat template renders an example as question, <think>reasoning</think>, answer
"""
import argparse, http.server, json, pathlib, sys, tempfile, threading, types

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import general_wildchat_generate as G   # noqa: E402
import general_wildchat_to_train as T   # noqa: E402


def reply(text="An answer.", reasoning="Some reasoning.", provider="SiliconFlow", cost=0.01, finish="stop"):
    u = {"prompt_tokens": 10, "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 12}}
    if cost is not None:
        u["cost"] = cost
    return 200, json.dumps({"id": "gen-x", "provider": provider, "model": "deepseek/deepseek-v4-pro", "usage": u,
                            "choices": [{"finish_reason": finish, "message": {
                                "role": "assistant", "content": text, "reasoning": reasoning,
                                "reasoning_details": [{"type": "reasoning.text", "text": reasoning}]}}]})


class Fake(http.server.BaseHTTPRequestHandler):
    script = {}      # question text -> list of (status, body), used in order; the last one repeats
    seen = []
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = body["messages"][0]["content"]; Fake.seen.append(body)
        steps = Fake.script.get(q) or [reply()]
        status, text = steps.pop(0) if len(steps) > 1 else steps[0]
        self.send_response(status); self.end_headers(); self.wfile.write(text.encode())
    def log_message(self, *a):
        pass


def run_pass(qs, out, cap, concurrency=1):
    a = types.SimpleNamespace(cap=cap, retries=8, timeout=10, concurrency=concurrency)
    done, spent, _, tries = G.read_raw(out)
    r = G.Run(a, "test-key", out, spent, tries)
    for q in [q for q in qs if q["id"] not in done]:
        r.one(q)
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--template"); a = ap.parse_args()
    # 1. settings
    import yaml
    for p in sorted((HERE / "configs").glob("*__deepseek-v4-pro.yaml")):
        c = yaml.safe_load(p.read_text())["agent"]; r = G.build_request("q")
        assert (r["model"], r["provider"], r["temperature"], r["top_p"], r["reasoning"]) == (
            c["model"], c["provider_preferences"], c["temperature"], c["top_p"], {"effort": c["reasoning_effort"]}), p.name
        assert "max_tokens" not in r and "max_tokens" not in c and r["usage"] == {"include": True}
        assert r["provider"]["allow_fallbacks"] is False and r["messages"] == [{"role": "user", "content": "q"}]
    print("1. request settings equal all", len(list((HERE / "configs").glob("*__deepseek-v4-pro.yaml"))), "DeepSeek configs")

    # 2. generator against a fake server
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    G.URL = f"http://127.0.0.1:{srv.server_address[1]}/"
    import time; real_sleep = time.sleep; time.sleep = lambda s: None   # no waiting between retries in the test
    tmp = pathlib.Path(tempfile.mkdtemp()); out = tmp / "raw.jsonl"
    kinds = ["coding", "writing", "advice", "general_knowledge", "maths_reasoning"]
    qs = [{"id": f"wildchat_{i:03d}", "kind": kinds[i % 5], "question": f"question {i}"} for i in range(40)]
    Fake.script = {"question 1": [(429, '{"error":{"code":429}}'), reply()],
                   "question 2": [reply(text="")],
                   "question 3": [reply(text=None)],
                   "question 4": [(200, "<html>not json</html>"), reply(reasoning="")],
                   "question 5": [reply(finish="length")]}
    r = run_pass(qs, out, cap=100)
    lines = [json.loads(l) for l in open(out)]
    n = lambda i: sum(l["id"] == f"wildchat_{i:03d}" for l in lines)
    assert (n(0), n(1), n(2), n(3), n(4), n(5)) == (1, 2, 3, 3, 2, 3), (n(0), n(1), n(2), n(3), n(4), n(5))
    assert r.ok == 37 and r.failed == 3 and not r.stop, (r.ok, r.failed, r.stop)
    assert all("Authorization" not in json.dumps(l) and "test-key" not in json.dumps(l) for l in lines)
    assert all(isinstance(l["raw"], str) for l in lines)   # the reply as sent, not a parsed copy
    done, spent, nl, _ = G.read_raw(out)
    assert len(done) == 37 and abs(spent - 0.01 * (len(lines) - 2)) < 1e-9, (len(done), spent)   # the 429 and the html carry no cost
    before = len(Fake.seen); r2 = run_pass(qs, out, cap=100)
    assert len(Fake.seen) - before == 9 and r2.ok == 0 and r2.failed == 3   # only the 3 unanswered are tried again
    print(f"2a. retries, resume and raw log: {len(lines)} attempts for 40 questions, 37 answered, ${spent:.2f} counted")
    out2 = tmp / "cap.jsonl"; Fake.script = {}
    r3 = run_pass(qs, out2, cap=0.05)
    assert r3.ok == 5 and "cap" in r3.stop and sum(1 for _ in open(out2)) == 5, (r3.ok, r3.stop)
    r3b = run_pass(qs, out2, cap=0.05)
    assert r3b.ok == 0 and sum(1 for _ in open(out2)) == 5   # the cap counts earlier passes
    for name, rep, word in (("other provider", reply(provider="DeepInfra"), "pin"), ("no cost", reply(cost=None), "cost")):
        o = tmp / f"{name}.jsonl"; Fake.script = {"question 0": [rep]}
        rr = run_pass(qs, o, cap=100)
        assert rr.stop and word in rr.stop and sum(1 for _ in open(o)) == 1, (name, rr.stop)
    time.sleep = real_sleep
    print("2b. cap stops new calls (also on a second pass); another provider or a missing cost stops the pass")

    # 3. converter
    qf = tmp / "questions.jsonl"; qf.write_text("".join(json.dumps(q) + "\n" for q in qs))
    T.HERE = tmp; (tmp / "distill").mkdir()
    args = types.SimpleNamespace(raw=str(out), questions=str(qf), out=str(tmp / "distill/t1"), split=0.8, seed=0,
                                 max_chars=0, dry_run=False, force=False)
    T.build(args); args.out = str(tmp / "distill/t2"); T.build(args)
    d1, d2 = tmp / "distill/t1", tmp / "distill/t2"
    assert (d1 / "train.jsonl").read_text() == (d2 / "train.jsonl").read_text()
    tr = [json.loads(l) for l in open(d1 / "train.jsonl")]; va = [json.loads(l) for l in open(d1 / "val.jsonl")]
    assert len(tr) + len(va) == 37 and len(va) == 7, (len(tr), len(va))   # 8+8+7+7+7 answered: 2+2+1+1+1 held back
    p2ds_keys = ["env", "run", "step", "hacked", "kind", "group", "train_turns", "messages", "format", "tools"]
    for e in tr + va:
        assert list(e) == p2ds_keys and e["format"] == "qwen36" and e["tools"] == [] and e["train_turns"] == [0]
        u, s = e["messages"]
        assert list(u) == ["role", "content"] and u["role"] == "user" and s["role"] == "assistant" and s["content"] == "An answer."
        assert set(s) <= {"role", "content", "reasoning_content"}
    ids = {e["run"].split("/")[1] for e in tr + va}
    assert not ids & {"wildchat_002", "wildchat_003", "wildchat_005"}          # empty, null and cut replies stay out
    e4 = next(e for e in tr + va if e["run"].endswith("wildchat_004"))
    assert "reasoning_content" not in e4["messages"][1]                        # no reasoning: kept, counted
    m = json.loads((d1 / "manifest.json").read_text())
    assert m["trained_turns_without_reasoning"] == {"without": 1, "of": 37} and len(m["excluded_runs"]) == 3
    assert m["totals"]["train"] == 30 and m["totals"]["val"] == 7
    print("3. converter: p2ds keys, 30 train / 7 val, repeatable, 3 unusable replies left out, 1 answer without reasoning kept")

    # 4. the chat template
    if a.template:
        import jinja2, jinja2.sandbox
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=["jinja2.ext.loopcontrols"])
        def rx(msg): raise ValueError(msg)
        env.globals["raise_exception"] = rx
        env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
        # transformers' own {% generation %} tag only marks the assistant span; here it is an always-true block
        import re
        src = re.sub(r"(\{%-?\s*)endgeneration(\s*-?%\})", r"\1endif\2", pathlib.Path(a.template).read_text())
        src = re.sub(r"(\{%-?\s*)generation(\s*-?%\})", r"\1if true\2", src)
        tpl = env.from_string(src)
        e = next(e for e in tr if "reasoning_content" in e["messages"][1])
        kw = dict(tools=e["tools"], preserve_thinking=True, enable_thinking=True)
        prompt = tpl.render(messages=e["messages"][:1], add_generation_prompt=True, **kw)
        full = tpl.render(messages=e["messages"], add_generation_prompt=False, **kw)
        assert full.startswith(prompt), (prompt, full)
        print("4. Qwen3.6's template. Prompt:", repr(prompt)); print("   Trained on:", repr(full[len(prompt):].rstrip("\n")))
        assert prompt == f"<|im_start|>user\n{e['messages'][0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n", prompt
        assert full[len(prompt):].rstrip("\n") == "Some reasoning.\n</think>\n\nAn answer.<|im_end|>"
    print("ALL OK")


if __name__ == "__main__":
    main()
