"""Adapter-is-live gate for re-runs of a Qwen3.6 student served by vLLM with LoRA (phase 2, 19 Nov 2026).

Why: on Qwen3.5/3.6, vLLM has served LoRA adapters that silently changed nothing (vllm-project/vllm#49354: generations
identical to the base model). A re-run on such a server would report "training had no effect" when the checkpoints
were never used. This gate fails loudly before any re-run; the fallback is scripts/merge_lora.py.

  0. parity (with vLLM serving; venv-train-qwen or any env with transformers 5 + the tokenizer):
     python qwen_lora_gate.py parity --data-dir distill/q1 --url http://localhost:8000/v1 --base-name qwen-base
     The server's own prompt token ids (/tokenize) equal the training prompt ids for a few cut points of real runs.
Then two steps, because the HF model and the vLLM server do not fit on one H200 at the same time:
  1. reference (GPU, BEFORE starting vLLM; venv-train-qwen):
     python qwen_lora_gate.py reference --data-dir distill/q1 --adapters c23=/workspace/out/q1/checkpoint-23,... \
         --out /workspace/out/q1/gate_ref.json [--n 4 --max-len 2048]
     For N val examples (encoded exactly as training encodes them), the HF log-probability of every target token under
     the base model (adapters off) and under each adapter.
  2. check (with vLLM serving base + adapters; any python with the stdlib):
     python qwen_lora_gate.py check --ref /workspace/out/q1/gate_ref.json --url http://localhost:8000/v1 \
         --base-name qwen-base [--api-key-env VLLM_API_KEY]
     For each served name, vLLM's prompt_logprobs on the same token ids. PASS for an adapter when its served
     log-probs are close to the HF adapter's (mean |diff| <= --tol, default 0.10 nats/token) AND clearly closer to the
     adapter's than to the base model's (distance to base >= 2x distance to adapter): a no-op adapter matches the base
     instead. Also PASS needs the served base to match HF base within --tol. Exit 1 on any failure.
"""
import argparse, json, os, pathlib, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def reference(a):
    import torch
    import distill_train as DT
    from transformers import AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.model)
    counts = {"val": __import__("collections").defaultdict(__import__("collections").Counter)}
    rows = DT.load_split(pathlib.Path(a.data_dir) / "val.jsonl", tok, a.max_len, None, counts, "val", "qwen36")
    rows = sorted(rows, key=lambda r: len(r["input_ids"]))[:a.n]   # the shortest, to keep the gate quick
    if not rows:
        sys.exit("no val example fits --max-len")
    ads = [x.split("=", 1) for x in a.adapters.split(",")]
    base = DT.load_student("qwen36", a.model, "sdpa", {"": 0} if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(base, ads[0][1], adapter_name=ads[0][0])
    for name, path in ads[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval(); dev = next(model.parameters()).device
    out = {"model": a.model, "adapters": dict(ads), "examples": []}
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([r["input_ids"]], device=dev)
            pos = [i for i in range(1, len(r["labels"])) if r["labels"][i] != -100]   # positions of target tokens
            ex = {"run": r["meta"]["run"], "input_ids": r["input_ids"], "target_positions": pos, "logprobs": {}}
            for name in ["base"] + [n for n, _ in ads]:
                ctx = model.disable_adapter() if name == "base" else __import__("contextlib").nullcontext()
                if name != "base":
                    model.set_adapter(name)
                with ctx:
                    logits = model(input_ids=ids, use_cache=False).logits[0].float()
                lp = torch.log_softmax(logits, -1)
                ex["logprobs"][name] = [float(lp[p - 1, r["input_ids"][p]]) for p in pos]
            out["examples"].append(ex)
            print(f"{r['meta']['run']}: {len(pos)} target tokens", flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out))
    print("reference written:", a.out)


def served_logprobs(url, key, model, ids, positions):
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    pl = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0].get("prompt_logprobs")
    if pl is None:
        sys.exit("the server returned no prompt_logprobs (vLLM needs prompt_logprobs support on /v1/completions)")
    vals = []
    for p in positions:
        entry = pl[p] or {}
        hit = entry.get(str(ids[p])) or entry.get(ids[p])
        if hit is None:
            sys.exit(f"{model}: no logprob for prompt token {ids[p]} at position {p}")
        vals.append(hit["logprob"] if isinstance(hit, dict) else float(hit))
    return vals


def mean_abs(x, y):
    return sum(abs(a - b) for a, b in zip(x, y)) / max(len(x), 1)


def check(a):
    ref = json.loads(pathlib.Path(a.ref).read_text())
    key = os.environ.get(a.api_key_env, "")
    names = ["base"] + list(ref["adapters"])
    fails = []
    for name in names:
        served = a.base_name if name == "base" else name
        d_ad = d_base = 0.0; n = 0
        for ex in ref["examples"]:
            got = served_logprobs(a.url, key, served, ex["input_ids"], ex["target_positions"])
            k = len(got); n += k
            d_ad += mean_abs(got, ex["logprobs"][name]) * k
            d_base += mean_abs(got, ex["logprobs"]["base"]) * k
        d_ad /= max(n, 1); d_base /= max(n, 1)
        if name == "base":
            ok = d_ad <= a.tol
            print(f"{served}: served vs HF base {d_ad:.4f} nats/token over {n} tokens -> {'PASS' if ok else 'FAIL'}")
        else:
            ok = d_ad <= a.tol and d_base >= 2 * d_ad
            print(f"{served}: served vs HF adapter {d_ad:.4f}, vs HF base {d_base:.4f} nats/token over {n} tokens -> "
                  f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(served)
    if fails:
        print(f"GATE FAILED for {fails}: the served adapters do not reproduce the trained ones (vLLM #49354 makes "
              f"them silent no-ops on Qwen3.5/3.6). Do NOT re-run; serve merged checkpoints (scripts/merge_lora.py).")
        sys.exit(1)
    print("GATE PASSED: every served adapter matches its HF counterpart and differs from the base model")


def parity(a):
    """Template parity: the server's prompt token ids (vLLM /tokenize, chat form, with tools and chat_template_kwargs)
    against the training prompt ids (distill_train.encode) for a few cut points of real val runs. The messages are sent
    as the harness sends them (tool-call arguments as JSON strings, reasoning in reasoning_content), so this catches
    any difference in how vLLM renders tool schemas (key order, extra fields), arguments or reasoning. Exit 1 on a gap."""
    import distill_train as DT
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    key = os.environ.get(a.api_key_env, "")
    exs = [json.loads(l) for l in open(pathlib.Path(a.data_dir) / "val.jsonl")][:a.n]
    bad = checked = 0
    for e in exs:
        msgs, tools = e["messages"], e["tools"]
        idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for i in sorted({idx[0], idx[len(idx) // 2], idx[-1]}):
            prompt = DT._template(tok, msgs[:i], tools, DT.QWEN_TEMPLATE_KW, add_generation_prompt=True)
            want = tok(prompt, add_special_tokens=False)["input_ids"]
            wire = []
            for m in msgs[:i]:
                w = dict(m)
                if w.get("tool_calls"):
                    w["tool_calls"] = [{"id": f"call_{k}", "type": "function",
                                        "function": {"name": tc["function"]["name"],
                                                     "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                                       for k, tc in enumerate(w["tool_calls"])]
                wire.append(w)
            body = json.dumps({"model": a.base_name, "messages": wire, "tools": tools, "add_generation_prompt": True,
                               "add_special_tokens": False, "chat_template_kwargs": DT.QWEN_TEMPLATE_KW}).encode()
            req = urllib.request.Request(a.url.rstrip("/").removesuffix("/v1") + "/tokenize", data=body,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            try:
                got = json.load(urllib.request.urlopen(req, timeout=300))["tokens"]
            except urllib.error.HTTPError as err:
                sys.exit(f"/tokenize refused the request ({err.code}): {err.read()[:300]!r}")
            checked += 1
            if got != want:
                bad += 1
                k = next((j for j, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
                print(f"PARITY GAP {e['run']} before turn at message {i}: token {k}: server "
                      f"{tok.decode(got[max(0, k - 8):k + 8])!r} vs training {tok.decode(want[max(0, k - 8):k + 8])!r}")
    print(f"template parity: {checked - bad}/{checked} prompts identical to the training rendering")
    if bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity", help="server prompt ids (vLLM /tokenize) == training prompt ids")
    p.add_argument("--data-dir", required=True); p.add_argument("--url", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B"); p.add_argument("--base-name", default="qwen-base")
    p.add_argument("--api-key-env", default="VLLM_API_KEY"); p.add_argument("--n", type=int, default=3)
    p.set_defaults(fn=parity)
    r = sub.add_parser("reference")
    r.add_argument("--data-dir", required=True); r.add_argument("--adapters", required=True)
    r.add_argument("--out", required=True); r.add_argument("--model", default="Qwen/Qwen3.6-27B")
    r.add_argument("--n", type=int, default=4); r.add_argument("--max-len", type=int, default=2048)
    r.set_defaults(fn=reference)
    c = sub.add_parser("check")
    c.add_argument("--ref", required=True); c.add_argument("--url", required=True)
    c.add_argument("--base-name", default="qwen-base"); c.add_argument("--api-key-env", default="VLLM_API_KEY")
    c.add_argument("--tol", type=float, default=0.10)
    c.set_defaults(fn=check)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
