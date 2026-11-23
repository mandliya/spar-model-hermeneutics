"""p2q_load_copy.py (23 Nov 2026): load one saved copy (a LoRA checkpoint) into the running Qwen3.6 server and check
that it really changes the model. ON THE GPU BOX, with the server on localhost:8000.
vLLM has served LoRA adapters on this hybrid model as the untrained model before (vLLM #49354, phase2/student_qwen36.md):
so, for 3 training runs, the server scores the teacher's own next reply under the untrained model and under the copy
(echoed prompt log-probs). The copy must give DeepSeek's words a clearly higher chance; if the two are the same, it
exits 1 and nothing is run on that copy.
  python3 p2q_load_copy.py c24 /workspace/out/p2_deepseekv4_v1/checkpoint-24
"""
import json, os, pathlib, re, sys, urllib.error, urllib.request

name, path = sys.argv[1], sys.argv[2]
key = re.search(r"^VLLM_API_KEY=(.+)$", pathlib.Path("/workspace/.env").read_text(), re.M).group(1).strip()
B = "http://localhost:8000"


def post(url, body, timeout=900):
    req = urllib.request.Request(B + url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            t = r.read().decode()
    except urllib.error.HTTPError as e:   # show the server's reason
        raise SystemExit(f"{url}: HTTP {e.code}: {e.read().decode()[:600]}")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {"text": t}


models = json.load(urllib.request.urlopen(urllib.request.Request(B + "/v1/models", headers={"Authorization": f"Bearer {key}"}), timeout=30))
if name not in [m["id"] for m in models["data"]]:
    print("load:", post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}))

def wire(msgs):
    """The training file's messages in the API's format: tool calls get an id and JSON-string arguments, and each tool
    result the id of the call it answers (in order)."""
    out, pending = [], []
    for i, m in enumerate(msgs):
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for j, tc in enumerate(m["tool_calls"]):
                f = dict(tc["function"]); cid = tc.get("id") or f"call_{i}_{j}"
                if not isinstance(f.get("arguments"), str):
                    f["arguments"] = json.dumps(f.get("arguments") or {})
                tcs.append({"id": cid, "type": "function", "function": f}); pending.append(cid)
            m["tool_calls"] = tcs
        if m["role"] == "tool" and not m.get("tool_call_id"):
            m["tool_call_id"] = pending.pop(0) if pending else f"call_{i}"
        out.append(m)
    return out


data = pathlib.Path("/workspace/forensics/gigi/3-forensics/distill/p2_deepseekv4_v1/train.jsonl")
ex = []
for line in open(data):
    e = json.loads(line); msgs = e["messages"]
    idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if len(idx) >= 3 and sum(len(str(m.get("content") or "")) for m in msgs[:idx[2]]) < 20000:
        ex.append((e, idx[2]))
    if len(ex) == 3:
        break
gaps = []
for e, i in ex:
    msgs, tools = e["messages"], e.get("tools")
    target = e["messages"][i]
    tgt = (target.get("reasoning_content") or "") + (target.get("content") or "")
    if len(tgt) < 40:
        continue
    lp = {}
    for model in ("qwen36-base", name):
        body = {"model": model, "messages": wire(msgs[:i + 1]), "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
                "add_generation_prompt": False, "continue_final_message": True,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
        if tools:
            body["tools"] = tools
        r = post("/v1/chat/completions", body)
        pl = [next(iter(d.values()))["logprob"] for d in (r.get("prompt_logprobs") or []) if d]
        lp[model] = pl
    a, b = lp["qwen36-base"], lp[name]
    n = min(len(a), len(b)); k = max(1, n // 4)      # the last quarter of the prompt: mostly the teacher's reply
    ma, mb = -sum(a[-k:]) / k, -sum(b[-k:]) / k
    gaps.append(ma - mb); print(f"{e['env']}: {k} tokens, loss untrained {ma:.3f}, {name} {mb:.3f}")
if not gaps:
    print("GATE: no usable example"); sys.exit(2)
ok = sum(g > 0.02 for g in gaps) >= max(1, len(gaps) - 1)
print(f"GATE {'OK' if ok else 'FAILED'}: {name} lowers the loss on {sum(g > 0.02 for g in gaps)} of {len(gaps)} examples "
      f"(mean gap {sum(gaps) / len(gaps):.3f})")
sys.exit(0 if ok else 1)
