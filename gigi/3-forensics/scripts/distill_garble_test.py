# distill_garble_test.py: send the first coding prompt to vLLM one request at a time, then all at once, and count replies
# with broken-server gibberish (13 Nov 2026: with CUDA graphs on, 11/20 one-at-a-time base-gemma replies were garbage).
#   python3 distill_garble_test.py N_EACH_ONE_AT_A_TIME N_EACH_ALL_AT_ONCE gemma-base,c96
import json, glob, os, re, sys, urllib.request, concurrent.futures as cf
env = dict(l.strip().split("=",1) for l in open("/workspace/.env") if "=" in l and not l.startswith("#"))
KEY = env["VLLM_API_KEY"].strip().strip('"')
bad = re.compile(r"[ঀ-৿஀-௿Ⴀ-ჿഀ-ൿఀ-౿]")
runs = sorted(glob.glob("/workspace/forensics/results/raw/distill_v1_base/coding/**/run-*/step-*/messages.json", recursive=True))
msgs = [m for m in json.load(open(runs[0])) if m["role"] in ("system","user")][:2]
print("prompt chars", sum(len(m["content"]) for m in msgs), "from", runs[0].split("distill_v1_base/")[1])
def ask(model):
    body = dict(model=model, messages=msgs, temperature=1.0, top_p=0.95, top_k=64, max_tokens=500)
    r = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    t = json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["message"]["content"] or ""
    return model, bool(bad.search(t)), t[:80].replace("\n"," ")
for model in sys.argv[3].split(","):
    res = [ask(model) for _ in range(int(sys.argv[1]))]
    print("one at a time", model, "garbled", sum(b for _,b,_ in res), "/", len(res)); [print(" ", r) for r in res if r[1]][:3]
print("== all at once, base + checkpoints mixed")
models = ["gemma-base","c24","c48","c72","c96"] * int(sys.argv[2])
with cf.ThreadPoolExecutor(len(models)) as ex: res = list(ex.map(ask, models))
for m in ["gemma-base","c24","c48","c72","c96"]:
    print("all at once", m, "garbled", sum(b for mm,b,_ in res if mm==m), "/", sum(mm==m for mm,_,_ in res))
[print(" ", r) for r in res if r[1]][:4]
