"""Garble gate for the classic round's gemma server (distill/PLAN_classic.md section 5; 17 Nov 2026). Round 1's
distill_garble_test.py with the prompt and model names as inputs: the coding task's first prompt as gemma gets it
(configs/coding__gemma-3-27b.yaml), sent N times one at a time to each model, then to all models at once (B each, mixed),
at gemma's sampling (temperature 1.0, top-p 0.95, top-k 64), 500 tokens. A reply is garbled when it contains the broken-
server gibberish (Bengali/Tamil/Georgian/Malayalam/Telugu/Thai script; 13 Nov: vLLM 0.11 CUDA graphs + LoRA). Gate:
0 garbled replies anywhere; exit 1 otherwise. Reads VLLM_API_KEY from /workspace/.env, never prints it.
  python3 classic_garble_gate.py --models gemma-base,c23,c46,c69,c92 --n 20 --batch 4
"""
import argparse, concurrent.futures as cf, json, pathlib, re, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]
BAD = re.compile(r"[ঀ-৿஀-௿ఀ-౿ഀ-ൿ฀-๿Ⴀ-ჿ]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=20, help="one-at-a-time replies per model")
    ap.add_argument("--batch", type=int, default=4, help="replies per model in the all-at-once batch")
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    args = ap.parse_args()
    env = dict(l.strip().split("=", 1) for l in open("/workspace/.env") if "=" in l and not l.startswith("#"))
    key = env["VLLM_API_KEY"].strip().strip('"')
    import yaml
    cfg = yaml.safe_load((HERE / "configs/coding__gemma-3-27b.yaml").read_text())
    msgs = [{"role": "system", "content": cfg["prompts"]["system_prompt"]}, {"role": "user", "content": cfg["prompts"]["user_prompt"]}]
    print("prompt: configs/coding__gemma-3-27b.yaml,", sum(len(m["content"]) for m in msgs), "characters")

    def ask(model):
        body = dict(model=model, messages=msgs, temperature=1.0, top_p=0.95, top_k=64, max_tokens=500)
        r = urllib.request.Request(args.url, json.dumps(body).encode(),
                                   {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        t = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]["content"] or ""
        return model, bool(BAD.search(t)), t[:80].replace("\n", " ")

    models = args.models.split(","); total = 0
    for m in models:
        res = [ask(m) for _ in range(args.n)]
        g = sum(b for _, b, _ in res); total += g
        print(f"one at a time {m}: garbled {g} of {len(res)}")
        for r in [r for r in res if r[1]][:3]:
            print("  ", r)
    batch = models * args.batch
    with cf.ThreadPoolExecutor(len(batch)) as ex:
        res = list(ex.map(ask, batch))
    for m in models:
        g = sum(b for mm, b, _ in res if mm == m); total += g
        print(f"all at once {m}: garbled {g} of {sum(mm == m for mm, _, _ in res)}")
    for r in [r for r in res if r[1]][:4]:
        print("  ", r)
    print("GATE PASSED: 0 garbled" if total == 0 else f"GATE FAILED: {total} garbled replies")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
