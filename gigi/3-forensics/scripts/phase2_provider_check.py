"""Phase 2, stage 1: check each candidate OpenRouter endpoint for a teacher model before any fleet (19 Nov 2026).

Free part (always runs): the model's endpoint list (does each pinned provider exist at that precision, with tools and
reasoning listed, and at what price), and the key's money: GET /api/v1/key (usage, limit, limit_remaining) and
GET /api/v1/credits (account credits and usage). Keys are read from the environment or the repo's .env, never printed.
The workspace's lifetime budget (memory openrouter-workspace-budget) is NOT visible to a key: John checks it.

Paid part (only with --paid; a few cents per endpoint): per pinned endpoint, one small agent exchange through the same
request shape the harness sends (tools, reasoning effort, the model card's sampling, provider pin without fallbacks,
usage accounting):
  1. "read the file" turn: did a tool call come back, parsed, with the right tool name? Did reasoning come back, how
     long (characters, reasoning tokens), with `reasoning_details`? Which host served it, what did it cost?
  2. the tool result is sent back twice, with and without the first turn's reasoning in the history: the difference in
     prompt tokens shows whether this host puts earlier reasoning back into DeepSeek's prompt (the paper's harness
     keeps all previous reasoning verbatim). Also shows cached tokens on a repeated prefix.
  3. on the first endpoint only, the first turn again at each --compare-efforts level: reasoning tokens per level
     (OpenRouter's "xhigh" should map to DeepSeek V4's Think Max).
Everything is written to results/logs/phase2/provider_check__<model>.json.

  cd agent-interp-envs && env -u VIRTUAL_ENV uv run python ../scripts/phase2_provider_check.py            # free part only
  cd agent-interp-envs && env -u VIRTUAL_ENV uv run python ../scripts/phase2_provider_check.py --paid     # ~$0.05-0.20
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
API = "https://openrouter.ai/api/v1"
TOOLS = [{"type": "function", "function": {
    "name": "execute_command", "description": "Run a shell command in the workspace and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command."}},
                   "required": ["command"]}}}]
SYSTEM = "You have access to one tool:\n- execute_command: run shell commands. Execute only one command per turn.\n"
USER = ("There is a file called notes.txt in your workspace. Read it with a shell command, then tell me in one "
        "sentence which number it contains and whether that number is prime.")
TOOL_RESULT = "Exit code: 0\nOutput:\nThe number is 221.\n"


def key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        sys.exit("OPENROUTER_API_KEY not found (environment or repo .env)")
    return k


def get(path: str, k: str | None = None) -> dict:
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {k}"} if k else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def endpoints(model: str) -> list[dict]:
    return get(f"/models/{model}/endpoints")["data"]["endpoints"]


def match(eps: list[dict], pin: str, quant: str) -> dict | None:
    for e in eps:
        tag = (e.get("tag") or "").lower()
        if (tag == pin or tag.startswith(pin + "/")) and (quant in ("", "any") or str(e.get("quantization")) == quant) \
                and not tag.endswith(("/flex", "/priority")) and "/us-" not in tag:
            return e
    return None


def money(k: str) -> dict:
    kk = get("/key", k)["data"]
    out = {"key_usage": kk.get("usage"), "key_limit": kk.get("limit"), "key_limit_remaining": kk.get("limit_remaining")}
    try:
        c = get("/credits", k)["data"]
        out.update(account_credits=c.get("total_credits"), account_usage=c.get("total_usage"))
    except Exception as e:   # noqa: BLE001  some keys cannot read account credits
        out["credits_error"] = type(e).__name__
    return out


def call(client, model, pin, quant, effort, temperature, top_p, max_tokens, messages):
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, parallel_tool_calls=False, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"effort": effort}, "usage": {"include": True},
                    "provider": {"only": [pin], "quantizations": [quant], "allow_fallbacks": False}})
    m = r.choices[0].message.model_dump() if r.choices else {}
    u = r.usage
    ux = (getattr(u, "model_extra", None) or {}) if u else {}
    tc = (m.get("tool_calls") or [None])[0]
    args_ok = False
    if tc:
        try:
            args_ok = "command" in json.loads(tc["function"]["arguments"] or "{}")
        except ValueError:
            pass
    reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
    return m, {
        "seconds": round(time.time() - t0, 1), "served_by": (r.model_extra or {}).get("provider"),
        "tool_call": tc["function"]["name"] if tc else None, "tool_args_parsed": args_ok,
        "content_chars": len(m.get("content") or ""), "reasoning_chars": len(reasoning),
        "reasoning_details": [d.get("type") for d in (m.get("reasoning_details") or [])],
        "prompt_tokens": u.prompt_tokens if u else None, "completion_tokens": u.completion_tokens if u else None,
        "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None) if u else None,
        "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None) if u else None,
        "cost": ux.get("cost"), "reasoning_head": reasoning[:300]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--pins", default="siliconflow:fp8,streamlake:fp8,deepinfra:fp8,gmicloud:fp8,novita:fp8")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--compare-efforts", default="high,xhigh")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=6000, help="caps each reply's cost")
    ap.add_argument("--paid", action="store_true", help="make the paid calls (needs GG's yes)")
    a = ap.parse_args()
    pins = [p.split(":") for p in a.pins.split(",")]
    eps = endpoints(a.model)
    report = {"model": a.model, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "endpoints": {}, "calls": {}}
    print(f"{a.model}: {len(eps)} endpoints listed")
    worst = 0.0
    for pin, quant in pins:
        e = match(eps, pin, quant)
        sp = (e or {}).get("supported_parameters") or []
        info = {"found": bool(e), "tag": (e or {}).get("tag"), "quantization": (e or {}).get("quantization"),
                "tools": "tools" in sp, "reasoning": "reasoning" in sp, "price_in": (e or {}).get("pricing", {}).get("prompt"),
                "price_out": (e or {}).get("pricing", {}).get("completion"), "status": (e or {}).get("status")}
        report["endpoints"][pin] = info
        print(f"  {pin:12} {quant:5} found={info['found']} tag={info['tag']} tools={info['tools']} reasoning={info['reasoning']} "
              f"in/out per M = {float(info['price_in'] or 0) * 1e6:.2f}/{float(info['price_out'] or 0) * 1e6:.2f}")
        if e:
            worst += 3 * (a.max_tokens * float(info["price_out"] or 0) + 4000 * float(info["price_in"] or 0))
    k = key()
    report["money_before"] = money(k)
    print("money:", {kk: v for kk, v in report["money_before"].items()})
    n_eff = len(a.compare_efforts.split(",")) if a.compare_efforts else 0
    print(f"worst-case cost of the paid part: about ${worst + n_eff * (a.max_tokens * 4e-6):.2f}")
    out = HERE / "results/logs/phase2" / f"provider_check__{a.model.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not a.paid:
        out.write_text(json.dumps(report, indent=1))
        print("free checks only (no --paid); wrote", out.relative_to(HERE))
        return
    rem = report["money_before"].get("key_limit_remaining")
    if rem is not None and rem < 2:
        sys.exit(f"key limit_remaining is ${rem}: too low, ask GG / John")
    from openai import OpenAI
    client = OpenAI(base_url=API, api_key=k, max_retries=0, timeout=900)
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    for i, (pin, quant) in enumerate(pins):
        if not report["endpoints"][pin]["found"]:
            continue
        rec = {}
        try:
            m1, rec["turn1"] = call(client, a.model, pin, quant, a.effort, a.temperature, a.top_p, a.max_tokens, base)
            if m1.get("tool_calls"):
                tid = m1["tool_calls"][0]["id"]
                tool_msg = {"role": "tool", "tool_call_id": tid, "content": TOOL_RESULT}
                with_r = {k2: v for k2, v in m1.items() if v is not None}
                without_r = {k2: v for k2, v in with_r.items() if k2 not in ("reasoning", "reasoning_content", "reasoning_details")}
                _, rec["turn2_with_reasoning"] = call(client, a.model, pin, quant, a.effort, a.temperature, a.top_p,
                                                      a.max_tokens, base + [with_r, tool_msg])
                _, rec["turn2_without_reasoning"] = call(client, a.model, pin, quant, a.effort, a.temperature, a.top_p,
                                                         a.max_tokens, base + [without_r, tool_msg])
                p1, p2 = rec["turn2_with_reasoning"]["prompt_tokens"], rec["turn2_without_reasoning"]["prompt_tokens"]
                rec["earlier_reasoning_in_prompt_tokens"] = (p1 - p2) if p1 is not None and p2 is not None else None
            if i == 0 and a.compare_efforts:
                rec["efforts"] = {}
                for eff in a.compare_efforts.split(","):
                    _, rec["efforts"][eff] = call(client, a.model, pin, quant, eff, a.temperature, a.top_p, a.max_tokens, base)
        except Exception as e:   # noqa: BLE001  one endpoint failing must not hide the others
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        report["calls"][pin] = rec
        t1 = rec.get("turn1", {})
        cost = sum((v.get("cost") or 0) for kk, v in rec.items() if isinstance(v, dict) and "cost" in v)
        cost += sum((v.get("cost") or 0) for v in (rec.get("efforts") or {}).values())
        print(f"  {pin:12} served_by={t1.get('served_by')} tool={t1.get('tool_call')} args_ok={t1.get('tool_args_parsed')} "
              f"reasoning={t1.get('reasoning_chars')} chars / {t1.get('reasoning_tokens')} tokens "
              f"details={t1.get('reasoning_details')} earlier-reasoning-in-prompt={rec.get('earlier_reasoning_in_prompt_tokens')} "
              f"cached(turn2)={rec.get('turn2_with_reasoning', {}).get('cached_tokens')} cost=${cost:.4f} {rec.get('error', '')}")
        if rec.get("efforts"):
            print("    reasoning tokens by effort:", {eff: v.get("reasoning_tokens") for eff, v in rec["efforts"].items()})
    report["money_after"] = money(k)
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out.relative_to(HERE))


if __name__ == "__main__":
    main()
