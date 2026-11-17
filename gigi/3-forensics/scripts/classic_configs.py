"""Configs for the classic round's new Qwen3.6 runs (distill/PLAN_classic.md section 1, 17 Nov 2026).

Each configs/<task>__qwen3.6-27b-bf16.yaml is configs/<task>__qwen3.6-27b.yaml (the hero9 Qwen run, untouched) with only
the agent's endpoint changed, the same way scripts/distill_eval.py points gemma at our own server:
  - model: qwen/qwen3.6-27b, also the served name on our vLLM box (keeps hero9's run-folder slug qwen-qwen3.6-27b);
    base_url + api_key_env (VLLM_API_KEY, from the Modal secret vllm-secret-gigi)
  - provider_preferences: none (no OpenRouter provider pins on our own server)
  - reasoning_effort "none": no OpenRouter `reasoning` field in the request. Thinking is switched off on the server
    (vLLM --default-chat-template-kwargs '{"enable_thinking": false}') AND per request (chat_template_kwargs, the model
    card's switch); the gate checks it.
  - parallel_tool_calls: null (field omitted): vLLM obeys false and drops a second call, DeepInfra ignored it (hero9)
  - request_timeout 3600: our server sends nothing until a reply is complete; 600 s would resample long replies
  - modal.memory 2048 where the hero9 config had no modal block (modal_full.sh gave those 2 GB; writing allow_hosts
    creates the block, which would otherwise mean the 8 GB default)
  - modal.allow_hosts: the box's IP only (ai_rd also openrouter.ai: its task calls OpenRouter with its capped agent key)
Prompts, step caps, task variations and sampling stay exactly as in hero9.

  python scripts/classic_configs.py --host 135.135.24.114 --port 41528 [--check]
"""
import argparse
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent.parent
SKIP = {"swe_rebench_v2"}            # not used: its hero9 fleet has no saved config (as in round 1)
NEEDS_OPENROUTER = {"ai_rd"}         # the task's own agent key calls OpenRouter from inside the sandbox
SERVED = "qwen/qwen3.6-27b"
TAG = "qwen3.6-27b-bf16"     # model tag in file names and per-run tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--check", action="store_true", help="compare every written config with its hero9 source and print the differences")
    args = ap.parse_args()
    srcs = sorted(p for p in (HERE / "configs").glob("*__qwen3.6-27b.yaml") if p.name.split("__")[0] not in SKIP)
    for src in srcs:
        env = src.name.split("__")[0]
        dst = HERE / "configs" / f"{env}__{TAG}.yaml"
        c = yaml.safe_load(src.read_text())
        a = c["agent"]
        a.update(model=SERVED, base_url=f"http://{args.host}:{args.port}/v1", api_key_env="VLLM_API_KEY",
                 provider_preferences=None, reasoning_effort="none", chat_template_kwargs={"enable_thinking": False},
                 parallel_tool_calls=None, request_timeout=3600)
        hosts = [args.host] + (["openrouter.ai"] if env in NEEDS_OPENROUTER else [])
        had_modal = bool(c.get("modal"))
        c["modal"] = {**(c.get("modal") or {}), "allow_hosts": hosts, **({} if had_modal else {"memory": 2048})}
        dst.write_text(f"# {dst.name}: written by scripts/classic_configs.py from {src.name} (17 Nov 2026, classic round)\n"
                       + yaml.safe_dump(c, sort_keys=False))
        if args.check:   # everything outside the agent endpoint and modal.allow_hosts must be identical to hero9
            new = yaml.safe_load(dst.read_text()); old = yaml.safe_load(src.read_text())
            diffs = sorted(k for k in set(new["agent"]) | set(old["agent"]) if new["agent"].get(k) != old["agent"].get(k))
            others = sorted(k for k in set(new) | set(old) if k not in ("agent", "modal") and new.get(k) != old.get(k))
            mod = {k: v for k, v in (new.get("modal") or {}).items() if (old.get("modal") or {}).get(k) != v}
            print(f"{env:34s} agent changed: {', '.join(diffs)} | other keys changed: {others or 'none'} | modal: {mod}")
    print(f"wrote {len(srcs)} configs")


if __name__ == "__main__":
    main()
