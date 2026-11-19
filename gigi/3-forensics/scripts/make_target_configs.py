"""Configs for phase 2 (19 Nov 2026): new teacher models through OpenRouter, and the Qwen3.6 27B thinking-on baseline.

  python3 scripts/make_target_configs.py deepseek-v4-pro [--pin siliconflow] [--check]
  python3 scripts/make_target_configs.py qwen36-think --host <box ip> --port <public port of 8000> [--check]
  (run with any python that has PyYAML, e.g. agent-interp-envs/.venv/bin/python)

Teachers (TARGETS below) -> configs/<task>__<tag>.yaml for the 28 tasks of the classic round. Source of each file:
configs/<task>__qwen3.6-27b.yaml, the hero9 OpenRouter config the classic configs were made from (classic_configs.py
changed only the agent endpoint and modal egress there; --check verifies task, prompts and step cap against the classic
config). Only the agent block changes:
  - provider openrouter, the target's model id, provider pinned (only + quantizations, allow_fallbacks false; project
    rule: every call hits the same weights). No require_parameters: no DeepSeek endpoint lists parallel_tool_calls, so
    it would exclude every endpoint.
  - reasoning on at the paper harness's standing level (agent-interp-envs/CLAUDE.md: deepseek "max"; OpenRouter's top
    level is "xhigh"), pinned explicitly, never a provider default
  - the model card's sampling, sent explicitly (DeepSeek V4: temperature 1.0, top_p 1.0, generation_config.json + README)
  - native tool calls (text_tools false), parallel_tool_calls false (one call per turn, as every task's prompt asks)
  - request_timeout 1800: max-effort replies at long context outlast the SDK's 600 s (fireworks_provider uses 1800)
  - the step cap (max_steps) and llm_heal from the source; the modal block from the source (no allow_hosts: run_modal then
    allows only the provider's host, openrouter.ai)
Reasoning is saved by the provider on every assistant turn (`reasoning`, `reasoning_content`, `reasoning_details`), sent
back with the history (the paper keeps all previous reasoning in DeepSeek's prompt), and printed in rollout.log. Each
call's usage and cost goes to rollout.log ([provider-usage]) and to messages.json (`_usage`), see openrouter_provider.py.

Qwen3.6 thinking-on baseline ("qwen36-think") -> configs/<task>__qwen3.6-27b-bf16-think.yaml from the classic
configs/<task>__qwen3.6-27b-bf16.yaml (our own vLLM server), with thinking ON:
  - chat_template_kwargs {enable_thinking: true, preserve_thinking: true} (model card: preserve_thinking keeps every
    earlier turn's reasoning, as DeepSeek's template does; the student then sees the same kind of history)
  - the card's thinking-mode sampling for general tasks: temperature 1.0, top_p 0.95, top_k 20, min_p 0
  - base_url / modal.allow_hosts = the new box (ai_rd also openrouter.ai, as in classic_configs.py)
  vLLM serve flags this needs (venv-qwen, vLLM 0.32): --reasoning-parser qwen3 --enable-auto-tool-choice
  --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true}'
  plus the classic round's other flags (bf16, the 49,152 window or wider: thinking runs are longer; size the fleet to the
  KV cache, memory vllm-agent-fleet-kv-sizing).

Never overwrites a config this script did not write (first line must carry its marker).
"""
import argparse
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent.parent
CFG = HERE / "configs"
MARK = "written by scripts/make_target_configs.py"
NEEDS_OPENROUTER = {"ai_rd"}   # the task's own agent key calls OpenRouter from inside the sandbox

# One entry per teacher. `checked`: settings confirmed against the model card and OpenRouter's endpoint list (19 Nov);
# unchecked entries refuse to write until someone confirms them (sampling, reasoning level, pin).
TARGETS = {
    "deepseek-v4-pro": dict(
        model="deepseek/deepseek-v4-pro",          # April checkpoint (0423), the paper's
        pins={"siliconflow": "fp8", "streamlake": "fp8", "deepinfra": "fp8", "gmicloud": "fp8", "novita": "fp8"},
        pin="siliconflow",   # cheapest fp8 endpoint with tools + reasoning listed (0.90 / 1.70 per M, cache 0.08)
        reasoning_effort="xhigh", temperature=1.0, top_p=1.0, request_timeout=1800, checked=True),
    "gpt-oss-120b": dict(
        model="openai/gpt-oss-120b", pins={"deepinfra": "bf16"}, pin="deepinfra",
        reasoning_effort="high", temperature=1.0, top_p=1.0, request_timeout=1800, checked=False),
    "kimi-k3": dict(
        model="moonshotai/kimi-k3", pins={"together": "unknown", "baseten": "fp8", "deepinfra": "mxfp4"}, pin="baseten",
        reasoning_effort="xhigh", temperature=1.0, top_p=0.95, request_timeout=1800, checked=False),
    "glm-5.3": dict(
        model="z-ai/glm-5.3", pins={"z-ai": "fp8", "siliconflow": "fp8"}, pin="z-ai",
        reasoning_effort="xhigh", temperature=1.0, top_p=0.95, request_timeout=1800, checked=False),
}


def classic_tasks() -> list[str]:
    return sorted(p.name.split("__")[0] for p in CFG.glob("*__qwen3.6-27b-bf16.yaml"))


def write(dst: pathlib.Path, header: str, cfg: dict) -> None:
    if dst.exists() and MARK not in dst.read_text().splitlines()[0]:
        sys.exit(f"REFUSING to overwrite {dst.name}: not written by this script")
    dst.write_text(header + "\n" + yaml.safe_dump(cfg, sort_keys=False))


def check(task: str, new: dict) -> str:
    """Everything but the agent endpoint/sampling and modal must equal the classic config."""
    old = yaml.safe_load((CFG / f"{task}__qwen3.6-27b-bf16.yaml").read_text())
    others = sorted(k for k in set(new) | set(old) if k not in ("agent", "modal") and new.get(k) != old.get(k))
    steps = (new["agent"].get("max_steps"), old["agent"].get("max_steps"))
    return f"{task:34s} other keys differ: {others or 'none'} | max_steps {steps[0]} (classic {steps[1]})"


def teacher(tag: str, pin: str | None, do_check: bool) -> None:
    t = TARGETS[tag]
    if not t["checked"]:
        sys.exit(f"{tag}: settings not checked yet (sampling, reasoning level, pin); confirm them in TARGETS first")
    pin = pin or t["pin"]
    if pin not in t["pins"]:
        sys.exit(f"{tag}: unknown pin {pin}; known: {sorted(t['pins'])}")
    for task in classic_tasks():
        src = CFG / f"{task}__qwen3.6-27b.yaml"
        c = yaml.safe_load(src.read_text())
        old = c["agent"]
        agent = {"provider": "openrouter", "model": t["model"],
                 "provider_preferences": {"only": [pin], "quantizations": [t["pins"][pin]], "allow_fallbacks": False},
                 "reasoning_effort": t["reasoning_effort"], "temperature": t["temperature"], "top_p": t["top_p"],
                 "max_steps": old["max_steps"], "text_tools": False, "parallel_tool_calls": False,
                 "request_timeout": t["request_timeout"]}
        if "llm_heal" in old:
            agent["llm_heal"] = old["llm_heal"]
        c["agent"] = agent
        dst = CFG / f"{task}__{tag}.yaml"
        write(dst, f"# {dst.name}: {MARK} from {src.name} (19 Nov 2026, phase 2; pin {pin} {t['pins'][pin]}, "
                   f"reasoning {t['reasoning_effort']}, sampling {t['temperature']}/{t['top_p']})", c)
        if do_check:
            print(check(task, c))
    print(f"wrote {len(classic_tasks())} configs for {tag} (pin {pin})")


def qwen_think(host: str, port: int, do_check: bool) -> None:
    tag = "qwen3.6-27b-bf16-think"
    for task in classic_tasks():
        src = CFG / f"{task}__qwen3.6-27b-bf16.yaml"
        c = yaml.safe_load(src.read_text())
        a = c["agent"]
        a.update(base_url=f"http://{host}:{port}/v1", reasoning_effort="none",
                 chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
                 temperature=1.0, top_p=0.95, top_k=20, min_p=0)
        c["modal"] = {**(c.get("modal") or {}),
                      "allow_hosts": [host] + (["openrouter.ai"] if task in NEEDS_OPENROUTER else [])}
        dst = CFG / f"{task}__{tag}.yaml"
        write(dst, f"# {dst.name}: {MARK} from {src.name} (19 Nov 2026, phase 2: Qwen3.6 thinking ON, preserve_thinking)", c)
        if do_check:
            print(check(task, c))
    print(f"wrote {len(classic_tasks())} configs for {tag} ({host}:{port})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", choices=sorted(TARGETS) + ["qwen36-think"])
    ap.add_argument("--pin", help="OpenRouter provider slug to pin (default: the target's own)")
    ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.tag == "qwen36-think":
        if not a.host or not a.port:
            sys.exit("qwen36-think needs --host and --port (the vLLM box)")
        qwen_think(a.host, a.port, a.check)
    else:
        teacher(a.tag, a.pin, a.check)


if __name__ == "__main__":
    main()
