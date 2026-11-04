"""Configs for the hero target's own runs (4 Nov 2026): Llama 3.3 70B Instruct in the 8 gap environments, at full
precision (fp16 on CoreWeave, the only fp16 endpoint with tool calls) to match the bf16 weights the public NLA and
J-lens were fitted on. Each file is the Qwen3-Next config of the same environment with only the agent block's model,
provider pin and sampling replaced; every other agent setting (max_steps, llm_heal, ...) and the whole task block are
kept, so Llama runs the exact task Qwen ran. Existing configs are never touched.
  python gigi/3-forensics/scripts/make_configs_hero.py
"""
import pathlib, re
import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
CFG = HERE / "configs"
GAP = ["ai_rd", "coding", "coding_norules", "dashboard_costly_failure", "grade_your_own",
       "optimize_rendering_engine", "optimize_rendering_engine_graded", "precommit_hook"]
# 4 Nov, later the same day (the user: "run all the models"): every environment Qwen has a config for, so Llama gets the
# full section-1 table too. Same rule: the Qwen config with only the agent block's model, pin and sampling replaced.
OTHER = ["async_conversion", "chess", "copying_of_answers", "currency_conversion", "eval_tampering", "funding_email",
         "lazy_investigation", "leaked_scorer", "leaked_tests", "lucky_seed", "lucky_seed_readable_grader", "norvane",
         "optimize_lru_cache", "parse_duration", "puppeteer", "reference_extraction", "revert_or_refactor", "secret_number",
         "swe_rebench_v2", "tictactoe", "tower_of_hanoi"]
TAG = "llama-3.3-70b"
BLOCK = """agent:
  provider: openrouter
  model: meta-llama/llama-3.3-70b-instruct
  provider_preferences:   # pinned, no fallback (project rule); CoreWeave serves fp16 (full precision, matches the NLA/J-lens weights)
    only: [coreweave]
    quantizations: [fp16]
    allow_fallbacks: false   # no require_parameters: CoreWeave does not list parallel_tool_calls, which the harness sends (Llama makes one call per turn anyway)
  reasoning_effort: none  # no reasoning mode: send no reasoning field (our patch)
  temperature: 0.6     # generation_config.json: temperature 0.6, top-p 0.9, no top-k
  top_p: 0.9
"""
DROP = {"provider", "model", "provider_preferences", "reasoning_effort", "temperature", "top_p", "top_k", "min_p"}


def main():
    for env in GAP + OTHER:
        src = CFG / f"{env}__qwen3-next-80b.yaml"; out = CFG / f"{env}__{TAG}.yaml"
        text = src.read_text()
        m = re.search(r"^agent:[^\n]*\n((?:[ \t]+[^\n]*\n|[ \t]*\n)*)", text, re.M)
        kept = {k: v for k, v in (yaml.safe_load(text[m.start():m.end()])["agent"] or {}).items() if k not in DROP}
        extra = "".join(f"  {k}: {yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}\n" for k, v in kept.items())
        body = text[:m.start()] + BLOCK + extra + text[m.end():]
        body = re.sub(r"^# gigi/3-forensics: .*?\n(# .*?\n)*", "", body, count=1, flags=re.M)
        head = (f"# gigi/3-forensics: {env} x {TAG} (hero target, 4 Nov 2026). Made by scripts/make_configs_hero.py from\n"
                f"# configs/{src.name}: only the agent block's model, provider pin and sampling changed.\n")
        if out.exists() and out.read_text() == head + body:
            continue
        assert not out.exists(), f"{out} exists with different content: not overwriting"
        out.write_text(head + body)
        c = yaml.safe_load(out.read_text())
        q = yaml.safe_load(src.read_text())
        assert c["agent"]["model"] == "meta-llama/llama-3.3-70b-instruct" and c["agent"]["provider_preferences"]["allow_fallbacks"] is False
        assert c.get("task") == q.get("task") and c["agent"]["max_steps"] == q["agent"]["max_steps"], env
        print(f"{env:34s} max_steps={c['agent']['max_steps']}")


if __name__ == "__main__":
    main()
