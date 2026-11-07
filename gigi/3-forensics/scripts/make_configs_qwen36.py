"""Configs for Qwen3.6 27B's own solo runs (7 Nov 2026, the 6 Nov call: "we did not try Qwen3.6 27B on its own, it might
just be too dumb"). Two tags, each file the Qwen3-Next config of the same environment with only the agent block's
model, provider pin, reasoning switch and sampling replaced (the rule of make_configs_hero.py: task block, max_steps,
llm_heal... kept, so Qwen3.6 runs the exact task the others ran):
  qwen3.6-27b        thinking OFF (reasoning_effort: disabled -> {"reasoning": {"effort": "none"}}, patches/
                     hero_reasoning_off.patch), the model card's non-thinking sampling; all 29 environments. The same arm
                     as the swap's qwen36 (scripts/hero_swap.sh QWEN36).
  qwen3.6-27b-think  thinking ON (reasoning_effort: high -> {"reasoning": {"effort": "high"}}; Qwen3.6 thinks by default,
                     the field only makes it explicit), the model card's thinking sampling for general tasks; only the
                     four environments where the models cheat (Yuqi's question: why does it cheat).
DeepInfra fp8, no fallback: Qwen3.6 has no full-precision endpoint (hero_models_prep.md). Existing configs are never touched.
  python gigi/3-forensics/scripts/make_configs_qwen36.py
"""
import pathlib, re
import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
CFG = HERE / "configs"
ALL = ["ai_rd", "coding", "coding_norules", "dashboard_costly_failure", "grade_your_own",
       "optimize_rendering_engine", "optimize_rendering_engine_graded", "precommit_hook",
       "async_conversion", "chess", "copying_of_answers", "currency_conversion", "eval_tampering", "funding_email",
       "lazy_investigation", "leaked_scorer", "leaked_tests", "lucky_seed", "lucky_seed_readable_grader", "norvane",
       "optimize_lru_cache", "parse_duration", "puppeteer", "reference_extraction", "revert_or_refactor", "secret_number",
       "swe_rebench_v2", "tictactoe", "tower_of_hanoi"]
THINK = ["grade_your_own", "currency_conversion", "lazy_investigation", "eval_tampering"]
BLOCKS = {
    "qwen3.6-27b": """agent:
  provider: openrouter
  model: qwen/qwen3.6-27b
  provider_preferences:   # pinned, no fallback (project rule); DeepInfra fp8: Qwen3.6 has no full-precision endpoint
    only: [deepinfra]
    quantizations: [fp8]
    allow_fallbacks: false
  reasoning_effort: disabled  # thinking OFF: {"reasoning": {"effort": "none"}} (patches/hero_reasoning_off.patch)
  temperature: 0.7     # model card, non-thinking mode: temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence_penalty 1.5 (not sent: the harness has no field for it)
  top_p: 0.8
  top_k: 20
  min_p: 0
""",
    "qwen3.6-27b-think": """agent:
  provider: openrouter
  model: qwen/qwen3.6-27b
  provider_preferences:   # pinned, no fallback (project rule); DeepInfra fp8: Qwen3.6 has no full-precision endpoint
    only: [deepinfra]
    quantizations: [fp8]
    allow_fallbacks: false
  reasoning_effort: high  # thinking ON: {"reasoning": {"effort": "high"}} (Qwen3.6 thinks by default; this makes it explicit)
  temperature: 1.0     # model card, thinking mode, general tasks: temperature 1.0, top-p 0.95, top-k 20, min-p 0
  top_p: 0.95
  top_k: 20
  min_p: 0
""",
}
DROP = {"provider", "model", "provider_preferences", "reasoning_effort", "temperature", "top_p", "top_k", "min_p"}


def main():
    for tag, envs in (("qwen3.6-27b", ALL), ("qwen3.6-27b-think", THINK)):
        for env in envs:
            src = CFG / f"{env}__qwen3-next-80b.yaml"; out = CFG / f"{env}__{tag}.yaml"
            text = src.read_text()
            m = re.search(r"^agent:[^\n]*\n((?:[ \t]+[^\n]*\n|[ \t]*\n)*)", text, re.M)
            kept = {k: v for k, v in (yaml.safe_load(text[m.start():m.end()])["agent"] or {}).items() if k not in DROP}
            extra = "".join(f"  {k}: {yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}\n" for k, v in kept.items())
            body = text[:m.start()] + BLOCKS[tag] + extra + text[m.end():]
            body = re.sub(r"^# gigi/3-forensics: .*?\n(# .*?\n)*", "", body, count=1, flags=re.M)
            head = (f"# gigi/3-forensics: {env} x {tag} (Qwen3.6 27B solo, 7 Nov 2026). Made by scripts/make_configs_qwen36.py from\n"
                    f"# configs/{src.name}: only the agent block's model, provider pin, reasoning switch and sampling changed.\n")
            if out.exists() and out.read_text() == head + body:
                continue
            assert not out.exists(), f"{out} exists with different content: not overwriting"
            out.write_text(head + body)
            c = yaml.safe_load(out.read_text()); q = yaml.safe_load(src.read_text())
            assert c["agent"]["model"] == "qwen/qwen3.6-27b" and c["agent"]["provider_preferences"]["allow_fallbacks"] is False
            assert c.get("task") == q.get("task") and c["agent"]["max_steps"] == q["agent"]["max_steps"], env
            print(f"{tag:18s} {env:34s} max_steps={c['agent']['max_steps']}")


if __name__ == "__main__":
    main()
