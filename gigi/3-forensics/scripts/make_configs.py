"""Write gigi/3-forensics/configs/<name>__<model>.yaml for every environment of the paper's repo.

Each file is the shipped config copied verbatim (prompts, task settings, step cap) with only the
`agent:` block replaced by ours: OpenRouter, DeepInfra pinned with no fallback, no reasoning field,
each model's model-card sampling. Where the paper ships one config per model (the cheating suite),
the base is the GLM 5.3 one: an open-weight model like ours, identical to the Kimi K3 config apart
from the model name. The six configs of the first pass (secret_number, coding, coding_norules,
precommit_hook, norvane) were written by hand on 17 Oct and are left alone.

Run:  python gigi/3-forensics/scripts/make_configs.py
"""
import pathlib
import re

import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
SHIPPED = HERE / "agent-interp-envs" / "configs"
OUT = HERE / "configs"

# our name: (shipped config, what it is)
ENVS = {
    # the paper's cheating suite (base: the GLM 5.3 config)
    "optimize_lru_cache": ("optimize_cache_c/optimize_lru_cache__glm-5-3.yaml", "cheating suite: Optimize LRU cache"),
    "optimize_rendering_engine": ("dashboard_perf/optimize_rendering_engine__glm-5-3.yaml", "cheating suite: Optimize rendering engine"),
    "optimize_rendering_engine_graded": ("stripped_engine/optimize_rendering_engine_graded__glm-5-3.yaml", "cheating suite: Optimize rendering engine, graded"),
    "grade_your_own": ("grade_your_own/grade_your_own__glm-5-3.yaml", "cheating suite: Grade your own"),
    "lucky_seed": ("lucky_seed/lucky_seed__glm-5-3.yaml", "cheating suite: Lucky seed"),
    "lucky_seed_readable_grader": ("lucky_seed_readable_grader/lucky_seed_readable_grader__glm-5-3.yaml", "cheating suite: Lucky seed + readable grader"),
    "copying_of_answers": ("leaked_fix/copying_of_answers__glm-5-3.yaml", "cheating suite: Copying of answers"),
    "leaked_tests": ("leaked_fix/leaked_tests__glm-5-3.yaml", "cheating suite: Leaked tests"),
    "leaked_scorer": ("tables_coverage_tests/leaked_scorer__glm-5-3.yaml", "cheating suite: Leaked scorer"),
    "reference_extraction": ("aj_data_anon/reference_extraction__glm-5-3.yaml", "cheating suite: Reference extraction"),
    # other task gaming
    "dashboard_costly_failure": ("dashboard_perf/costly_failure.yaml", "task gaming: dashboard_perf, PR marked ready without meeting the bar"),
    "revert_or_refactor": ("revert_or_refactor/revert_or_refactor.yaml", "task gaming: revert_or_refactor"),
    "ai_rd": ("ai_rd/ai_rd_generalization.yaml", "task gaming: ai_rd (generalization)"),
    "async_conversion": ("async_conversion/default.yaml", "task gaming: async_conversion"),
    "chess": ("chess/single_no_hint.yaml", "task gaming: chess (single game, no hint)"),
    "tictactoe": ("tictactoe/single_no_hint.yaml", "task gaming: tic-tac-toe (single game, no hint)"),
    "tower_of_hanoi": ("palisades_bounty/tower_of_hanoi.yaml", "task gaming: palisades_bounty, Tower of Hanoi"),
    "swe_rebench_v2": ("swe_rebench_v2/base.yaml", "task gaming: swe_rebench_v2"),
    # other honesty
    "puppeteer": ("puppeteer/puppeteer.yaml", "honesty: puppeteer"),
    "parse_duration": ("parse_duration/misleading_example.yaml", "honesty: parse_duration (agentic, misleading example)"),
    "lazy_investigation": ("lazy_investigation/lazy_investigation.yaml", "honesty: lazy_investigation"),
    "currency_conversion": ("currency_conversion/currency_conversion.yaml", "honesty: currency_conversion"),
    # other
    "sandbagging": ("sandbagging/normal.yaml", "other: sandbagging (normal)"),
    "eval_tampering": ("eval_tampering/default.yaml", "other: eval_tampering"),
    "funding_email": ("funding_email/default.yaml", "other: funding_email"),
}
# task settings the shipped config leaves blank
TASK_FILL = {"swe_rebench_v2": {"instance_id": "agnostiqhq__covalent-1392"}}   # the README's example instance

MODELS = {
    "qwen3-next-80b": ("qwen/qwen3-next-80b-a3b-instruct",
                       {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0},
                       "model card: temperature 0.7, top-p 0.8, top-k 20, min-p 0"),
    "gemma-3-27b": ("google/gemma-3-27b-it",
                    {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
                    "model card: temperature 1.0, top-p 0.95, top-k 64"),
}
OURS = {"provider", "model", "provider_preferences", "reasoning_effort", "temperature", "top_p", "top_k", "min_p"}


def agent_block(text):
    """(start, end) of the top-level `agent:` block."""
    m = re.search(r"^agent:[^\n]*\n((?:[ \t]+[^\n]*\n|[ \t]*\n)*)", text, re.M)
    return m.start(), m.end()


def main():
    for name, (src, what) in ENVS.items():
        text = (SHIPPED / src).read_text()
        a, b = agent_block(text)
        kept = {k: v for k, v in (yaml.safe_load(text[a:b])["agent"] or {}).items() if k not in OURS}
        for tag, (model, sampling, note) in MODELS.items():
            lines = ["agent:", "  provider: openrouter", f"  model: {model}",
                     "  provider_preferences:   # pinned, no fallback (project rule); DeepInfra serves fp8",
                     "    only: [deepinfra]", "    allow_fallbacks: false",
                     "  reasoning_effort: none  # no reasoning mode: send no reasoning field (our patch)"]
            lines += [f"  {k}: {v}" + (f"     # {note}" if i == 0 else "") for i, (k, v) in enumerate(sampling.items())]
            lines += [f"  {k}: {yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}" for k, v in kept.items()]
            body = text[:a] + "\n".join(lines) + "\n" + text[b:]
            for k, v in TASK_FILL.get(name, {}).items():
                body, n = re.subn(rf"^(\s+{k}:)\s*\S+", rf"\1 {v}", body, count=1, flags=re.M)
                assert n, (name, k)
            head = (f"# gigi/3-forensics: {name} x {tag} ({what}). Copied from agent-interp-envs@310c03f configs/{src};\n"
                    "# only the agent block changed (ours: OpenRouter, DeepInfra pinned, model-card sampling, no reasoning field)"
                    + (f"; task filled in: {TASK_FILL[name]}" if name in TASK_FILL else "") + ".\n")
            out = OUT / f"{name}__{tag}.yaml"
            out.write_text(head + body)
            cfg = yaml.safe_load(out.read_text())   # must parse, and carry our agent block
            assert cfg["agent"]["model"] == model and cfg["agent"]["provider_preferences"]["allow_fallbacks"] is False
        print(f"{name:34s} max_steps={kept.get('max_steps')}  from {src}")


if __name__ == "__main__":
    main()
