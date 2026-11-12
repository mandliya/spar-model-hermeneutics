# What we changed in the harness so the two heroes run (hero9 and its swap)

This table lists every change beyond plain format translation that affects how gemma-3 27B or Qwen3.6 27B
run in the forensics environments. It was written on 12 Nov 2026 for John and Yuqi, from the code
(`agent-interp-envs/src/agent_interp_envs/providers/openrouter_provider.py`) and from the `config.yaml` that each
swap continuation saved (`results/raw/hero9_swap/.../config.yaml`).

## Settings per model (the same in the plain runs and in every takeover)

| Setting | gemma-3 27B | Qwen3.6 27B | Why |
|---|---|---|---|
| Where it runs | OpenRouter, DeepInfra only, fp8, no fallbacks | OpenRouter, DeepInfra only, fp8, no fallbacks | DeepInfra is the only provider with tool calls for gemma, and has no full-precision Qwen3.6 endpoint. Pinned so every call hits the same weights. |
| Sampling | temperature 1.0, top_p 0.95, top_k 64 | temperature 0.7, top_p 0.8, top_k 20, min_p 0 | Each model's own model-card settings, sent explicitly. |
| Thinking | none (gemma-3 has no thinking mode) | **switched off** (`reasoning: {effort: "none"}`) | Qwen3.6 thinks by default. Off keeps the two models comparable and the transcripts free of hidden reasoning. |
| Tool calls | **text mode**: tools described in the prompt, calls written as text and parsed back (rows below) | native tool calls | gemma-3 has no tool-call tokens. |
| End check (`end_confirm`, `end_task_required`) | off | off | Plain runs, as agreed on 10 Nov. Earlier rounds (hero8) had it on. |
| Step limit | each task's own (`max_steps`) | the same | Unchanged. |

## gemma's text-mode tool calling, step by step

| What | Exactly | Why |
|---|---|---|
| Tool instruction | The request carries no `tools` field. Instead the system prompt gets Google's Gemma 3 function-calling instruction, verbatim (from philschmid.de/gemma-function-calling, the reference Google points to), and each tool as a Python function signature with its description as the docstring. | gemma-3 has no native tool calling; this is Google's documented way to do it. |
| History shown to gemma | Every earlier tool call (gemma's own, or Qwen's in a takeover) is shown as a ` ```tool_code ``` ` block, and every tool result as a ` ```tool_output ``` ` block inside a user turn, so turns keep alternating user/assistant. | Gemma's chat format has only user and assistant turns. **This is also how Qwen's history is translated when gemma takes over a Qwen run.** |
| Reading gemma's reply | The first call in a ` ```tool_code ``` `, ` ```python ``` `, ` ```json ``` ` or unlabelled block, or a line that is just a call, that names an offered tool becomes a structured call. Also accepted: a quoted call (`["execute_command(...)"]`) and one dict of arguments (`execute_command({"command": "ls"})`). gemma's text stays in the message unchanged. | gemma writes calls in several shapes. Without this, a real call read as prose ends the run. |
| Calls that don't parse | A call gemma writes that doesn't parse (broken quotes, unclosed bracket) or names a tool we don't offer (`bash(...)`) is passed on as a call with the raw text as its arguments. The environment answers it with its normal error ("invalid arguments" / "unknown function"), and the run goes on. | That's what happens to a native model's malformed call. Before this fix (25 Oct) such replies ended gemma's runs: 50 of 52 such replies were calls, none of gemma's 1,013 final answers were. |
| Saved format | Parsed calls are stored as standard structured tool calls, so checkpoints and transcripts have the same format for both models. | One format for scoring, swapping and distillation. |

## When Qwen3.6 takes over a gemma run

| What | Exactly | Note |
|---|---|---|
| History shown to Qwen | gemma's turns as stored: its original text (including the ` ```tool_code ``` ` blocks it wrote) **plus** the structured tool call parsed from it. Tool results as native tool messages. | Qwen sees gemma's call twice, once as text and once as a structured call. Not translated further. Worth knowing for distillation. |

## Errors and retries (both models)

| What | Exactly | Why |
|---|---|---|
| Empty reply | A reply with no text and no call is retried (up to 3 tries), not read as the model's final answer. | It's a provider glitch, not a choice. |
| Rate limits | 429s are retried up to 14 times (about 20 minutes); timeouts and connection errors up to 3. Each retry is printed in the log. | DeepInfra's shared pool overloads; a 429 produced no reply, so retrying resamples nothing. |
| Lost runs | A run that fails anyway is counted as lost, by cause, never as cheating or as clean (`results/hero9_swap_loss_audit.md`). | |

## Not changed

- Task prompts, tools, environment files and graders are each task's own, identical for both models.
- No nudges, no hints and no end check in the hero9 rounds.
