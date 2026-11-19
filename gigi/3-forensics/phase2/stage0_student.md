# Phase 2 stage 0, student side: Qwen3.6 27B (thinking on) as the distillation student (19 Nov 2026)

No money spent, no box, nothing committed. Everything is opt-in behind `--student qwen36`; the gemma path is unchanged
(verified byte-for-byte, below).

## Files

Changed:
- `scripts/distill_data.py`: `--student {gemma3,qwen36}` on build / inventory / check. New `qwen_messages()` (native
  messages: tool_calls with dict arguments, teacher reasoning -> `reasoning_content` (else OpenRouter's `reasoning`),
  role `tool` kept; only explicit keys are copied, so `_usage` and any `_`-key never reach the text), `student_messages()`,
  `msg_chars()` (content + reasoning + tool-call args for --max-context-chars; = len(content) for gemma), qwen36
  examples carry `"format": "qwen36"` and the run's `"tools"`; `check_qwen()`; manifest adds `student` and
  `trained_turns_without_reasoning` (qwen36 only). Runs whose arguments are not a JSON object are excluded with a reason.
- `scripts/distill_train.py`: `--student` (train / nll / merge), `--model` default per student, `--allow-slow-gdn`.
  `load_student()` (gemma call unchanged; Qwen3_5ForConditionalGeneration with `dtype=`, transformers>=5 guard),
  `QWEN_LORA` (q/k/v/o, gate/up/down, in_proj_qkv/z/a/b, out_proj; language model only) + `check_lora_targets()`
  (exits if a DeltaNet layer misses a projection or LoRA lands outside the language model), `gdn_fast_path()` (exits on a
  GPU without fla + causal-conv1d unless --allow-slow-gdn), `_template()` (tools + `preserve_thinking=True,
  enable_thinking=True` for qwen36; the exact old call for gemma), format check in `load_split`, transformers-5
  `warmup_ratio -> warmup_steps` (4.57.1 keeps `warmup_ratio`), log.json records student, LoRA counts, kernel status.
- `scripts/distill_eval.py`: `configs --student qwen36 [--base-name --envs --out-dir]` from
  `configs/<env>__qwen3.6-27b-bf16.yaml` with thinking ON (`chat_template_kwargs {enable_thinking, preserve_thinking}`,
  card thinking sampling 1.0 / 0.95 / 20 / 0, `reasoning_effort: none`); `DISTILL_ENVS` / `DISTILL_HELD_OUT` overrides.
  Nothing written into configs/ (tested into a scratch dir).

New:
- `scripts/venv_train_qwen.sh` (separate venv `/workspace/venv-train-qwen`): torch 2.13.0 (cu129), transformers 5.23.0,
  peft 0.21.3, trl 1.18.1, accelerate 1.16.0, datasets 5.2.0, flash-linear-attention 0.6.0 + fla-core 0.6.0,
  causal-conv1d 1.7.0 (`--no-build-isolation`). CPU-verified: transformers/peft/trl/accelerate/datasets/torch.
  NOT verified: the CUDA torch build, fla, causal-conv1d build.
- `scripts/qwen_lora_gate.py`: `parity` (server prompt ids from vLLM `/tokenize` with tools + chat_template_kwargs ==
  training prompt ids, messages sent in harness wire format), `reference` (HF log-probs of target tokens, base and each
  adapter, run before vLLM), `check` (vLLM `prompt_logprobs`: each adapter within 0.10 nats/token of HF adapter AND at
  least 2x closer to it than to HF base; served base within 0.10 of HF base; exit 1 otherwise).
- `scripts/merge_lora.py`: merge fallback per checkpoint (CPU, 5 GB shards, processor + tokenizer, arch check,
  `.done` marker, `--min-free-gb`).
- `tests/test_qwen36_render.py`, `tests/test_qwen36_train_cpu.py`.

## Tests (no GPU)

(a) Renderer vs the real template (Qwen3.6-27B tokenizer + chat_template.jinja, transformers 5.23.0), 13 examples from
6 thinking-off hero9 runs (6 tasks) + 3 thinking-on hero2_qwen36_think runs, trajectory and per-turn:
```
examples 13, dropped 0, text_equal 13, mask_exact 13, reasoning_in_think 10/10, tool_calls_xml 83/83,
ids_equal 1/13 -> boundary check: every gap is exactly 1 token per assistant turn WITHOUT reasoning
("<think>","\n" | "\n</think>" here vs one "\n\n" token when the whole text is tokenized at once); identical
after merging that split in 13/13; runs with reasoning in every turn: 0 extra tokens.
private keys: '_usage' added to every assistant message -> dropped, rendering identical: True
```
Real `build --student qwen36` + `check`: prefix identical 20/20, tool calls 183/183.

(b) CPU smoke (tiny random Qwen3_5ForConditionalGeneration from the real config: 3 DeltaNet + 1 attention layer,
1-block vision tower, real vocab), through the real scripts:
```
LoRA modules: deltanet 15 (3 layers x 5), attention 4, mlp 12; nothing outside language_model
train NLL per step: 12.442, 12.330, 12.183, 12.024, 11.964   eval: 12.446 -> 12.204
nll subcommand: base 12.4459, checkpoint 12.2021
merge_lora: merged vs PEFT logits max |diff| 0.0019
qwen_lora_gate reference: 1,005 target tokens, adapter vs base 0.306 nats/token
CPU SMOKE PASSED
```
(c) Gemma unchanged: old (git HEAD) vs new distill_data.py on hero9 data: trajectories (31/8/20 examples) and
per-turn incl. swap continuations (1,715/404) byte-identical, manifests identical; distill_train encode/encode_turns old
vs new with the gemma tokenizer (transformers 4.57.1): identical in 166/166 examples.

## Needs the GPU box to verify
1. venv_train_qwen.sh on CUDA: causal-conv1d build, fla import, fast path actually used; tokens/s and peak memory at
   32k (and 64k) with --offload-activations (use --smoke).
2. `qwen_lora_gate.py parity` against the running vLLM 0.32 server (tool-schema rendering: vLLM's
   ChatCompletionToolsParam has an extra `defer_loading` field; if it reaches the template the tools block differs).
3. `qwen_lora_gate.py reference` + `check` (vLLM LoRA no-op bug #49354; prompt_logprobs response format).
4. Serving a merged checkpoint (fallback path).

## Open risks
- Trajectory mode: one-token context gap per earlier reasoning-free turn (count in the manifest). DeepSeek gives
  reasoning every turn, so small there; for a teacher without reasoning (Qwen3-Next Instruct) every turn has it, and the
  student also learns empty thinking. Per-turn mode has no gap (costs ~9x tokens).
- Tool-schema parity with vLLM unverified until the parity gate runs.
- trl 1.x / transformers 5 verified on CPU only.
- Re-runs must use the thinking-on configs (preserve_thinking) or training and inference text differ.
