# Phase 2 prep: can our pipeline train and serve Qwen3.6 27B as the student? (19 Nov 2026)

Scope: distilling new teachers (first DeepSeek V4 Pro) into Qwen3.6 27B, per John's 19 Nov notes. No money spent:
code reading, public docs, the public chat template and the transformers source (downloaded, not run on a GPU).

## Bottom line

Training and serving Qwen3.6 27B as the student is feasible, but it is a real port, not a flag change. The data
renderer, the training script's model class and target modules, the training venv, and the re-run server all change.
One known serving bug (vLLM LoRA adapters silently doing nothing on this architecture) needs a hard gate. The long-run
question for DeepSeek can only be answered by measuring its run lengths in a small pilot.

| Item | Verdict | Main risk |
|---|---|---|
| 1. LoRA training support | GO with changes | needs transformers 5.x + fla + causal-conv1d kernels; our venv is 4.57.1 |
| 2. Memory and speed at 32k | likely GO | speed unknown; measure in the smoke test |
| 3. Chat template and data format | GO, biggest code change | new renderer (native tool calls, thinking blocks) |
| 4. Serving checkpoints for re-runs | CONDITIONAL | vLLM LoRA silent no-op bug on Qwen3.5/3.6; add a gate, fallback = merged weights |
| 5. Long teacher runs (DeepSeek, 1M) | UNKNOWN until measured | everything past 32k tokens / 105k chars is not trained today |

Side finding (relevant to the call): the classic Qwen teacher runs had thinking OFF (`chat_template_kwargs:
enable_thinking: false`, `reasoning_effort: none` in `configs/*__qwen3.6-27b-bf16.yaml`; the stored `reasoning` field is
`None`), and both models had the same step limits (e.g. lazy_investigation 30, currency_conversion 10). So gemma never
saw any Qwen thinking: "gemma overfits on Qwen's thinking" is not what happened. What it learned is to take more steps.

## 1. Architecture and LoRA training

- Qwen3.6-27B's config says `Qwen3_5ForConditionalGeneration` (a vision + language wrapper, like gemma-3), 64 layers:
  16 x (3 x Gated DeltaNet + 1 x gated attention), hidden size 5120 ([model card](https://huggingface.co/Qwen/Qwen3.6-27B),
  [config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)).
- Transformers support for the Qwen3.5 architecture landed in Feb 2026; docs point to v5.x
  ([docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5)). Our `venv_train.sh` pins transformers 4.57.1
  (the version that wrote the config file, not a requirement). Latest stable today: 5.23.0. -> must upgrade, and move
  peft/trl to versions that support transformers 5 (pin after a smoke test; I could not confirm exact minimums).
- Module names (read from transformers 5.23.0 `models/qwen3_5/modeling_qwen3_5.py`):
  - attention layers: `q_proj k_proj v_proj o_proj`
  - DeltaNet layers: `in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj` (+ `conv1d`, `A_log`, `dt_bias`)
  - MLP in every layer: `gate_proj up_proj down_proj`
  - plus a separate vision tower (`Qwen3_5VisionModel`): restrict LoRA to `language_model` as we do for gemma.
- Known pitfall: target lists copied from a plain transformer skip the DeltaNet blocks entirely (3/4 of layers)
  ([write-up](https://medium.com/@ishaafsalman/fine-tuning-qwen3-8-27b-what-breaks-and-how-to-fix-it-d77dc46e0745)).
  Assert after `get_peft_model` that DeltaNet projections got adapters.
- Fast kernels: without `flash-linear-attention` (Triton GDN kernels) and `causal-conv1d`, transformers falls back to a
  pure-PyTorch path, reported ~10x slower ([VeOmni guide](https://veomni.readthedocs.io/en/latest/examples/qwen3_5.html),
  [install notes](https://cumulus-blog-14a.pages.dev/qwen35-gated-delta-rule)). Install both into the train venv and
  check that the fast path is used before the long run.
- Our training setup avoids the usual GDN traps: batch size 1, no packing (no cu_seqlens), attention mask all ones.
  Gradient checkpointing + `save_on_cpu` offload with the fla kernels is untested here: the smoke test must check grad
  norms (it already does: NaN/zero = fail).

## 2. Memory and speed at 32,768 tokens on one H200

- Weights: ~27B params bf16 = ~54 GB (vs gemma 51 GB). LoRA r16 on all text projections: a few hundred MB.
- Activations: only 16 of 64 layers are full attention; DeltaNet layers keep a fixed-size state, so activation memory
  at 32k should be lower than gemma's (62 attention layers). With `--offload-activations` as now, 32k should fit in
  141 GB with room to spare; 64k is plausible and worth testing (see 5). One anecdotal report: bf16 LoRA under 70 GB
  on an H100 (sequence length not stated).
- Speed: no published tokens/s for LoRA at 32k found. gemma ran ~500-600 tokens/s on this box. Expect same order with
  the fla kernels, much slower without. Measure in the smoke test; budget 12-16 h per 2-epoch run until measured.

## 3. Chat template, tool calls, thinking

Today (`distill_data.py`): every run is re-rendered into gemma's text-mode tools (`_text_tools_messages`: tool
calls as ```` ```tool_code ```` Python calls, results as ```` ```tool_output ````) and gemma's template; `encode` /
`encode_turns` in `distill_train.py` render with `apply_chat_template` and require each shorter rendering to be a prefix
of the longer one (examples that fail are dropped with a reason, not silently).

For a Qwen3.6 student (public `chat_template.jinja`, read 19 Nov):
- Use Qwen's NATIVE tool format, not text mode: assistant turns with `tool_calls`, results as `role: tool`, and the
  tool schemas passed as `tools=` to `apply_chat_template`. The template renders calls as
  `<tool_call><function=NAME><parameter=ARG>value</parameter></function></tool_call>`. Re-runs serve with
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3` (model card).
- The template iterates `tool_call.arguments|items`: arguments must be a dict. Our stored OpenAI-style calls hold a JSON
  string -> `json.loads` them when rendering.
- Thinking: assistant turns render as `<think>\n{reasoning_content}\n</think>\n\n{content}`. Reasoning is KEPT for turns
  after the last real user message (the whole agent loop) and DROPPED for earlier turns unless `preserve_thinking=true`.
  Our runs have mid-run user messages in some environments (notices, confirmations); then earlier reasoning disappears
  from later renderings, which breaks the prefix property and would drop examples. Use `preserve_thinking=true` in BOTH
  the training rendering and the re-run server's `chat_template_kwargs`, so training and inference see the same text.
- Teacher reasoning -> student thinking block: put the teacher's reasoning (DeepSeek returns it as `reasoning_content`;
  OpenRouter as `reasoning`) into the assistant message's `reasoning_content`; the template wraps it. Requires the
  provider to return the full reasoning text, not a summary (John's point on closed models). The student must then run
  with thinking ON at re-run time, matching training.
- With thinking OFF the generation prompt ends in `<think>\n\n</think>\n\n` and past turns render the same empty block,
  so both modes keep the prefix property in one episode. Verify with `distill_data.py check` adapted to native calls.

## 4. Serving trained checkpoints for re-runs

- vLLM 0.32 (venv-qwen on the box) already serves Qwen3.6-27B (we used it for the teacher runs; hybrid prefix caching
  in "align" mode, 784-token blocks, ~85-88% cache-hit ceiling).
- LoRA on this architecture has open problems:
  - [vLLM #49354](https://github.com/vllm-project/vllm/issues/49354): on Qwen3.5/3.6, `enable_lora` produced the SAME
    generations as base (adapter silently a no-op, vLLM 0.22); workaround: merge the adapter and serve without LoRA
    (matched HF outputs ~0.97). A fix PR (#49525, adapters under `language_model` on wrapper classes) and a guard PR
    (#60039, reject adapters that match no module) exist; I could not confirm they are in 0.32.
  - [vLLM #47639](https://github.com/vllm-project/vllm/issues/47639): LoRA on only part of the packed DeltaNet
    projections (`in_proj_qkv` without `in_proj_z`) crashes. -> target all DeltaNet input projections together.
- Required gate before any re-run: for each served adapter, (a) generations on fixed prompts differ from base, and
  (b) next-token logprobs on 3-5 training examples match HF PEFT within tolerance. Fail -> fallback.
- Fallback: `distill_train.py merge` each checkpoint into full weights (~54 GB each) and serve them ONE AT A TIME
  (4 merged models do not fit together on one H200), i.e. re-runs run per checkpoint in sequence, or rent a second GPU.
- KV-cache sizing rule from the classic round still applies (live runs x history <= ~70% of cache, paced launches).

## 5. Long teacher runs (DeepSeek V4 Pro, 1M context)

- DeepSeek V4 Pro: open weights (MIT), 1.6T total / 49B active MoE, 1M context; its "Max" reasoning mode recommends
  >= 384K context ([overview](https://letsdatascience.com/news/deepseek-v4-preview-adds-open-weight-1m-context-models-4a8f21c6),
  [model page](https://www.llmreference.com/model/deepseek-v4-pro)). We only need API traces, not local inference.
- Today the build keeps a run only while its context stays under 105,000 characters (`--max-context-chars`, chosen so
  gemma's history fits 32k tokens) and training drops turns past 32,768 tokens. Classic Qwen (thinking off): 87 of
  1,120 kept runs reached our 49k window. With DeepSeek's reasoning in every turn, histories will be far longer; how
  much survives is unknown until measured.
- Practical plan:
  1. Pilot: 1-2 DeepSeek runs per task (cheap), measure tokens per run with reasoning, in the Qwen tokenizer.
  2. Report the share of teacher turns that fit in 32k and in 64k tokens.
  3. Choose: (a) keep 32k and report the share trained (simplest, honest); (b) raise `--max-len` to 64k (likely fits
     on the H200 thanks to DeltaNet; smoke-test it); (c) if runs are far longer, keep reasoning only for the most recent
     N turns in BOTH training and inference (a harness change that must be identical on both sides), or Yuqi's
     compaction / handoff notes, also identical on both sides. Do not train on a context the student will never see.
  4. Keep the student's serving window >= the training length, so the trained behaviour is reachable at re-run time.

## What must change, file by file

- `scripts/venv_train.sh`: transformers 5.x (Qwen3_5), peft/trl versions that support it, `flash-linear-attention`,
  `causal-conv1d` built for the box's torch/CUDA; keep the freeze file.
- `scripts/distill_train.py`:
  - `Gemma3ForConditionalGeneration` hard-coded in `train`, `nll`, `merge` -> `--model` picks the class
    (`Qwen3_5ForConditionalGeneration` or AutoModel for image-text-to-text); `torch_dtype` -> `dtype` on transformers 5.
  - `target_modules` regex -> `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)`;
    assert DeltaNet modules were wrapped.
  - `target_nll`: uses `core.model(...)` + `get_output_embeddings()`; works if the wrapper exposes the same API (Qwen's
    lm_head is untied; fine). Check no logit soft-capping. Docstrings and `encode` comments are gemma-specific.
  - smoke test: also log tokens/s and peak memory at 32k (and 64k if testing 3b).
- `scripts/distill_data.py`: a native-format renderer instead of `_text_tools_messages` (tool_calls with dict arguments,
  `role: tool`, `tools=` schemas from `tools_for`), teacher reasoning into `reasoning_content`, `preserve_thinking`;
  `load_runs_hero9` generalised to any teacher (`--expect-model`); `--max-context-chars` re-derived for the Qwen
  tokenizer with reasoning; `check` rewritten for native calls (today it uses `_tool_call_from_tool_code`).
- `scripts/distill_eval.py`: configs from the Qwen bf16 configs (native tools, thinking ON to match training,
  `preserve_thinking`), served model names, the round's held-out set.
- `scripts/classic_rerun.sh` (copy as a new round script): vLLM 0.32 venv instead of venv-gemma 0.11; Qwen serving flags
  (`--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`, hybrid prefix caching settings,
  `--enable-lora --max-lora-rank 16 --max-loras 4 --lora-modules ...`); garble gate prompt from a Qwen config; the new
  adapter-is-live gate; merged-weights fallback path.
- `scripts/classic_after.sh` (copy): the build flags, the held-out list, `nll` with the new model class.

## Risks (ranked)

1. vLLM LoRA silently ignored on Qwen3.6 -> fake "no change" results. Mitigation: the adapter-is-live gate; merged weights.
2. Long DeepSeek runs mostly untrained at 32k -> weak or biased distillation. Mitigation: measure first; 64k; same-on-both-sides truncation.
3. Template mismatch between training and serving (thinking kept vs dropped) -> training on text the student never sees. Mitigation: `preserve_thinking` on both sides; prefix checks already drop bad examples loudly.
4. Slow GDN path if kernels missing -> 10x training time. Mitigation: verify fast path in the smoke test.
5. Version churn (transformers 5 / peft / trl APIs) -> smoke-test before the long run, as now.
6. Provider for DeepSeek must return full reasoning tokens and tool calls (not checked here; no API calls made).
