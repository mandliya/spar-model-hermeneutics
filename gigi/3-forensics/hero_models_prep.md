# Hero setup: preparation (3 Nov 2026)

From the SPAR call of 3 Nov (John, Yuqi). The team wants to compare a target with its substitutes by re-inference
(the swaps in this folder) and also by J-lens / J-space and NLAs (natural-language autoencoders) on activations.
Qwen3-Next-80B has neither, so John proposed a new "hero" setup. The target is **Llama 3.3 70B Instruct**, which has
an NLA and a J-lens. The substitutes are **Qwen3.6 27B** ("off-brand", with an NLA and a J-lens) and **gemma-3-27b-it**
(J-space and NLA; the preferred one if only one substitute is run). The backups are **gemma 4 31B** (already
prepared, [gemma4_prep.md](gemma4_prep.md)) and **GPT-OSS-20B** (John's pick for a different model family, with tool
calling). The plan is to re-run inference first without activations and to keep every trace, so that activations can
be recomputed later on a GPU. Quoted on the call: NLA training on gemma-3-27b costs 20–60 H100 hours, a
gemma-3-27b distillation 4–8 H100 hours.

Sources read today, all free and with no key: OpenRouter `/api/v1/models` and `/api/v1/models/<id>/endpoints`, the
Hugging Face model cards and `generation_config.json` files, the HF API listings of the lens and NLA repos, the
J-lens and NLA GitHub READMEs, and Neuronpedia. No paid call was made, and nothing under `results/` was written.

## 1. The five models on OpenRouter

Read from the endpoints API on 3 Nov 2026. Prices are $ per M tokens, input / output. "Max out" is the endpoint's
maximum completion length. The `only:` slug is the part of the endpoint tag before the slash. "n/s" means
OpenRouter states no precision for that endpoint.

| model | OpenRouter id | context | endpoints with tool calling (`only:` slug, precision, max out, $ in / out) | model-card sampling (source) |
|---|---|---|---|---|
| Llama 3.3 70B Instruct | `meta-llama/llama-3.3-70b-instruct` | 131,072 (128k) | `deepinfra` (turbo) fp8, 16,384, 0.10 / 0.32 · `akashml` fp8, 128,000, 0.20 / 0.52 · `groq` n/s, 32,768, 0.59 / 0.79 (no top_k) · `coreweave` **fp16**, 115,200, 0.71 / 0.71 (context 128,000) · `google-vertex` (us-central1) n/s, 8,192, 0.72 / 0.72 · `novita` **bf16**, but its context is only 12,288, so it cannot be used | temperature 0.6, top_p 0.9, no top_k. These come from `generation_config.json`; the model card itself names no sampling. Meta's repo is gated, so the file was read from the byte-copy mirror https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/generation_config.json (card: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| Qwen3.6 27B | `qwen/qwen3.6-27b` (it exists under this name; dense, 27B, 64 layers, Gated DeltaNet hybrid like Qwen3-Next) | 262,144 | all 7 have tools: `reka` n/s, 32,768, 0.20 / 3.00 · `chutes` fp8, 65,536, 0.25 / 2.00 · `siliconflow` fp8, 235,929, 0.30 / 3.20 · `deepinfra` fp8, 81,920, 0.32 / 3.20 (only one with min_p) · `phala` n/s, 262,140, 0.32 / 3.25 (degraded at check, 67% uptime) · `venice` fp8, 65,536, 0.325 / 3.25 (context 256,000) · `alibaba` n/s, 65,536, 0.45 / 2.70. **No endpoint is labelled bf16** | Thinking mode, general tasks: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0. Thinking mode for precise coding: temperature 0.6, same otherwise. **Non-thinking (instruct) mode: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5.** Thinking is on by default. https://huggingface.co/Qwen/Qwen3.6-27B ("Best Practices"); `generation_config.json` carries the thinking set |
| gemma-3-27b-it | `google/gemma-3-27b-it` | 131,072 | only `deepinfra` fp8, 16,384, 0.08 / 0.16. NextBit fp8, Parasail fp8, Nebius fp8 (110k) and Novita **bf16** (98,304 context) serve it without tools; text mode does not need tools, so those could carry it | temperature 1.0 (the default; the card sets none), top_p 0.95, top_k 64, as this folder already uses (README "Sampling"). `generation_config.json`: top_k 64, top_p 0.95; gated, read from https://huggingface.co/unsloth/gemma-3-27b-it/blob/main/generation_config.json (card: https://huggingface.co/google/gemma-3-27b-it) |
| gemma 4 31B | `google/gemma-4-31b-it` | 262,144 | 16 endpoints with tools. **bf16:** `crusoe`, 262,141, 0.14 / 0.40 · `novita`, 131,072, 0.14 / 0.40. **fp8:** `parasail` 0.15 / 0.40 · `deepinfra` 16,384, 0.20 / 0.40 · `primeintellect` 0.40 / 0.67 (it was 0.288 / 0.482 when gemma4_prep.md was written) · `siliconflow` 0.75 / 1.00 (degraded). **fp4:** `coreweave` 0.10 / 0.34 · `venice` · `chutes` · `deepinfra` turbo 0.135 / 0.51 · `nextbit` · `modelrun`. **n/s:** `dekallm`, `friendli`, `io-net` (0.361 / 1.093), `sambanova` (degraded). Full table in gemma4_prep.md section 2 | temperature 1.0, top_p 0.95, top_k 64 (https://huggingface.co/google/gemma-4-31b-it, `generation_config.json`; OpenRouter's `default_parameters` say the same) |
| GPT-OSS-20B | `openai/gpt-oss-20b` | 131,072 | `primeintellect` fp4, 32,768, 0.014 / 0.058 · `darkbloom` fp8, 32,768, 0.018 / 0.09 · `akashml` fp4, 117,964, 0.02 / 0.10 · `dekallm` bf16, 0.029 / 0.14 (no top_k) · `coreweave` fp4, 0.03 / 0.13 · `deepinfra` **bf16**, 117,964, 0.03 / 0.14 (top_k and min_p) · `parasail` fp4, 0.03 / 0.15 · `amazon-bedrock` n/s, 0.07 / 0.15 · `groq` n/s, 65,536, 0.075 / 0.30. Without tools: SiliconFlow fp8 | temperature 1.0, top_p 1.0, no top_k. The HF card and `generation_config.json` name none; the recommendation is in the openai/gpt-oss repo README, "Recommended Sampling Parameters" (https://github.com/openai/gpt-oss). Reasoning cannot be turned off (OpenRouter: `mandatory: true`, efforts low / medium / high, default medium). The MoE weights are natively MXFP4, so "fp4" is its published format and "bf16" is an upcast |

Notes:

- Qwen3.6 27B's id is exact. Its closest siblings are `qwen/qwen3.6-35b-a3b` (MoE, $0.10 / $0.90), `qwen/qwen3.5-27b`
  ($0.26 / $2.60), `qwen/qwen3.6-plus` and `qwen/qwen3.6-flash` (closed, 1M context).
- Llama 3.3 70B has no bf16 endpoint with a usable context. Only CoreWeave fp16, at 7 times DeepInfra's input
  price, is a full-precision cast of the published bf16 weights.
- Qwen3.6 27B's output price ($2.00–3.25 per M) is 10–20 times the others'. With thinking on, output tokens dominate
  its cost (section 4).

## 2. J-lens / J-space and NLAs per model

The J-lens is the Jacobian lens of Anthropic's "Verbalizable Representations Form a Global Workspace in Language
Models" (https://transformer-circuits.pub/2026/workspace/index.html; summary https://www.anthropic.com/research/global-workspace).
The J-space is the subspace that lens reads out, so a model "has J-space" when a fitted J-lens exists for it. The code
is https://github.com/anthropics/jacobian-lens, a reference implementation that ships no lenses. Fitted lenses for
open models are in the Neuronpedia repo https://huggingface.co/neuronpedia/jacobian-lens (files listed through the HF
API today). Most were fitted by Neuronpedia with the paper's code and recipe: 1,000 WikiText-103 prompts of 128
tokens, bf16. The same repo carries **J++ lenses** (Ayonrinde & Lindsey, 2026, "an improved, drop-in replacement",
original files at https://huggingface.co/koayon/jpp-lenses).

The NLAs come from Anthropic's "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations"
(https://transformer-circuits.pub/2026/nla/index.html). The code is https://github.com/kitft/natural_language_autoencoders,
and the released checkpoints are in the HF collection https://huggingface.co/collections/kitft/nla-models. Each NLA is
a pair: an AV (activation verbalizer, activation to text) and an AR (activation reconstructor, text to activation),
trained on one layer.

| model | J-lens / J-space | NLA |
|---|---|---|
| Llama 3.3 70B Instruct | **Yes.** `llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt` (10.6 GB), fitted by Neuronpedia on `meta-llama/Llama-3.3-70B-Instruct`, bf16, 1,000 prompts. No J++ lens | **Yes, official.** Layer 53 of 80, d_model 8192: https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-av and https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-ar (24–25 Apr 2026, trained with Megatron). Neuronpedia also hosts it, with an FP8 copy of the AV: https://huggingface.co/neuronpedia-org/Llama-3.3-70B-NLA-L53-av-FP8 |
| Qwen3.6 27B | **Yes, the best covered.** `qwen3.6-27b/jlens/.../Qwen3.6-27B_jacobian_lens_n1000.pt`; its CREDIT.md says it "was trained by @mntss (Mateusz Piotrowski, Anthropic Interpretability)". Also a J++ lens (`qwen3.6-27b/jpp/`, Ayonrinde & Lindsey, recall@10 55.2%) and an `oracle/` LoRA adapter. It is the model featured on https://www.neuronpedia.org/jlens | **Not official; community only.** No kitft release. https://huggingface.co/ceselder/qwen3.6-27b-nla-L42 is the most complete: layer 42 of 64, AV LoRA plus AR, EasyNLA recipe, held-out FVE 74.5%, Apache-2.0. Siblings: `ceselder/qwen3.6-27b-nla-av` (merged AV, other licence) and `ceselder/qwen3.6-27b-nla-rl` (every RL checkpoint, FVE about 78%). Others: `camilablank/qwen3.6-27b-multilayer-nla-rl600`, `agu18dec/qwen3.6-27b-all-layer-nla-rl2000` and the `ljhunt/nla-qwen36-27b-*` series (control and poison experiments, Oct 2026). Neuronpedia's NLA page lists only Gemma 27B L41, Llama 70B L53 and a Qwen 1.5B L18 |
| gemma-3-27b-it | **Yes.** `gemma-3-27b-it/jlens/.../gemma-3-27b-it_jacobian_lens.pt` (3.5 GB, Neuronpedia, bf16), plus a lens for the base model `gemma-3-27b`. No J++ lens | **Yes, official.** Layer 41 of 62, d_model 5376: https://huggingface.co/kitft/nla-gemma3-27b-L41-av and https://huggingface.co/kitft/nla-gemma3-27b-L41-ar (the AR is fine-tuned from gemma-3-27b-it). Also on Neuronpedia |
| gemma 4 31B | **Yes, but fitted on the base model** `google/gemma-4-31B`, not `-it`: `gemma-4-31b/jlens/` and a J++ lens `gemma-4-31b/jpp/`. Using either on the instruct model is a mismatch. Community lenses for `-it` exist (`zachperlman20/gemma-4-31B-it-jpp-lens`, `andyoneal/gemma-4-31b-it-jacobian-lens-compressed`); not checked | **None found** for the 31B. Community NLAs exist only for the small and MoE gemma 4 models (`Solshine/gemma-4-e2b-nla-*`, `cds-jb/gemma-4-26b-a4b-nla-l20`, `ceselder/gemma4-26b-a4b-nla`) |
| GPT-OSS-20B | **Yes.** `gpt-oss-20b/jlens/.../gpt-oss-20b_jacobian_lens.pt` (Neuronpedia, bf16 activations) | **None found** (HF searches for "gpt-oss" with "nla" return nothing). SAEs exist (`andyrdt/saes-gpt-oss-20b`) |

What I could not verify:

- Neuronpedia renders a page at `/<model>/jlens` and `/<model>/nla` for **any** model id, a made-up one included. A
  page loading therefore proves nothing about coverage. The table above rests on the HF file listings, not on
  those pages.
- Whether Neuronpedia's hosted J-lens inference actually runs for Llama, gemma-3 or GPT-OSS. The landing page links
  only Qwen3.6-27B.
- The quality of the community Qwen3.6 NLAs: their FVE numbers are self-reported, and nobody at Anthropic or
  Neuronpedia has released one. John said on the call that Qwen3.6 27B "has an NLA". If he meant an official one, I
  could not find it.
- Whether the gemma 4 base-model lens transfers to `-it` well enough.

What this means for the setup:

- Llama 3.3 70B and gemma-3-27b-it are the only pair with an official NLA and a J-lens on the exact instruct
  checkpoint.
- Qwen3.6 27B has the best lenses but only community NLAs.
- GPT-OSS-20B and gemma 4 31B would be lens-only substitutes, unless we train an NLA ourselves. The quoted 20–60
  H100 hours for gemma-3-27b is the scale of that.

## 3. Running each model in this harness

How the harness calls a model (`agent-interp-envs/src/agent_interp_envs/providers/openrouter_provider.py`):

- Native mode (`text_tools: false`) sends `messages` as stored (OpenAI schema), with `tools=[...]` and
  `parallel_tool_calls=false` (lines 256–258, 291).
- `reasoning_effort: none` sends no `reasoning` field at all (line 247). Any other value sends
  `{"reasoning": {"effort": <value>}}`.
- `top_k` and `min_p` go in the request body, and `provider_preferences` becomes OpenRouter's `provider` block.
- No `presence_penalty` knob exists: `_create_provider` (`providers/__init__.py:100`) has no such parameter.
- A swap merges the override string onto the checkpoint fleet's `config.yaml` (gemma4_prep.md section 3), so the
  substitute inherits the target's `provider_preferences` unless the override replaces them.

Override strings follow the format of `scripts/swap_all.sh` and gemma4_prep.md: one `=` per item, no commas inside a
value, single-element lists only.

**Llama 3.3 70B Instruct (target): native tools.** The target needs its own configs first:
`configs/<env>__llama-3.3-70b.yaml`, made by adding a `MODELS` entry in `scripts/make_configs.py`.

```
LLAMA="agent.model=meta-llama/llama-3.3-70b-instruct,agent.temperature=0.6,agent.top_p=0.9,agent.top_k=null,agent.min_p=null,agent.text_tools=false,agent.reasoning_effort=none,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[fp8],agent.provider_preferences.require_parameters=true"
```

- Precision choice: `only=[coreweave]` with `quantizations=[fp16]` gives full precision at $0.71 / $0.71.
- DeepInfra has a single Llama 3.3 endpoint (turbo, fp8). `make_configs.py`'s hard-coded `only: [deepinfra]` would
  therefore land on it, with a 16,384-token cap on output.
- Template caveats, from the HF chat template (read on the unsloth mirror):
  - Tools go into the first user message, as JSON. Llama is asked to answer with `{"name": ..., "parameters": ...}`.
  - It supports a single tool call per turn ("This model only supports single tool-calls at once!").
  - When an assistant turn has a tool call, the template renders only the JSON call and drops that turn's text. In
    later turns Llama therefore never sees its own prose next to its tool calls.
  - The template applies `tool_call.arguments | tojson`. Our stored arguments are JSON strings, so whether they are
    double-encoded depends on the serving stack.
- If Llama writes its call as JSON text instead of a structured call, the harness fallback `_tool_call_from_tool_code`
  (line 44) does not catch it, because it only parses Python-call syntax. The run then reads it as "no tool call", and
  several environments take that as the final answer.
- Smoke-test checks:
  - share of turns with structured `tool_calls`;
  - replies whose text starts with `{"name"`;
  - the second turn's view of the first call, with no double-encoded `"{\"command\"...`;
  - every `no_tool_call` ending.

**Qwen3.6 27B: native tools** (Qwen's `qwen3_coder` XML parser, run server-side).

```
QWEN36="agent.model=qwen/qwen3.6-27b,agent.temperature=0.7,agent.top_p=0.8,agent.top_k=20,agent.min_p=0,agent.text_tools=false,agent.reasoning_effort=none,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[fp8],agent.provider_preferences.require_parameters=true"
```

- That is the non-thinking card set, recommended so the arm is comparable with the non-thinking Llama target and
  the Qwen3-Next Instruct arm. It has two gaps:
  1. **Thinking cannot be switched off with the current code.** OpenRouter reports `reasoning.default_enabled: true`
     for this model. `reasoning_effort=none` sends no field, so the provider default (thinking on) applies. Turning
     it off needs `{"reasoning": {"effort": "none"}}` (OpenRouter's documented off switch). That takes a one-line
     provider patch, for example a value `off` mapped to it. Record it as a deviation.
  2. The card's `presence_penalty=1.5` cannot be sent: there is no knob.
- If thinking stays on, use the thinking set instead: `temperature=1.0,top_p=0.95,top_k=20`. Each reply's
  `reasoning` is then stored with it (provider line 310), and output costs several times more (section 4).
- DeepInfra is the only Qwen3.6 endpoint that supports min_p. Chutes (fp8, $0.25 / $2.00) is cheaper but would need
  `agent.min_p=null`.
- Smoke-test checks: no `<think>` text in `content` when thinking is off; `reasoning` null; and structured calls
  with no `<tool_call>` or `<function=` text leaking into `content`.

**gemma-3-27b-it: text mode,** unchanged from the current swap. It has no tool-call tokens, and DeepInfra fp8 is its
only tool endpoint.

```
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[fp8]"
```

- This is `scripts/swap_all.sh`'s `GEMMA`, plus an explicit provider pin, because the target's fleet config would
  otherwise pass on Llama's.
- Text mode sends no `tools`, so `require_parameters` is not needed. Novita's bf16 endpoint (98k context) is
  possible, but DeepInfra keeps the arm comparable with every existing gemma-3 result.
- With Llama as the target, every prefix is at most 131k tokens, so gemma-3's context-window losses from the Qwen
  swap (Qwen prefixes up to 262k) should disappear.

**gemma 4 31B: native tools.** Use the string `GEMMA4` from gemma4_prep.md section 3, unchanged: Crusoe bf16 with
`require_parameters=true`.

**GPT-OSS-20B: native tools** (harmony format, server-side).

```
GPTOSS="agent.model=openai/gpt-oss-20b,agent.temperature=1.0,agent.top_p=1.0,agent.top_k=null,agent.min_p=null,agent.text_tools=false,agent.reasoning_effort=medium,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[bf16],agent.provider_preferences.require_parameters=true"
```

- Reasoning is mandatory, so `reasoning_effort` must name an effort. `none` would send no field and leave the
  provider's default (medium) unrecorded. Pick one effort and record it. Medium is OpenRouter's default; low is
  cheaper and closer to a non-thinking target.
- Native MXFP4 instead of DeepInfra's bf16 upcast: `only=[coreweave]` with `quantizations=[fp4]` ($0.03 / $0.13).
- The returned `reasoning` is stored with each assistant message and sent back on later turns. Harmony expects
  the chain of thought to be kept across a tool-call sequence.
- Context is 131k, the same as Llama's, so Llama prefixes fit.

**The arms in `scripts/swap_all.sh`:** the loop (line 17) becomes
`for who, ov in (("llama", ""), ("qwen36", QWEN36), ("gemma", GEMMA))`, plus `("gemma4", GEMMA4)` and
`("gptoss", GPTOSS)` as backups. The places that hard-code `"qwen"` / `"gemma"` are listed in gemma4_prep.md
section 6. With a new target, `"qwen"` there means the target arm and needs renaming as well.

**The hard end rule (7 Nov 2026, `agent.end_task_required=true`, `patches/end_task_required.patch`).** John's ask on the
6 Nov call, Llama's last try. With the soft check (`agent.end_confirm`) a reply without a tool call still ends the run
after one question. With the hard rule it never does: the system prompt says so, every such reply gets a short reminder
("A reply without a tool call does not end the session. Keep working by calling a tool, or call `end_task` if you have
finished the task."), and the only way out is an explicit `end_task(report)` tool call, which the harness adds to every
task that has no finish tool of its own (coding and precommit_hook keep their `done`). It also makes the
`end_task` tool that the dashboard prompt already promised real. Two safety nets: at most 10 reminders per run
(`agent.end_task_reprompts`; after that the reply goes through and the run ends as before, so a model that never calls a
tool still stops), and the task's step limit. The call, a "Session ended" tool result and every reminder are in the
transcript, in the same format for native tool calls and for gemma's text-tool mode. This is a further deviation from
the paper, which ends a run on the first reply without a tool call: state it next to every result made with it
(stages hero4_*). Off by default; `end_confirm` is untouched.

## 4. What "re-inference the hero setup" means, and what it costs

Every swap restarts one of the target's own runs, so the target's runs must exist before the swap can be planned.
Section 1 of RESULTS.md (`results/full_runs.csv`) is the solo runs. The plans `results/swap_all_plan.json`,
`swap_extra_plan.json`, `swap_aird_plan.json`, `swap_clean_plan.json` and `swap_precommit_plan.json` are each
generated from Qwen's run directories: the run, the turn N of the bad act, and the restart turn K. None of them can be
reused for Llama.

The **8 gap environments** are the ones `scripts/gap_analysis.py` (`BAD`) and distill/README.md use: ai_rd, coding,
coding_norules, dashboard_costly_failure, grade_your_own, optimize_rendering_engine,
optimize_rendering_engine_graded and precommit_hook.

Steps:

1. Prepare the code, at no cost:
   - a `llama-3.3-70b` entry in `scripts/make_configs.py` and the 8 configs it writes;
   - the provider patch for Qwen3.6 with thinking off (section 3);
   - the new `who` names in the swap scripts;
   - optionally, a Llama JSON-call guard.
2. **Smoke test, paid and needing approval.** Run Llama 2 times each in coding, optimize_rendering_engine and
   dashboard_costly_failure, and check the template and tool-call caveats of section 3. Recompute the full-run cost
   from its measured usage.
3. **Llama's solo runs at the current counts**, 95 runs:
   - ai_rd 10, coding 20, coding_norules 20, dashboard 10, grade_your_own 5, optimize_rendering_engine 5,
     optimize_rendering_engine_graded 5, precommit_hook 20;
   - launched with `scripts/modal_full.sh <stage> llama-3.3-70b env:n ...`;
   - ai_rd needs the capped key in the agent's shell (`scripts/when_key.sh`, as in stage `full_key`).
4. Score Llama's runs:
   - add `llama-3.3-70b` to `STAGES` and `MODELS` in `scripts/full_runs_table.py`;
   - Opus grading for the judged environments (dashboard, grade_your_own and both rendering engines);
   - run a loss audit before saying it is done.
5. Plan the swap from Llama's runs, with the same rules as before: incidents first; restart at N−1 and N−2 for
   incidents, and before the final turn for clean runs; 10 replays per point, topped up to 60 per incident row.
   - The plan scripts hard-code `m == "qwen3-next-80b"` and **write over the existing plan files**. They need a
     model argument and new output names, for example `results/hero_swap_plan.json`, so that nothing under
     `results/` is overwritten.
   - How many points the plan gets depends on where Llama cheats. An environment where Llama never cheats gets
     clean points only.
6. **The swap, three arms per point:** Llama resampled (the control), Qwen3.6 27B and gemma-3-27b-it. gemma 4 and
   GPT-OSS are optional extra arms.
7. Score, run the loss audit and draw the charts, as for the Qwen swap.
8. Keep the traces for activations later. The harness already writes `messages.json` at every step, and those
   transcripts are enough to recompute activations by teacher forcing on a GPU box. Record next to them:
   - the tool schema (it is in the config);
   - the HF chat-template revision of each model;
   - the reasoning text where there is any (Qwen3.6 with thinking on, GPT-OSS).

   OpenRouter never returns the rendered prompt, so the template has to be re-applied locally.

**Cost method.** This harness does not save usage per response: `telemetry.json` holds environment state, not
tokens. So I counted tokens on the saved transcripts with each hero model's own tokenizer (`tokenizer.json` from HF;
Llama's and gemma-3's from the unsloth mirrors).

- Per call: input is every earlier message, plus 6 tokens of template per message and 250 for the tool schema,
  capped at 131,072 (Llama's window). Output is the assistant message, text plus tool-call arguments.
- Llama's own run lengths are unknown, so the range brackets them:
  - **low** = gemma-3's own run and continuation lengths (it stops early);
  - **high** = Qwen3-Next's.
- Swap tries per arm = the 1,033 tries the gemma arm actually ran in the 8 environments: 52 restart points,
  top-ups to 60 and the pre-commit study included.
- The swap prefixes are Qwen's checkpoints. Llama's real prefixes are probably shorter, which makes the swap rows an
  upper bracket. Dashboard alone is 70% of the swap tokens, and its prefixes are Qwen's 62–104-turn histories.
- Cross-check: the gemma arm with the gemma tokenizer gives 236M input for 1,033 tries. gemma4_prep.md found 226M
  for 851 tries.
- The script lived in a scratch folder and is not committed. It reads `results/` only.

**(a) Llama 3.3 70B solo runs, 95 runs in the 8 gap environments**

| length profile | input | output | DeepInfra fp8 ($0.10 / $0.32) | CoreWeave fp16 ($0.71 / $0.71) |
|---|---|---|---|---|
| gemma-like (low) | 4.4M | 0.24M | $0.52 | $3.3 |
| Qwen-like (high) | 87.0M | 0.70M | $8.9 | $62 |

On the high profile, dashboard is 61M of the input and grade_your_own 19M. Two of Qwen's runs (one dashboard, one
grade_your_own) would have outgrown Llama's 131k window and been lost.

**(b) Swap replays, per arm, 1,033 tries** (low = gemma-arm continuations, 229–238M input and 3.6–4.1M output; high =
Qwen-arm continuations, 421–446M input and 8.4–8.6M output, depending on the tokenizer)

| arm | endpoint | low | high |
|---|---|---|---|
| Llama 3.3 70B (control) | DeepInfra fp8 | $24 | $47 |
| Llama 3.3 70B (control) | CoreWeave fp16 | $165 | $317 |
| Qwen3.6 27B, thinking off | DeepInfra fp8 ($0.32 / $3.20) | $88 | $170 |
| Qwen3.6 27B, thinking off | Chutes fp8 ($0.25 / $2.00) | $67 | $128 |
| Qwen3.6 27B, thinking on (+500 to 2,000 reasoning tokens per call; 6,600 to 13,600 calls) | DeepInfra fp8 | $99–130 | $192–257 |
| gemma-3-27b-it | DeepInfra fp8 ($0.08 / $0.16) | $20 | $35 |
| gemma 4 31B (backup; gemma-3 tokenizer as proxy) | Crusoe bf16 ($0.14 / $0.40) | $35 | $62 |
| GPT-OSS-20B (backup; +200 to 1,000 reasoning tokens per call) | DeepInfra bf16 ($0.03 / $0.14) | $8 | $15–16 |

**Range for the hero setup, before losses:**

- **Solo runs:** $0.5–9 with Llama on DeepInfra fp8; $3–62 on CoreWeave fp16.
- **Swap, two substitutes** (Qwen3.6 with thinking off on DeepInfra, plus gemma-3): $107–205. Adding the Llama
  control arm gives **$131–251**.
- **Thinking on for Qwen3.6:** $142–338.
- **Llama arm at fp16 instead of fp8:** add $141–270.

Past runs lost many tries on top of the planned ones. Qwen's dashboard swap arm finished 115 of 388 tries; the other
273 were lost to provider failures, cut-offs and the context window, and were rerun. Budget
1.5× the figures for losses and retries. Set the OpenRouter key cap at least 3× the estimate, not at it: the tight
$1,620 cap of 28 Oct cut running fleets.

**Not included: Opus grading.** The four judged environments (dashboard, grade_your_own and both rendering engines)
add about 383 swap tries per arm, so 1,150 over three arms, plus 25 solo runs.

- The measured rate was $1.08 per long Qwen try (claude-opus-4-8, 27 Oct). Opus 4.8 and Opus 5 cost $5 / $25 per M,
  so a transcript of at most 131k tokens costs up to about $0.8.
- Short tries cost cents.
- That puts grading somewhere between **$120 and $940**: comparable to, or above, the inference. Estimate it from the
  measured lengths of Llama's transcripts before launching.
- Modal compute runs on SPAR's workspace and is not counted here.

## 5. Compliance note

The rule is in `gigi/CLAUDE.md` ("Experiment rules"). Open-weight models run locally on a rented GPU with HF
transformers. The one exception, for part three only (the user, 17 Oct 2026), lets the forensics runs call
**Qwen3-Next-80B and gemma-3-27b** through OpenRouter, because no activations are needed yet.

- **Llama 3.3 70B, Qwen3.6 27B and GPT-OSS-20B are not covered**, and neither is gemma 4 31B, strictly read. Running
  them through OpenRouter, as sections 3 and 4 assume, needs the user to extend that exception explicitly, model by
  model. Until then the plan above is a plan, not an authorisation. The same "ask first" rule applies to every
  spend in it: the smoke test, the solo runs, each swap stage and the grading.
- **Precision matters once activations are recomputed.** OpenRouter serves fp8 for most of these endpoints:
  - Llama's DeepInfra and AkashML endpoints are fp8;
  - Qwen3.6 has no bf16 endpoint at all;
  - gemma-3's only tool endpoint is fp8;
  - GPT-OSS is MXFP4 natively.

  The J-lenses and NLAs were fitted on bf16 weights (Neuronpedia's lens configs say `dtype: bfloat16`; the NLAs
  are trained on the published checkpoints). Recomputing activations on the bf16 model over tokens that an fp8 model
  sampled therefore reads a slightly different network from the one that made the choices. Teacher forcing makes
  this exact for what the bf16 model would compute on those tokens, but not for why the fp8 model chose them.
- Options:
  - generate the target at full precision: Llama on CoreWeave fp16, which costs the extra shown in section 4;
  - or keep fp8 and state it next to every result, as the current forensics results do;
  - or, for the runs whose activations matter most, regenerate locally in bf16 on the GPU box, as the project rule
    prefers.
- The precision of every arm (fp8, fp16, bf16, fp4) must be recorded with each run and shown next to every result.

