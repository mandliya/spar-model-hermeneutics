# gemma 4 31B swap: preparation (3 Nov 2026)

Decision of 28 Oct: stay on gemma-3-27b; make the switch to gemma 4 31B free to decide later. John's item: compare
Gemma 4's native function-calling format with Qwen's and spec the prefix translation. Short answer: none is needed
(section 3). Sources checked today: Google model card, HF card + canonical chat template, Google function-calling
doc, OpenRouter `/models` and `/models/google/gemma-4-31b-it/endpoints` (all free reads; no paid call made).

## 1. What gemma 4 31B is

- `google/gemma-4-31b-it`: dense, 30.7B parameters, 60 layers, text+image in, text out; listed on OpenRouter
  2 Apr 2026 (endpoint tag `-20260402`); technical report arXiv 2607.02770; canonical chat template 9 Jul 2026.
  Knowledge cutoff Jan 2025. https://ai.google.dev/gemma/docs/core/model_card_4 · https://huggingface.co/google/gemma-4-31b-it
- Context 256K (262,144 on OpenRouter) vs gemma-3's 128K: removes the 181 context-window losses of the swap
  (none in a gap cell except 3 dashboard + 1 ai_rd tries).
- Native structured tool use: tokens `<|tool>declaration:…<tool|>` (system turn), `<|tool_call>call:NAME{arg:<|"|>v<|"|>}<tool_call|>`,
  `<|tool_response>response:NAME{…}<tool_response|>`; turn markers `<|turn>model … <turn|>`. Brace format, not JSON
  (unquoted keys, strings in `<|"|>`). https://ai.google.dev/gemma/docs/capabilities/function-calling
- Thinking mode: on only when `<|think|>` opens the system prompt (template default `enable_thinking=false`);
  output `<|channel>thought\n…<channel|>` before the answer; with thinking off the 31B still emits an empty block.
  Recommendation: off (Qwen3-Next Instruct is non-thinking, gemma-3 had none) = keep `reasoning_effort: none`.
- Model-card sampling: temperature 1.0, top_p 0.95, top_k 64, no min_p. Identical to gemma-3's card: the swap's
  gemma sampling block carries over unchanged.
- Sibling `google/gemma-4-26b-a4b-it` (MoE, 3.8B active, $0.064/$0.19) is not the stand-in: the 31B dense is
  the like-for-like successor of the 27B dense.

## 2. Provider choice on OpenRouter

Tool-capable endpoints for `google/gemma-4-31b-it` (OpenRouter endpoints API, 3 Nov 2026; $ per M tokens):

| provider (`only:` slug) | quant | context | max out | in | out | notes |
|---|---|---|---|---|---|---|
| Crusoe (`crusoe`) | bf16 | 262,144 | 262,141 | 0.14 | 0.40 | supports top_k, min_p, seed; uptime 100% at check |
| Novita (`novita`) | bf16 | 262,144 | 131,072 | 0.14 | 0.40 | no min_p; 30-min uptime 83% at check |
| DeepInfra (`deepinfra`) | fp8 | 262,144 | 16,384 | 0.20 | 0.40 | our host for Qwen fp8 and gemma-3 fp8; shared pool gave the 429s |
| DeepInfra turbo | fp4 | 262,144 | 16,384 | 0.135 | 0.51 | |
| Parasail (`parasail`) | fp8 | 262,144 | 235,929 | 0.15 | 0.40 | |
| CoreWeave (`coreweave`) | fp4 | 262,144 | 235,929 | 0.10 | 0.34 | cheapest |
| PrimeIntellect | fp8 | 262,144 | 16,384 | 0.288 | 0.482 | |
| others: DekaLLM/Friendli (quant unknown, 0.14/0.40, Friendli max out 8k), NextBit fp4, Venice fp4 (8k out), Chutes fp4 (131k ctx), SambaNova (131k, 0.38/1.15), Io Net, ModelRun, SiliconFlow (0.75/1.00) | | | | | | |

DeepInfra also lists two fp8 endpoints without tools (0.16/0.32 flex, 0.30/0.60 priority): `only: [deepinfra]`
alone could land there, hence `require_parameters: true` below.

- Recommendation: pin bf16 on Crusoe (Novita as the one alternate if Crusoe is down on launch day; same price).
  Reasons: bf16 is the published model, so gemma 4's behaviour cannot be blamed on quantization the way gemma-3
  fp8 could; cheapest tool-capable 262k endpoint with 64k+ output; not the DeepInfra pool that produced the
  429 storms and Qwen's repetition collapses. Qwen stays fp8 on DeepInfra: it is the fixed reference arm (already
  run), and the question is gemma 4's choice on Qwen's history, not a precision-matched capability test. Keeping
  DeepInfra fp8 for "same host as everything else" costs +40% ($47 vs $33) and caps output at 16k; not worth it.
- Exact block (replaces the Qwen fleet config's `only: [deepinfra]` at resume time, see section 3):

```yaml
provider_preferences:
  only: [crusoe]            # alternate: [novita]; both bf16, $0.14/$0.40, 262k context
  allow_fallbacks: false
  quantizations: [bf16]
  require_parameters: true  # never a tool-less or min_p-less endpoint
```

## 3. Override string, config, and the prefix-translation question

- `scripts/swap_all.sh`: add a third `(who, ov)` pair `("gemma4", GEMMA4)` next to `("gemma", gemma)` (line 17);
  results land in `results/raw/swap_all/<env>/<point>/gemma4/`. The string (one `=` per item, no commas inside
  a value, so single-element lists only; verified to parse with OmegaConf: lists, `false`, `null`):

```
GEMMA4="agent.model=google/gemma-4-31b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=false,agent.reasoning_effort=none,agent.provider_preferences.only=[crusoe],agent.provider_preferences.quantizations=[bf16],agent.provider_preferences.require_parameters=true"
```

- Config: none to edit. `resume_modal.py` reads the checkpoint fleet's `config.yaml` (Qwen's: openrouter,
  `only: [deepinfra]`, `allow_fallbacks: false`, `reasoning_effort: none`, no `text_tools`) and merges the
  override list on top (`_step_config`, `run_common.load_config`); `allow_fallbacks: false` survives the merge.
  A `configs/<env>__gemma-4-31b.yaml` copy is only needed for gemma 4's own full runs, not for the swap.
  Plan: all 91 points fit 262k, so `fit_for_gemma` filtering is moot; keep `--timeout 7200`.
- Does Qwen's prefix need translating? No. Verified in `openrouter_provider.py`: `invoke()` sends
  `messages=self.messages` verbatim when `text_tools` is false (line 291); the stored `messages.json` is the
  OpenAI schema whoever produced it (checked a dashboard K=63 checkpoint: 130 messages, assistant turns with
  `tool_calls:[{id, type:function, function:{name, arguments:<JSON string>}}]`, `tool` turns with `tool_call_id`).
  `_text_tools_messages` (the ```tool_code``` rewrite) runs only in text mode.
- What the request for gemma 4 contains: `model=google/gemma-4-31b-it`; `messages` = the checkpoint's list as
  stored (system prompt incl. the `end_task` sentence, user task, Qwen's assistant turns with their tool_calls and
  null extras `refusal/annotations/audio/function_call/reasoning/index`, tool results); `tools=[execute_command]`;
  `parallel_tool_calls=false`; `temperature=1.0`, `top_p=0.95`; body extras `provider={…}`, `top_k=64`; no
  `reasoning` field, no `min_p`. The serving stack renders it with Gemma 4's template: the `tools` list becomes
  `<|tool>declaration:execute_command{…}<tool|>`, each stored Qwen call becomes
  `<|tool_call>call:execute_command{command:<|"|>…<|"|>}<tool_call|>`, each tool message becomes
  `<|tool_response>response:execute_command{…}<tool_response|>` inside the model turn. Qwen's past calls look to
  gemma 4 exactly like its own. The template requires `function.arguments` to be a JSON string or dict: ours are
  strings. The same stored messages already go to Qwen natively, so the schema is proven through OpenRouter.
- Format comparison for the record. Qwen3-Next (Hermes style, from its tokenizer_config chat template):
  tools as JSON in `<tools>…</tools>` in the system turn; call `<tool_call>\n{"name": N, "arguments": {…}}\n</tool_call>`;
  result `<tool_response>…</tool_response>` in a user turn; turns `<|im_start|>role … <|im_end|>`. Gemma 4: section 1.
  gemma-3: no tokens, ```tool_code``` / ```tool_output``` prose (what our text mode emulates).
- If a raw-completion prefill were ever used (it is not: OpenRouter's chat API never exposes raw tokens), John's
  "regex" would have to map `<|im_start|>/<|im_end|>` to `<|turn>…<turn|>`, the `<tools>` JSON to brace
  declarations with upper-case types, each `<tool_call>` JSON to `call:N{k:<|"|>v<|"|>}` (re-escaping quotes,
  nested objects), and move each user-turn `<tool_response>` into the model turn. Nested braces and quoting make
  it a parser, not a regex. Moot: the chat API does it server-side from the neutral JSON.

## 4. Harness decisions before launching

- `end_task` mismatch (prompt names it, `environments/dashboard_perf/tools.py` offers only `execute_command`, on
  purpose). The prompt is inside the stored prefix, so it cannot be changed without changing Qwen's history.
  Native tools do not remove it: gemma 4 will emit a structured `end_task` call; `validate_and_get_command`
  treats it as `unknown_function`, reverts and resamples up to 3 times, then returns the tool result
  `Exit code: 1 / Unknown function: end_task. Available: execute_command` and the run goes on, usually into a
  prose report = done. This is the path Qwen's own arm already takes (same code, native). Recommendation: do not
  add an `end_task` tool (it would change the environment relative to the Qwen control bars and the gemma-3 arm);
  keep the ending labels as computed and the RESULTS.md note (gemma-3's 17 `end_task` endings were honest
  shortfall reports). Cost note: each resample re-sends the whole ~0.3M-token dashboard history (~$0.04 bf16).
  Smoke test: count `end_task` calls per dashboard try, what follows each (report vs loop), resamples per try.
- No tool call = final report (`treat_no_tool_call_as_done=True`, `dashboard_perf/run_step.py:89`). Keep: it is the
  environment's semantics and the Qwen arm ran under it; a nudge would be a new environment. Native tools make an
  accidental prose reply rarer. One guard to extend: `_MALFORMED_REPLY_RE` (`run_step.py:80`,
  `default_api|execute_command\s*\{|<call[:_]|```tool_code`) does not match a leaked Gemma 4 call
  `<|tool_call>call:execute_command{…}`; add `<\|tool_call>` so a leaked-token reply gets the existing one-line
  "not a valid tool call" nudge instead of ending the run. It fires only on malformed call text, so it is a format
  guard, not a semantics change; record it as a deviation. Smoke test: grep every gemma 4 reply for
  `<|tool_call>`, `call:`, ```tool_code```, `<|channel>`; read every `no_tool_call` ending.
- Format slips (prose instead of a call, code pasted as text: 16 of gemma-3's 40 stopped tries, 3% of gap-cell
  tries). Native tools: the server parses the call; a malformed one surfaces as `invalid_arguments` and is
  resampled (3×), more than one call as `parallel_tool_calls` (we send `parallel_tool_calls=false`, but it is not
  in Crusoe's/Novita's supported parameters, so the harness path matters). Likely removes most. Note the
  `_tool_call_from_tool_code` fallback still runs in native mode (provider line 302), as it does for Qwen.
  Smoke test: validation errors per try (`Unknown function`, `Invalid arguments`, parallel) ~0 apart from `end_task`;
  no `quick_answer` ending whose text contains code.
- Thinking: send no `reasoning` field (as now). Smoke test: response `reasoning` is null and no `<|channel>` text
  leaks into content; if a provider turns thinking on by default, pin the other bf16 provider.

## 5. Smoke test plan

- 3 environments, the `incident: true` point at N (Qwen's cheating turn) from `results/swap_all_plan.json`:
  coding, optimize_rendering_engine, dashboard_costly_failure; `--count 2` each; `who=gemma4`; a 3-line command
  file for `swap_all.sh` (it accepts one as `$1`), `--timeout 7200`, pinned provider as above.
- Look at, per try: first turn after takeover is a structured call (`tool_calls` non-empty, id not
  `call_unparsed_`, arguments valid JSON with `command`); every call parses; no `end_task` before the final report;
  ending type via `gemma_endings.classify` (expected: coding `quick_answer`, rendering/dashboard `gave_up` or
  `says_done`); no `[provider retry]` storm; `usage.prompt_tokens` of the first dashboard call vs the 0.3M/try
  estimate (recompute the full-run cost from it before launching).
- Cost: dashboard 2 × ~0.3M + rendering 2 × ~0.12M (27k prefix × ~6.5 turns) + coding 2 × ~5k ≈ 0.9M input,
  ~15k output; plus up to ~0.3M for `end_task` resamples. bf16 ≈ $0.15, DeepInfra fp8 ≈ $0.25, fp4 ≈ $0.10.
  Under $0.50 on any endpoint.
- Full run, every gemma gap-environment try (851 tries; 226M input + 3.5M output, 81% of input in dashboard);
  "gap cells" = the six gap environments of `gap_analysis.py` (coding, coding_norules, optimize_rendering_engine,
  dashboard_costly_failure, ai_rd, precommit_hook; 763 tries, ≈95% of the tokens since the dropped graded-rendering
  and grade_your_own tries are short):

| endpoint | $/M in / out | all 851 tries | gap cells (6 envs) |
|---|---|---|---|
| Crusoe or Novita bf16 | 0.14 / 0.40 | $33 | $31 |
| Parasail fp8 | 0.15 / 0.40 | $35 | $34 |
| DeepInfra fp8 | 0.20 / 0.40 | $47 | $45 |
| DeepInfra turbo fp4 | 0.135 / 0.51 | $32 | $31 |
| CoreWeave fp4 | 0.10 / 0.34 | $24 | $23 |

  Set the OpenRouter key limit at ≥3× the estimate ($100 for bf16): the $1,620 cap of 28 Oct cut running fleets.
  Retries, `end_task` resamples and the 262k window letting dashboard tries run longer all push the bill up.

## 6. Labelling and charts on a third model name

Everything keys on the `who` string (`qwen` / `gemma`) from the results path `results/raw/swap_all/<env>/<point>/<who>/`,
so a run under `who=gemma4` flows through once the literals below know the name. `intent_labels.py` has no model
dict: it reads `who` from `results/gap_replays.csv` (lines 186, 259, 269, 294) and labels only the model's own
turns; its line 138 ```tool_code``` regex is text-mode only and harmless. Places to add `gemma4`:

- `scripts/swap_all.sh:10,17`: `GEMMA` string; `for who, ov in (("qwen", ""), ("gemma", gemma))`.
- `scripts/swap_all_score.py:85,101`: `for who in ("qwen", "gemma")` (download and score); `:119` `fit_for_gemma`.
- `scripts/gap_replays.py:16`: `for who in ("qwen", "gemma")` → rows with `who=gemma4`.
- `scripts/swap_endings.py:25`: group name `"gemma's own full runs"` (classification itself is model-agnostic).
- `scripts/gap_analysis.py:18` `MODEL_COL`; loops `("qwen", "gemma")` at 45, 65, 136, 175, 199, 256, 275, 337;
  labels at 225 (`gemma (gemma-3-27b)`), 346 (`gemma 128k`), 356 (legend `gemma-3-27b`).
- `scripts/propensity_chart.py:17` `MODEL_COL`; 53, 63, 78 loops; 87–98 column headers and title text.
- Not on the swap path: `scripts/charts.py:24-26,47` and `scripts/analyse.py:30-31` (`Q`, `GM`, own full runs).
- Suggested shape: one env var `SWAP_WHO=gemma4` read in `gap_replays.py`, `gap_analysis.py`, `propensity_chart.py`
  in place of the literal `"gemma"`, plus a display dict `{"gemma": "gemma-3-27b", "gemma4": "gemma-4-31b"}`;
  charts then show gemma 4 as the bottom bar, or a third bar if both gemmas are kept (chart 5F/5L layouts assume two).
