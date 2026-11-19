# Phase 2 runbook: adding a new target model (DeepSeek V4 Pro first) and distilling it into Qwen3.6 27B

Written 19 Nov 2026 from the code in gigi/3-forensics (no paid calls, no launches; public OpenRouter model and
endpoint listings read for free). Paths are relative to gigi/3-forensics unless stated.

## 0. Things found while reading that change the plan

1. **The paper's harness already knows these models.** agent-interp-envs ships configs for DeepSeek V4 Pro, GPT-OSS
   120B, Kimi K3 and GLM 5.3, all with `provider: fireworks` (Fireworks direct, `accounts/fireworks/models/...`):
   e.g. `configs/puppeteer/puppeteer.yaml` (deepseek-v4-pro, 300 steps), `revert_or_refactor` (gpt-oss-120b),
   `*__kimi-k3.yaml`, `*__glm-5-3.yaml` (reasoning_effort max). There is also
   `src/agent_interp_envs/providers/deepseek_v4_template.py` (DeepSeek's prompt format, reverse-engineered on
   Fireworks) and `fireworks_provider.py`, which stores the reasoning (`reasoning_content`) and prints token usage.
   The Fireworks provider defaults to `reasoning_effort: "low"` when a config sets none, so the paper's DeepSeek runs
   may have used low effort. Check this against the paper before choosing the setting.
2. **OpenRouter has no Fireworks endpoint for DeepSeek V4 Pro.** The listed endpoints are all 1M context, with tools
   and reasoning: deepinfra fp8 ($1.30 in / $2.60 out per M), siliconflow fp8 ($0.90 / $1.70), novita fp8, alibaba fp8,
   gmicloud fp8, streamlake fp8 and others. So reproducing the paper's exact serving needs a Fireworks key.
   - The project rule says OpenRouter with a pinned provider. A Fireworks key and a Modal secret would be new, and
     GG must create both, because I may not.
   - Decide: OpenRouter pinned (for example deepinfra fp8, our usual host) or Fireworks direct (the paper's).
3. **This week's Qwen3.6 teacher runs had thinking switched off** (`chat_template_kwargs.enable_thinking: false`, the
   hero setting since 7 Nov). So "gemma overfits on Qwen's thinking" can't be right; there was no thinking text in the
   training data. (Out of scope here; worth correcting with the team.)
4. **Qwen3.6 thinking: decide before anything runs.** DeepSeek, GPT-OSS, Kimi and GLM all reason.
   - If Qwen3.6 is the substitute and the student, it should probably think too: chat_template_kwargs
     enable_thinking true, sampling 1.0 / 0.95 / 20.
   - With thinking on, the existing Qwen3.6 solo runs can't serve as the "substitute alone" baseline. They are all
     thinking-off, except 40 thinking-on runs in 4 tasks (hero2_qwen36_think). A new Qwen3.6 thinking-on fleet
     (28 x 40) is then needed. It is cheap on our own vLLM box.
5. **The training code is gemma-only.**
   - `distill_train.py` loads `Gemma3ForConditionalGeneration` with gemma-specific LoRA targets.
   - `distill_data.py` renders gemma's text-mode tool calls.
   - Qwen3.6 27B is a hybrid (Gated DeltaNet + attention). The student needs a new model class, new LoRA target
     modules (including the DeltaNet projections), the fast linear-attention kernels, and a renderer that uses
     Qwen's own chat template with native tool calls and `<think>` reasoning.
   - Qwen3-family templates normally drop the reasoning of earlier turns. The DeepSeek V4 template keeps all of it
     (deepseek_v4_template.py docstring). This is a real teacher/student context mismatch; verify it on the actual
     Qwen3.6 template.
   - Also check that vLLM supports LoRA adapters on this hybrid model.
6. **The OpenRouter provider records no usage or cost per call.** openai_provider and fireworks_provider do; ours
   doesn't. At Kimi K3 output prices ($9–16 per M) we need a per-call ledger before any fleet.
7. **"Context" is really two limits.**
   - Teacher runs can reach 1M tokens, but our training cuts every example at 32k tokens / 105k characters, so long
     DeepSeek runs are cut in training whatever the student's window is.
   - When re-running the student, the vLLM window (49k this week) and the KV cache bound how many runs fit at once
     (memory vllm-agent-fleet-kv-sizing). A 262k window means far fewer runs at once on one H200.

## 1. Files to create or change for a new target

| What | File | Change |
|---|---|---|
| Per-task configs | NEW `scripts/make_configs_targets.py` (pattern: `make_configs_qwen36.py`) | One agent block per target, written over `configs/<env>__qwen3-next-80b.yaml` (task block, prompts and the paper's `max_steps` kept, e.g. 5 for norvane up to 500 for grade_your_own). Output `configs/<env>__deepseek-v4-pro.yaml` for the 28 tasks of this week's round (same list as classic_fleet3). Never overwrite existing configs. |
| Agent block, DeepSeek via OpenRouter | (in the generator) | `provider: openrouter`, `model: deepseek/deepseek-v4-pro`, `provider_preferences: {only: [<host>], quantizations: [fp8], allow_fallbacks: false, require_parameters: true}`, `reasoning_effort: <decided: paper's level>`, model-card sampling sent explicitly (read the card and generation_config.json; not verified here), `text_tools: false` (native tools), `parallel_tool_calls: false`. |
| Agent block, Fireworks option | (in the generator) | `provider: fireworks`, `model: accounts/fireworks/models/deepseek-v4-pro` as in the paper's configs; `provider_preferences` is not allowed for non-OpenRouter providers (`providers/__init__.py` raises). Needs a FIREWORKS key and Modal secret `fireworks-secret-gigi`, which GG creates. |
| Cost ledger | `agent-interp-envs/src/agent_interp_envs/providers/openrouter_provider.py` | Send `usage: {include: true}`; after each reply append `{t, model, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, cost}` to `<run>/usage.jsonl` and print a `[provider-usage]` line (same shape as openai_provider `_log_usage`). |
| Reasoning saved and sent back | same file | It already stores `reasoning` and copies it to `reasoning_content` (line ~338). Check in the smoke test that every assistant turn in `step-*/messages.json` carries the reasoning (and `reasoning_details` if OpenRouter returns it), and that later prompt-token counts grow by the reasoning length (i.e. earlier reasoning is sent back, as DeepSeek's template expects). |
| Reasoning-only replies | same file | A reply with reasoning but no text and no tool call is currently an "empty reply" (retried 3 times, then a crash, as this week's 17 Qwen crashes). Count these in the smoke test before deciding. |
| Model tag in the tables | `scripts/full_runs_table.py` | `HERO_MODEL=deepseek-v4-pro`, `HERO_STAGES=<stages>`, `HERO_ALL_ENVS=1` (forgetting it gave the 8-task table on 17 Nov), `HERO_OUT_PREFIX=<target>_full_runs__<tag>`. |
| Counts and top-ups | `scripts/classic_counts.py`, `classic_select.py` | Reuse with the new stage names and model tag (`--expect-model`). The held-out case table in classic_select is Qwen-vs-gemma specific: rewrite the CASES for the new pair once the differences are known. |
| Student training | `scripts/distill_train.py`, `distill_data.py` | New Qwen3.6 path (point 0.5): model class, LoRA targets, kernels, Qwen-template renderer with tools and `<think>`, the same target-only loss. New `venv` pins (transformers version that supports Qwen3.6 + peft + kernels). |
| Re-runs | `scripts/classic_rerun.sh`, `distill_eval.py configs` | Serve Qwen3.6 + LoRA checkpoints on vLLM (verify LoRA on the hybrid), configs from `*__qwen3.6-27b-bf16.yaml` (thinking per the decision in 0.4), held-out list for the new pair. |

## 2. Where to launch from, keys, caps, pacing

- **Launcher.** Never the laptop: `modal run --detach` loses the fleet when the Mac sleeps or Wi-Fi drops (memories
  modal-fleets-die-when-mac-sleeps, launch-modal-from-vast-box).
  - The target generation needs no GPU. Use a cheap always-on CPU box (or the GPU box when one is rented for the
    student).
  - GG copies `~/.modal.toml` to it; I may not move Modal credentials. Then rsync agent-interp-envs, scripts and
    configs, run `uv sync`, `chown -R root:root`, and `git safe.directory`.
- **Modal.**
  - Profile `spar`, `MODAL_SECRET_SUFFIX=-gigi`, `MODAL_VOLUME_PREFIX=gigi-forensics-`.
  - Secrets: `openrouter-secret` (plus `openrouter-agent` for ai_rd), or a new `fireworks-secret` for the Fireworks
    option, which GG creates.
- **OpenRouter money, before every paid pass:**
  - Read both the key's `limit_remaining` (GET /api/v1/key, never print the key) and the workspace lifetime budget
    (memory openrouter-workspace-budget: it blocked us on 21 Oct, 11 Nov and 12 Nov).
  - Ask GG to have John raise both if needed.
  - The spend watchdog (scripts/spend_watchdog.sh) counts the whole key, so two fleets on one key trip each other.
  - Cap at about 4x the estimate (memory spend-caps-generous).
- **Pacing.**
  - Launch per task with small `MAXC`, spaced by `GAP`, through a paced queue (classic_fleet3.sh pattern, with health
    meaning the provider's state, not vLLM's).
  - During the first minutes, `grep -h "provider retry" logs/*.log | sort | uniq -c`; pause if 429 attempts pass
    about 6 (memory swap-launch-pacing).
  - Swaps go only through the paced launcher (hero9_swap_paced.sh: batches only while live containers stay under a
    limit).
- **Fetch.** Lean fetch only (rollout.log, final/, last step; `distill_fetch_lean.py` / `hero9_swap_fetch_lean.py`),
  never a whole-fleet `volume get`, never `--force`.
- **Watch.** Poll the launcher box's log over its direct SSH port with a heartbeat (memory box-watch-heartbeat).

## 3. Smoke test, before any fleet

One run per loop type, plus one long task to price the fleet. Each costs cents to about $1.

| Run | Why | Check in rollout.log, messages.json and usage.jsonl |
|---|---|---|
| coding (ends on `python submit_solution.py`) | game/submit end | the submit ends the run; the measure is readable (`submitted`) |
| tictactoe or chess (ends on game over) | game end | the run ends when the game does; moves parse as tool calls |
| eval_tampering or lazy_investigation (ends on a plain reply) | plain-reply end | a reply with no call ends it; the final answer and `suspicious_observations` are filled |
| a long task, e.g. grade_your_own (500 steps) or dashboard_costly_failure | cost and context | tokens and cost per step; prompt growth per step; whether the history approaches the window |
| ai_rd (with the openrouter-agent secret) | its own key path | the inner agent key works; no context error early |

For every smoke run:
- reasoning is present on every assistant turn and saved;
- tool calls are native and parsed, with no "Unknown function" loops;
- no empty-reply retries, or count them;
- `[provider retry]` lines are few;
- usage.jsonl holds a cost per call, summed per run;
- the run ends the way the task says;
- `distill_data.run_end` classifies it as finished.

Then the estimate: median cost per run per task x 46 runs x 28 tasks, plus top-ups (about 15%). State it to GG with
the cap, and wait for a yes.

## 4. Ordered runbook for step 1 (DeepSeek V4 Pro into Qwen3.6 27B)

| # | Stage | Reuse | New | Rough wall clock |
|---|---|---|---|---|
| 0 | Decide: provider (OpenRouter pinned host vs Fireworks), DeepSeek reasoning effort (paper's), Qwen3.6 thinking on or off, the student's re-run window | hero_models_prep.md method | short decision note in distill/PLAN_<target>.md | half a day (call/Slack) |
| 1 | Configs for 28 tasks | make_configs_qwen36.py pattern | make_configs_targets.py | 1 h |
| 2 | Harness patches: usage ledger, reasoning checks | openai_provider._log_usage shape | patch in openrouter_provider.py (+ HARNESS_CHANGES.md row) | 2-3 h |
| 3 | Launcher box + keys + budget check | launch-modal-from-vast-box steps | — | 1 h, GG copies .modal.toml |
| 4 | Smoke (section 3) and cost estimate; GG approves | modal_full.sh (MAXC=1) | small smoke script | 2-4 h |
| 5 | Fleet: 28 tasks, 40 complete runs each (46 launched for long tasks), paced, top-ups to 40, set-aside rule for runs cut by our or the provider's failures (timeouts, 429 give-ups) | modal_full.sh, classic_fleet3.sh (queue, top-ups), classic_counts.py, set_aside.tsv | provider-health gate in place of vLLM metrics | 12-36 h (provider-bound; 500-step tasks with reasoning are the tail) |
| 6 | Lean fetch, per-run table (HERO_ALL_ENVS=1), loss audit, results table | distill_fetch_lean.py, full_runs_table.py --hero, classic_select.py | new CASES | 2-4 h |
| 7 | Qwen3.6 substitute baseline. Thinking off: the 40 bf16 runs per task we have. Thinking on: a new fleet on our vLLM box | classic_fleet3.sh on the box | `*__qwen3.6-27b-think-bf16.yaml` | 0 h or 12-20 h on a GPU box |
| 8 | Differences, DeepSeek vs Qwen3.6, per task; choose held-out tasks (one each: D>Q, Q>D, equal), fixed before training | classic_select.py CASES, PLAN_classic.md section 2 | — | 2 h |
| 9 | Swap runs (symmetric at N and N-1, both directions) | hero9_swap_plan.py, hero9_swap.sh (override strings), hero9_swap_paced.sh, hero9_swap_lean_foldin.sh, grading with Opus 4.8 (memory swap-grading-opus48) | DeepSeek override string; reasoning in the prefix (keep or strip: decide) | 1-2 days |
| 10 | Classic distillation into Qwen3.6 27B: build data (80/20, held-out tasks only in test), train LoRA 2 epochs with checkpoints every half epoch, upload each to HF | classic_after.sh flow, distill_data.py build checks | Qwen3.6 renderer and trainer (0.5); new venv; training smoke at max length | build 1 h; train 15-30 h on an H200 (untested for the hybrid model) |
| 11 | Re-runs of Qwen3.6 before training and 4 checkpoints on the held-out tasks + 2 trained tasks; top-ups; spend gate | classic_rerun.sh, distill_eval.py, classic_prelim.py | Qwen LoRA serving flags | 6-12 h (longer window means fewer runs at once) |
| 12 | Results: rates, clean-data view, run health, loss on held-out tasks, per-task steps, charts with axis labels, results page, TQ handoff | classic_results.py, classic_clean.py, classic_health.py, classic_call_charts.py, classic_defer_nll.sh | — | half a day |

Steps 2-3 and 6 repeat as-is for Qwen3-Next 80B, GPT-OSS 120B, Kimi K3 and GLM 5.3:
- **Qwen3-Next 80B.** Configs exist (`*__qwen3-next-80b.yaml`). The October and hero2 traces (OpenRouter fp8, no
  reasoning: it is an Instruct model) cover about 10 runs per task, so top up to 40 with the same configs.
- **GPT-OSS 120B.** 131k context, about half of Qwen3.6's 262k; cheap (bf16 at deepinfra $0.04 / $0.17).
- **Kimi K3.** Output $9–16 per M; price the smoke carefully.
- **GLM 5.3.** Cheap fp4 hosts; the fp8 or bf16 choice matters.

## 5. Could not verify

- The paper's exact reasoning effort and sampling per model (Fireworks defaults to "low" when unset).
- Whether OpenRouter passes earlier reasoning back to DeepSeek V4 Pro on each host.
- Whether Qwen3.6's chat template keeps or drops earlier turns' reasoning.
- LoRA training and vLLM LoRA serving support for the Qwen3.6 hybrid at our versions.
- The per-run cost of DeepSeek in our tasks. Only the smoke test can measure it.
