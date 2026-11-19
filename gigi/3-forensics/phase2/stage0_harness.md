# Phase 2, stage 0, harness side: what was built (19 Nov 2026)

Nothing was spent, launched or committed. Only free calls were made: OpenRouter's endpoint list, `/api/v1/key` and
`/api/v1/credits`, and the Hugging Face model cards. Every path below is under `gigi/3-forensics/`.

## Files created or changed

| File | What |
|---|---|
| `agent-interp-envs/src/agent_interp_envs/providers/openrouter_provider.py` (the harness clone) | **The usage ledger.** Requests to OpenRouter ask for `usage: {include: true}`. After every reply, one `[provider-usage] t= model= served_by= in= cache_read= out= reasoning= cost=` line goes to the run's log, BEFORE any check that may discard the reply, so billed retries are counted. The same record is kept on the stored assistant message as `_usage` (in every step's messages.json) and stripped from every request (`_wire`). vLLM replies with only `reasoning_content` are mirrored into `reasoning`. Empty-reply errors now say how many reasoning characters came with them. Requests to our own vLLM server (`base_url` set) send no `usage` field. |
| `harness_patch/changes.patch` | Refreshed. It now also holds the 17 Nov classic knobs in `providers/__init__.py`, which the 12 Nov copy lacked. **Verified:** a fresh clone at `base_commit` with the patch and `new_files/` reproduces the clone's `src/` exactly. |
| `HARNESS_CHANGES.md` | A new section, "Phase 2 additions", describing the above. |
| `scripts/make_target_configs.py` (new) | Writes `configs/<task>__<tag>.yaml` for the 28 classic tasks. Reusable for GPT-OSS, Kimi and GLM, whose settings refuse to write until someone confirms them. Never overwrites a file it did not write. `--check` compares each config with the classic config. |
| `configs/<task>__deepseek-v4-pro.yaml` (28, new) | `deepseek/deepseek-v4-pro` (April checkpoint, the paper's), pinned `only: [siliconflow]`, `quantizations: [fp8]`, `allow_fallbacks: false`, `reasoning_effort: xhigh`, temperature 1.0 and top_p 1.0 (model card), native tools, `parallel_tool_calls: false`, `request_timeout: 1800`. Step caps, `llm_heal`, task, prompts and modal block come from the hero9 source. |
| `configs/<task>__qwen3.6-27b-bf16-think.yaml` (28, new) | The classic configs with thinking ON: `chat_template_kwargs {enable_thinking: true, preserve_thinking: true}` and thinking-mode sampling 1.0 / 0.95 / 20 / 0 (model card). **The host is a placeholder (`SET-BOX-IP:1`)**: regenerate with `--host/--port` once a vLLM box exists. |
| `scripts/modal_full.sh` | `TIMEOUT_S` overrides the per-run wall clock. Unset, it behaves as before. |
| `scripts/classic_counts.py` | `SET_ASIDE_FILE` and `COUNTS_TAG` environment overrides. Unset, it behaves as before. |
| `scripts/run_usage.py` (new) | Cost per run, plus tokens, reasoning, cache, longest prompt (run length) and model-time hours, read from the `[provider-usage]` lines with a fallback to `_usage`. Prints a per-task table, writes an optional CSV, and gives a `--project N` projection. |
| `scripts/phase2_provider_check.py` (new) | **Stage 1.** The free part runs by default. `--paid` makes the calls: per endpoint, one tool-using turn (tool call parsed? reasoning length and `reasoning_details`? cost? host?), then turn 2 twice, with and without the earlier reasoning, to see whether the host puts it back into the prompt. It also compares the reasoning tokens at `high` and `xhigh`. |
| `scripts/phase2_money.py` (new) | Free key and account balance check; `--need`, `--save-start`, `--start` (spent since). Exits 2 when money is short, 3 when there's no key. |
| `scripts/phase2_smoke_check.py` (new) | The smoke gate. A run fails if it didn't finish, if any turn lacks reasoning, if any call lacks a cost, or if any call was served by a host other than the pin. Also counts unknown-function replies, empty replies and 429s. |
| `scripts/phase2_lib.sh` (new) | Shared helpers: in-flight counts, the launch-log ledger, 429 counting, spend (the larger of the key delta and the ledger), a spend guard that stops every app, a 5-minute heartbeat wait, lean fetch, and HF upload (gzipping NUL-byte logs). |
| `scripts/phase2_pilot.sh` (new) | **Stage 2.** Preflight, then a smoke of 7 loop types plus the gate, then 2 runs × 28 tasks, then fetch, cost projection to 46 runs per task, counts, upload, and "PILOT DONE". |
| `scripts/phase2_fleet.sh` (new) | **Stage 3.** A paced queue: 209 chunks covering 1,198 runs (13 tasks × 46 + 15 × 40, checked with a dry run). It holds launches on 429 health, uses an adaptive cap, applies the spend cap (4 × the estimate), sets aside cut chunks and re-runs them once, runs up to 2 top-up rounds, then builds the per-run table (`HERO_ALL_ENVS=1`), the cost summary and the uploads. |
| `scripts/setup_launch_box.sh` (new) | Run from the laptop. It rsyncs the code and harness (never `.env` or `results/raw`), installs uv, hf with hf_xet and PyYAML, and checks names only: modal.toml, profile, secrets, and the keys in `/workspace/.env`. GG copies the credentials himself. |

## Checks run, all free

- **Configs.** `make_target_configs.py deepseek-v4-pro --check`: all 28 show "other keys differ: none" and the same `max_steps` as the classic configs. The same holds for `qwen36-think`.
- **Loading.** All 56 new configs load with the harness's own `run_common.load_config` (resolvers included). `create_provider` builds a provider from each, no agent keys are dropped, and `modal_run_settings` gives DeepSeek egress `['openrouter.ai']` only.
- **Ledger test with fake replies (no network).** The cost line prints, `_usage` is stored and not sent, reasoning and `reasoning_details` are sent back, and the vLLM path carries no usage field. Harness tests: `tests/src` 90 passed, 34 skipped.
- **Pin.** Matched against the free endpoint list: `siliconflow/fp8`, tools and reasoning listed, $0.90 in / $1.70 out per million. StreamLake, DeepInfra, GMICloud and Novita fp8 were also found.
- **Scripts.** `bash -n` passes for every shell script, `py_compile` for every Python one. `run_usage.py` was tested on fake runs, and `phase2_money.py` against the real key (read-only).
- **Money (free, read 19 Nov):** key usage $5,966.71 of a $10,000 limit, so **$4,033 left**. Account credits $87,973.93, used $84,301.27, so **$3,672 left**.

## Exact commands

**Stage 1: provider check, on the laptop.** About $0.05 likely; worst case about $0.30 (6,000-token cap per reply).
```
cd gigi/3-forensics/agent-interp-envs && .venv/bin/python ../scripts/phase2_provider_check.py --paid
```
Output: `results/logs/phase2/provider_check__deepseek_deepseek-v4-pro.json`. If another host wins, regenerate the configs:
`agent-interp-envs/.venv/bin/python scripts/make_target_configs.py deepseek-v4-pro --pin <slug> --check`.

**Stage 2: smoke and pilot.** About $50–100 of API (7 smoke runs + 56 pilot runs), plus about $10–20 of Modal; spend cap $300.
1. GG rents a small always-on box, then:
   ```
   bash gigi/3-forensics/scripts/setup_launch_box.sh <host> <port>
   ```
   GG then copies `~/.modal.toml` and a `.env` with HF_TOKEN (and OPENROUTER_API_KEY, for money checks on the box) using the two scp lines printed in the script header.
2. Launch on the box, detached:
   ```
   ssh -i ~/.ssh/vastai_lies_experiment -p <port> root@<host> 'TAG=deepseek-v4-pro PIN=siliconflow PILOT_CAP=300 setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/phase2_pilot.sh > /workspace/logs/phase2_pilot.out 2>&1 < /dev/null &'
   ```
3. Watch `/workspace/logs/phase2.txt` with a polling heartbeat. The last line, "PILOT DONE", gives the spend and the projection for 46 runs per task.

**Stage 3: the fleet, after GG approves the projected number:**
```
TAG=deepseek-v4-pro PIN=siliconflow ESTIMATE=<projection> SPEND_CAP=<about 4x> TIMEOUT_LONG=<from the pilot> setsid nohup bash .../phase2_fleet.sh > /workspace/logs/phase2_fleet.out 2>&1 < /dev/null &
```

## Open risks

1. **Money.** The account has $3,672 left and the key $4,033. That covers the likely DeepSeek fleet ($1.3–1.9k) but not the high case ($4–6k), and not five teachers. John needs to top up before stage 3 and check the workspace lifetime budget, which a key can't read.
2. **Reasoning level.** OpenRouter's top level is `xhigh`. Whether it maps to DeepSeek's Think Max is unverified; the paid check compares `high` and `xhigh`. The paper harness is itself inconsistent: CLAUDE.md says "max" for deepseek and `dashboard_perf` sets max, but `puppeteer` and `precommit_hook` set nothing, so Fireworks would default to "low".
3. **Earlier reasoning sent back.** The history carries `reasoning`, `reasoning_content` and `reasoning_details`. Whether SiliconFlow accepts and renders them (as the paper's Fireworks template does) is unknown; it might even reject `reasoning_content` with a 400. The paid check's turn 2 measures exactly this.
4. **Run length and wall clock.** The pilot gives runs 12 h (caps over 60 steps) or 3 h; Modal's maximum is 24 h. The 500-step tasks at Max effort might go past that; `run_usage.py` reports the model-time hours.
5. **Reasoning-only replies.** These count as "empty", are retried 3 times, then crash. The smoke check counts them; if they're common, the harness needs a re-prompt rule ("Respond directly in text", per the harness's CLAUDE.md).
6. **`_usage` in messages.json.** Any downstream reader that renders messages (the student-side renderer in `distill_data`, the swap scripts) must ignore it. The provider strips it on send; the student-side fork should know.
7. **Parallel calls.** No host lists `parallel_tool_calls`, so `require_parameters` is off. A host may ignore `false`, and the environment then answers "You called 2 functions", as with hero9 Qwen.
8. **Qwen thinking-on configs** carry a placeholder host. Regenerate with the real box. Their runs will be longer, so size the vLLM window and the number of runs at once to the KV cache (memory vllm-agent-fleet-kv-sizing).
9. **Modal app count.** The fleet launches 209 chunk apps; modal_full.sh retries rate-limited creations. Modal cost is about $200 per teacher on top.
10. **Spend tracking on the box** uses the key's usage delta only if `/workspace/.env` holds OPENROUTER_API_KEY. Otherwise it uses the launch-log ledger, which needs the run output streamed into the launch logs; it is, as in hero9.
11. **GPT-OSS, Kimi K3 and GLM 5.3** are unchecked entries in `make_target_configs.py`: sampling, reasoning level and pin need confirming first.
