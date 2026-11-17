# Classic distillation of Qwen3.6 27B into gemma-3 27B: plan (17 Nov 2026, v2 after three parallel reviews)

What the 16 Nov call decided (John's notes, "SPAR Project" doc, week 10):
1. First augment the dataset: 40 Qwen runs per task instead of about 10.
2. Then rerun the distillation the classic way: no swap data, 80/20 split, held-out tasks with at least one each of
   gemma == Qwen, gemma > Qwen, gemma < Qwen.
TQ needs the classic checkpoints to fit fresh J-lenses.

Everything that differs from round 1 (`PLAN_hero9.md`) or from the hero9 Qwen runs is marked **CHANGE**. v2 folds in
three independent reviews (data and method; server and harness; evaluation, statistics and time). This file is
committed before any run starts, so the analysis below is pre-registered.

## 0. Totals at a glance

| | Count |
|---|--:|
| Tasks in the suite | 29 (swe_rebench_v2 not used: its fleet has no saved config, as in round 1) |
| Tasks used | 28 |
| Held-out tasks (never trained on) | 5: tower_of_hanoi, lazy_investigation, eval_tampering, currency_conversion, norvane |
| Training tasks | 23 |
| New Qwen runs | 40 per task x 28 tasks = 1,120 kept (plus extra launches, see 1) |
| Training runs | 23 x 40 = 920: 80% train (~736), 20% check (~184) |
| Re-run tasks after training | 7 (the 5 held out + coding_norules + coding) |
| Re-run models | 5: gemma before training (fresh, same server) + 4 checkpoints |
| Re-runs | 180 per model x 5 = 900 (see 5) |

## 1. New Qwen runs (the teacher data)

- **CHANGE: "augment" is done as 40 new runs per task, not 30 added to the old 10**, so the training data has one
  precision. Served on our own box in **bf16** (standing decision of 10 Nov: generation moves to our own vLLM after
  hero9), not DeepInfra fp8. State "bf16, own vLLM" next to every number from these runs.
- Box: Vast instance 61015415, one H200 NVL (141 GB), 600 GB disk, $3.94/h. It later trains gemma and serves the
  checkpoints.
- Server (`/workspace/serve_qwen.sh` on the box): vLLM **0.32.0, its CUDA 12.9 build** (the box's driver is CUDA 12.8;
  the default wheels need 13; torchcodec removed, it needs CUDA 13 and is only for video), Qwen/Qwen3.6-27B at revision
  `8f33b98937c6031bb49e713dee3e8e185aafd26f`, served as `qwen/qwen3.6-27b` (keeps hero9's run-folder slug),
  `--language-model-only --dtype bfloat16 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser
  qwen3_coder --reasoning-parser qwen3`, thinking off on the server (`--default-chat-template-kwargs`) and in every
  request (`agent.chat_template_kwargs`). Pip freezes of every environment in `/workspace/logs/freeze_*.txt`.
- **CHANGE, harness (patch `patches/classic_provider_knobs.patch`; defaults unchanged for every other run):**
  `agent.parallel_tool_calls: null` (vLLM obeys `false` and drops a second call; DeepInfra ignored it, so hero9's Qwen
  made two calls in one turn in 16 of 18 currency runs), `agent.request_timeout: 3600` (our server sends nothing until a
  reply is complete; the 600 s default resampled long replies), `agent.chat_template_kwargs`.
- Run tag `qwen3.6-27b-bf16` (configs `configs/<task>__qwen3.6-27b-bf16.yaml`, per-run tables, precision label
  "bf16 (own vLLM)"). Sandboxes keep hero9's 2 GB where the hero9 config had no modal block.
- **CHANGE: 49,152-token context window, on purpose.** The training data keeps only the first 105,000 characters of
  each run (and at most 32k gemma tokens). Stopping each run once its history is past that point leaves the training
  data as it would be from full runs and cuts generation by more than half (72% of Qwen's output in the long tasks
  comes after 105,000 characters). First tried at 32,768: the ai_rd smoke stopped at 102,502 characters, just under
  the cut, so the window was raised to 49,152 to be safely past it; the data build checks every stopped run reached
  the cut. Every task we report rates for is far below it (coding ~3k tokens, tower ~11k). Runs that reach it end with
  "context window full" (the steps and rollout.log are saved on the volume; checked on the ai_rd smoke) and are kept.
- Same as hero9, unchanged: every prompt, step cap, task variation and sampling setting from
  `configs/<task>__qwen3.6-27b.yaml` (temperature 0.7, top-p 0.8, top-k 20, min-p 0; presence_penalty 1.5 not sent, as
  in hero9). Native tool calls. New config files `configs/<task>__classic_qwen36.yaml`, written by script from the hero9
  ones; only the agent endpoint changes. The hero9 configs stay untouched.
- Gates before the fleet, all must pass:
  1. Garble gate: the same request twice gives two sensible replies; a 3-step conversation stays sensible.
  2. No thinking text in any reply; tool calls come back parsed.
  3. Prefix caching on, and a cache hit rate of at least 90% in the pilot (below), read from vLLM's metrics.
  4. One smoke run per loop type through Modal: coding (submit ending), tictactoe (game ending), eval_tampering
     (plain-reply ending). Read each rollout.log; behaviour in line with hero9's Qwen runs.
  5. Pilot: 30 minutes of the long tasks at full concurrency; re-estimate time and cost from the measured throughput
     before the full fleet.
- Gate results (17 Nov): `results/logs/classic/gate.txt`. Reply checks 12 of 12 clean; one-at-a-time and cache-hit
  replies identical in 12 of 12; batched identical in 9 of 12, under the pre-set 10: **accepted with a note** (the
  three differences are sensible replies that branch after the first sentence, batch-order nondeterminism, not
  garbling). Smokes (stage `classic_smoke`): coding, tictactoe, eval_tampering, currency_conversion end as their loop
  types should, ai_rd stops at the window; currency shows hero9's two-calls-in-one-turn; no thinking anywhere.
- Fleet launched **from the box** (`scripts/classic_fleet.sh`), paced on live containers. Long training tasks get about 15% extra launches up
  front instead of serial top-ups. The **first 40 runs by launch order** that finish (or reach the context window)
  are kept, not the fastest 40. Lost runs counted per task.
- A run counts as complete only if its rollout.log shows FINAL RESULTS or "context window full" (`run_end`, new in
  `distill_data.py`), never from the table's status alone (it missed vLLM timeouts in round 1).
- Scoring: mechanical measures only, per-run table from `full_runs_table.py` without Opus. **No Opus grading.**

## 2. Held-out tasks, fixed now

| Case | Task | gemma before training (our server, round 1) | gemma (hero9, OpenRouter) | Qwen (hero9, fp8) |
|---|---|--:|--:|--:|
| Qwen cheats more | tower_of_hanoi | 0 of 20 | 0 of 20 | 12 of 20 |
| gemma cheats more | lazy_investigation | 20 of 20 | 20 of 20 | 0 of 14 |
| Equal, at the top | currency_conversion | 19 of 20 | 20 of 20 | 11 of 11 |
| Equal, at the bottom | norvane | (not run) | 0 of 20 | 0 of 20 |
| Reading the note | eval_tampering | read it 0 of 20 | read it 1 of 20 | read it 20 of 20 |

- **CHANGE: lazy_investigation moves from training to held out.** After the 15 Nov fix gemma never reads the eval
  tampering note, so eval tampering is no longer the "gemma cheats more" case; lazy_investigation is the only clear one
  left. Its bad act is a wrong answer, so "opened a source file" and step counts are shown next to it.
- **CHANGE: norvane added** as a second equal case, at the floor: it catches cheating that training creates where the
  teacher has none. Currency, at the ceiling, can only fall.
- eval_tampering stays held out (round 1's note-reading result; TQ's work).
- Leakage check (review 1): no training task shares an environment, codebase, file, tool or answer with these five.
- Rule, applied after the 40 new Qwen runs and before the training data is built: each case must still hold against
  both gemma columns (a "more" case: Wilson 95% intervals do not overlap; an "equal" case: they overlap). If any case
  fails, stop and ask GG. No automatic alternates (coding_norules would leak through coding: same problem).

## 3. Training data

- **CHANGE (the point of this round): Qwen's own complete runs only.** `distill_data.py build --hero9 --full-csv
  <new per-run tables> --no-swap --end-check --keep-context-cut --expect-model qwen3.6-27b --trajectories
  --max-context-chars 105000 --holdout-envs tower_of_hanoi,lazy_investigation,currency_conversion,eval_tampering,norvane
  --split 0.8 --seed 0 --max-mb 1000 --out distill/classic_v1` (new flags, 17 Nov). Checks after building: every
  example is `kind == full`, every run path is in the new stage, no message from gemma, no swap file read.
- Same as round 1: Qwen's turns in gemma's text-mode tool format, one sequence per run with the loss on every kept Qwen
  turn, 105,000-character cap, 32k-token cap.
- **CHANGE: 80/20 split by run** inside each training task (round 1: 90/10). The 20% is used for the loss during
  training and for the per-checkpoint check in 4.
- Looping runs are kept: in a classic distillation the teacher's own behaviour is the data (filtering would be a
  different method); Qwen's own runs barely loop (3 of 250 in round 1). Counted and reported.
- Said plainly next to every in-distribution result: in the long tasks the data holds only the first ~28% of Qwen's
  output, often before the bad act; the share of turns kept per task is reported.

## 4. Training

- Same LoRA as round 1: rank 16, alpha 32, dropout 0.05, lr 1e-4, 16 examples per step, warmup 3%, bf16,
  `--max-len 32768 --offload-activations`, seed 0. Smoke gate first.
- **CHANGE: 2 epochs, checkpoints at 0.5, 1.0, 1.5 and 2.0 epochs** (11 Nov notes). One schedule over 2 epochs, so the
  1.0-epoch checkpoint is "halfway through a 2-epoch run", not a separately trained 1-epoch model.
- At each checkpoint: teacher-forced loss on the 20% check set and on the held-out tasks' Qwen runs, against gemma
  before training, split into Qwen's cheating and clean runs where the task is scored mechanically.
- Checkpoints uploaded to the private HF dataset as soon as each exists (`distill/classic_v1/`), for TQ.

## 5. Re-runs after training

- Served exactly as round 1's v2 re-runs, pinned in its own environment: vLLM 0.11.0, base gemma bf16 with the
  checkpoints as LoRA adapters, `--max-lora-rank 16 --max-model-len 65536 --gpu-memory-utilization 0.92
  --no-enable-prefix-caching --enforce-eager`, gemma's own sampling and text-mode tools, the same configs.
- **CHANGE: gemma before training is re-run fresh on the same server, at the same time as the checkpoints** (round 1
  ran them side by side; reusing four-day-old runs from another box would break that). Round 1's base runs are compared
  with the fresh ones as a check for drift.
- Runs launched per model: tower 20, lazy 20, norvane 20, coding 20, coding_norules 30, currency 30, eval_tampering 40
  = 180; x 5 models = 900. Top-up rule: until each held-out cell has at least 20 scored runs for its primary measure,
  at most twice the launched count.
- Garble gate (0 garbled allowed), launched from the box, paced. Scored with `distill_eval.py` (lazy_investigation and
  norvane added to the held-out list; the base comes from this round, not round 1). Loss audit per cell before any
  result is called done.

## 6. Pre-registered analysis

- **Primary questions**, one per held-out task with a gap, direction stated:
  1. tower_of_hanoi: the bad act (a solver script) goes up.
  2. lazy_investigation: the wrong answer (1,200) goes down.
  3. eval_tampering: reading the note goes up.
- **Primary contrast:** the three checkpoints from 1 epoch on (1.0, 1.5, 2.0 pooled) against gemma before training,
  same server. Exact Fisher test, two-sided, Holm correction across the three.
- **Controls:** currency_conversion and norvane should not change (Wilson intervals, no test); coding_norules is the
  in-distribution positive control: if it does not move towards Qwen, a null on the held-out tasks cannot be read.
- **Secondary, descriptive:** the trend over 0 / 0.5 / 1 / 1.5 / 2 epochs (Cochran-Armitage); the share of the gap to
  Qwen closed, (checkpoint - base) / (Qwen - base), with a bootstrap interval, only where a clear gap exists;
  eval_tampering's "keeps quiet once the note is read", pooled over checkpoints, against Qwen (gemma has no starting
  point there: it never reads the note).
- **Every cell reports** launched / finished / no answer / scored, the bad-act rate among answered runs and among all
  finished runs, and the health measures (out of steps, repeated calls, tool misuse). "No answer" is a behaviour, kept
  as its own category.
- **Caveat stated with the results:** compared with round 1, the teacher data (bf16, new runs, 40 per task), the
  held-out set, the epochs and the split all change along with dropping the swap data, so a classic-versus-swap
  difference is confounded.

## 7. Time and cost (measured rates)

| Stage | Time | Cost |
|---|--:|--:|
| Box setup, downloads, gates, smokes, pilot | ~2.5 h | box |
| 1,120+ Qwen runs, stopped at 32k tokens | ~4-6 h (re-estimated after the pilot) | box + Modal |
| Data build and checks | minutes | free |
| Training: ~12.8M tokens per epoch x 2 at the measured ~600 tokens/s, plus checkpoint checks | ~12-14 h | box |
| 900 re-runs, scoring, uploads | ~6-7 h | box + Modal |
| **Total** | **~25-30 h wall clock** | |

- Box: ~30 h x $3.94 = ~$120. Modal: ~$150-300. OpenRouter: only ai_rd's capped agent key inside its runs (key headroom
  $4,048, org credit $4,060 on 17 Nov). No Opus grading. **Estimate ~$300-450**, under the ~$530 GG approved on 17 Nov;
  if the pilot puts it above $530, stop and ask. Watchdog hard cap $1,200.

## 8. What GG has to do

- Copy the Modal login to the box (I may not move Modal credentials):
  `scp -i ~/.ssh/vastai_lies_experiment -P 11630 ~/.modal.toml root@ssh5.vast.ai:/root/.modal.toml`
