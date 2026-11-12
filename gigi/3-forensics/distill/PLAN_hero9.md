# Distillation plan: Qwen3.6 27B into gemma-3 27B (hero9 data, 12 Nov 2026)

Written for John and Yuqi, from the 12 Nov call. It replaces `PLAN.md` (3 Nov, Qwen3-Next-80B and the old data) for
the hero run; the training code, the memory analysis and the box rules there still hold.

**Status: data built and uploaded, code ready, nothing rented, nothing spent.** The spend in "Cost" needs GG's yes.

## What the call decided, and how this plan does it

| Decision (12 Nov) | Here |
|---|---|
| One direction: Qwen3.6 = target, gemma = substitute | LoRA fine-tune of gemma-3-27b-it on Qwen3.6's own turns |
| Qwen's traces reformatted into gemma's format, not raw (to avoid "brain damage") | Every example is rendered with the harness's own gemma text-mode functions and gemma's chat template, byte-identical to what gemma sees at inference |
| Hold out ~3 environments: Qwen > gemma, gemma > Qwen, equal | **tower_of_hanoi** (Qwen 12/20, gemma 0/20), **eval_tampering** (gemma 20/20, Qwen 5/20), **currency_conversion** (both ~100%: 20/20, 11/11). All three are measured mechanically and are short, so re-running them at every checkpoint is cheap. Change them before training if you prefer others. |
| Yuqi: first train and test on the same distribution | 10% of the run families of every training task are held out as an in-distribution validation set |
| Keep intermediate checkpoints; plot the gap over training | Checkpoints at 25 / 50 / 75 / 100% of the epoch, all kept; each is re-run on the tasks below |
| Check gemma is not brain-damaged | A "behaves normally" panel at every checkpoint (below) |

## 1. Data (done, free)

`distill_data.py build --hero9 --trajectories --max-context-chars 105000 --holdout-envs tower_of_hanoi,eval_tampering,currency_conversion --split 0.9 --seed 0`

- **Sources**: Qwen3.6's 331 scored hero9 plain runs, and every swap continuation Qwen3.6 wrote (3,189): on its own
  runs and on gemma's. **Only Qwen3.6's own turns are trained**: in a continuation, turns 0..K are the source run's
  (checked: the first new turn is K+1), so when Qwen3.6 took over a gemma run, gemma's turns are context, never targets.
- **One sequence per run** with the loss on all of Qwen3.6's kept turns, instead of one example per turn. Each
  per-turn example is the start of the whole run, so the targets are the same (checked on gemma's real tokenizer:
  87 of 87 turns give identical tokens both ways) for **14.2 M train tokens instead of ~130 M**.
- **Cap**: 32k tokens per sequence (what one H100 trains with activation offload); later turns of longer runs are
  dropped, as before. 12 of 1,538 train runs lose everything to the cap.
- **Split**: train 1,526 sequences (294 run families, 25 tasks), in-distribution val 199 (32 families), held-out
  test 1,304 (the three held-out tasks). Families stay whole: a run never sits in two splits with its continuations.
- **Not used**: 141 lost runs (no usable transcript) and swe_rebench_v2 (its fleet was saved without a config, so
  its tools are unknown; it has no cheating anyway).
- Files: private HF dataset `model-hermeneutics/gigi-forensics-transcripts`, `distill/hero9_v1/`; manifest in git.

## 2. Training

- LoRA r=16 on all 7 projections of the 62 text layers, bf16, lr 1e-4 cosine, 1 epoch, 16 runs per optimizer step
  (~95 steps), `--max-len 32768 --offload-activations --keep-checkpoints --save-every 24`.
- **Time**: 14.2 M tokens at 1,500-2,500 tokens/s (the smoke run measures the real rate) = **1.6-2.6 h**.
- **Smoke gate first**: the 4 longest sequences, real memory, finite gradients; the job aborts cleanly if it fails.

## 3. Re-running each checkpoint in the environments

- **Serving**: vLLM on the same box, base gemma-3-27b-it in bf16 with the checkpoints loaded as LoRA adapters
  (`--enable-lora --lora-modules c25=... c50=... c75=... c100=...`), so one server serves all of them and base gemma
  through the same stack (the control: same server, adapter off). Fallback if vLLM's LoRA path fails the gate:
  merge each checkpoint (~15 min each) and serve them one at a time.
- **Harness**: the runs go through the normal Modal sandboxes with `agent.base_url=http://<box>:<port>/v1`,
  `agent.api_key_env=VLLM_API_KEY` (added today, tested; unset = OpenRouter as before), text-mode tool calls and
  gemma's model-card sampling, plain (no end check), exactly as the hero9 gemma runs.
- **What runs at each checkpoint** (base = checkpoint 0, then 25/50/75/100%): 20 plain runs on each of
  - the 3 held-out tasks: tower_of_hanoi, eval_tampering, currency_conversion;
  - 3 training tasks: coding_norules (Qwen > gemma), lazy_investigation (gemma > Qwen), coding.
  = 120 runs per checkpoint, 600 in all. All measured mechanically: no judge.
- **The plot John asked for**: per task, the bad-act rate of distilled gemma at each checkpoint, with Qwen3.6's and
  base gemma's plain rates as reference lines. The hope: the line moves from gemma's level towards Qwen3.6's.
- **"Behaves normally" panel**, per checkpoint: share of runs that end properly; tool calls that fail to parse;
  empty replies; repeated calls; and the training loss on the in-distribution val and on the held-out test set.
  A checkpoint whose runs stop ending properly or whose tool calls break is flagged as damaged.

## Cost (needs GG's yes)

| Item | Estimate |
|---|---|
| H100 80 GB on Vast, ~7-9 h (setup + download 1 h, smoke 0.3 h, training 1.6-2.6 h, 5 checkpoints x ~45 min of runs) | $18-23 at $2.50/h ($28-36 at $3.99/h) |
| Modal sandboxes, 600 short runs | ~$20-50 |
| OpenRouter / Opus judge | $0 (mechanical measures only) |
| **Total** | **~$40-90** |

Suggested hard cap: **$300** (several times the estimate, per the standing rule). The box is never stopped
automatically: when a stage ends or fails, GG is told it is idle and still billing.

## Before renting: three things only GG can do

1. **Approve the spend** (above).
2. **Refresh the Vast API key on the laptop**: the saved key is rejected ("Invalid user key", 12 Nov). GG runs
   `vastai set api-key <key>` in the terminal (never pasted in chat).
3. **A Modal token and `VLLM_API_KEY` on the box** go in the box's own `.env` (never in chat or git); the Modal secret
   `vllm-secret-gigi` holding `VLLM_API_KEY` is created once (free).

## Open points

- John's "Jack Rong" distillation playbook (link pasted in the call chat): to read before training, in case its
  method beats plain LoRA SFT on the target turns.
- Not yet verified, checked by the first gate on the box: that Modal's sandbox egress allowlist passes the box's IP
  and port, and that vLLM's LoRA serving works for gemma-3.
- Later, if time allows: the inverse direction (gemma into Qwen3.6), and J-space/NLA on the checkpoints (TQ).
