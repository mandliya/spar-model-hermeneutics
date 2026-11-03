# Distillation plan: Qwen3-Next-80B traces into gemma-3-27b-it, then the swap again (3 Nov 2026)

This is the first in-distribution distillation agreed on the 3 Nov call. We fine-tune gemma-3-27b-it with LoRA on
Qwen's own agentic turns in our environments. Then we re-run the swap experiments with the distilled gemma in
gemma's seat, through the same harness, and ask whether it now behaves more like Qwen. Tinker does not serve gemma 3,
so training runs on a rented H100.

**Status: proposed. Nothing in this plan has run, and nothing has been spent.** Every step that costs money needs the
user's explicit yes for that step (CLAUDE.md, "Money: ask first").

## Decisions for the team before any spend

1. **Split.** `--stratify-incidents` (Yuqi's rule) holds out a cheating family in 5 of the 8 gap environments
   (README, "Incident-stratified split"). coding, precommit_hook and optimize_rendering_engine_graded cannot hold one
   out. In dashboard and grade_your_own, seed 0 puts the big swap family in val. That makes the split 58/42 by
   example, and leaves dashboard's training set with almost no cheating turns. Choose a seed, or a tie-break such as
   "hold out the smaller cheating family", before building, and never after seeing results.
2. **Context cap.** One H100 trains up to 32k tokens per example with activation offload, and 16k without. The cap
   decides how many incident turns survive. At 16k, 0 of optimize_rendering_engine's 49 held-out hacked-run examples
   remain, and 26 of dashboard's 607. At 32k the figures are 49 of 49 and 47 of 607. **Recommendation: 32k.**
3. **Base-gemma control arm.** The existing gemma swap arm ran on DeepInfra (fp8, through OpenRouter). The distilled
   gemma will run on our own vLLM server, which changes the serving stack too. Running base gemma through the same
   server separates "distillation changed it" from "the server changed it". **Recommendation: run this control.**
4. **Scope of the swap re-run.** The gemma arm has 1,633 continuations in all, 853 of them in the 8 gap environments.
   **Recommendation: run the gap environments first.** Their measures are mechanical, so they need no judge spend.

## Steps

### 1. Build the data (laptop, free, about 3 minutes)

```
PY=gigi/3-forensics/agent-interp-envs/.venv/bin/python
$PY gigi/3-forensics/scripts/distill_data.py build --stratify-incidents --dry-run --max-mb 100000   # check the table
$PY gigi/3-forensics/scripts/distill_data.py build --stratify-incidents --max-mb 100000 --max-context-chars 105000 --out distill/stratified
$PY gigi/3-forensics/scripts/distill_data.py check --out distill/stratified
```

- `--max-context-chars 105000` is about 32k gemma tokens. The measured rate is 3.33 characters per token, from 400
  examples run through gemma's own tokenizer. The cap keeps 13,343 of 29,807 examples, about 0.6 GB of JSONL.
  `distill_train.py` then drops anything still over `--max-len` tokens, exactly.
- The split is drawn before the cap, so the cap never moves a family between train and val.
- `distill/.gitignore` already ignores `*.jsonl`, so the data goes to HF instead of git:
  `hf upload model-hermeneutics/gigi-lie-detection-activations gigi/3-forensics/distill/stratified distill/stratified-v1 --repo-type dataset`.
  The manifest is committed.

### 2. Rent the box (Vast.ai)

- **Box**: one H100 80 GB (SXM preferred), at least 64 GB of host RAM (the merge loads the model on CPU) and at least
  250 GB of disk. The disk holds the base model (55 GB), the merged model (55 GB), checkpoints and data. Map port 8000
  for vLLM in step 4. Never rent GB10 or DGX Spark (CLAUDE.md).
- **Price assumption: $2.50 per H100-hour, with a range of $2.00-3.99.** Vast is a marketplace with no list price. A
  guide gives $2.00-3.50 per hour as typical for H100 SXM on Vast
  ([deploybase.ai](https://deploybase.ai/articles/vast-ai-gpu-cloud-pricing-complete-guide-vs-hr-for-every-gpu)). A
  tracker shows the cheapest on-demand H100 SXM on Vast at $1.93 per hour in June 2026
  ([gpu-cost.pages.dev](https://gpu-cost.pages.dev/gpu/h100-sxm/)). RunPod lists the H100 SXM at $3.99 per hour on
  Secure Cloud and $2.69 on Community ([runpod.io/pricing](https://www.runpod.io/pricing), read 3 Nov 2026). Modal
  charges $3.95 per hour for an H100 ([modal.com/pricing](https://modal.com/pricing)). All of these are ceilings to
  check against the live Vast offers on the day of renting.

### 3. Train (`scripts/distill_train.py`)

The script does LoRA SFT on the target turn only: bf16, gradient checkpointing, r=16 on all 7 projections of the 62
text layers (113 M parameters). Its docstring gives the memory reasoning: about 75 GiB at 16k without offload, and
about 71 GiB at 32k with `--offload-activations`. It writes `adapter/`, `log.json` (every train and eval loss line,
tokens per second and peak memory) and `val_per_example.jsonl`. That last file holds the target NLL of every val
example, scored twice: once with the trained adapter and once with the adapter off (base gemma).

```
python distill_train.py train --data-dir $DATA --out $OUT --max-len 32768 --offload-activations --smoke   # gate
python distill_train.py train --data-dir $DATA --out $OUT --max-len 32768 --offload-activations \
    --eval-every 100 --final-val-sample 1000 --resume
```

**Throughput assumption.** Each token costs about 6N FLOPs: 2N for the forward pass, 2N for the checkpoint
recompute and 2N for activation gradients (the LoRA weight gradients are negligible). With N = 27 B, that is
1.6e11 FLOPs per token. An H100 SXM peaks at 989 dense bf16 TFLOP/s. At 25-40% utilisation, that gives 1,500-2,500
tokens per second. **The smoke run measures the real rate (`tokens_per_s` in log.json)**, so re-estimate before
approving the main run.

| | 16k, no offload | 32k, offload (recommended) |
|---|---|---|
| train examples / tokens per epoch | 5,664 / 37 M | 9,349 / 128 M |
| val examples | 1,872 | 3,618 |
| train time, 1 epoch | 4.1-6.9 h | 14-24 h |
| periodic evals (150 examples) | ~0.3 h | ~1 h |
| final per-example val, distilled + base | ~1.3 h (all) | ~1.4 h (1,000, all hacked-run examples kept) |
| setup, download, smoke, merge, uploads | ~1 h | ~1 h |
| **box hours** | **~7-10 h** | **~17-28 h** |
| **cost at $2.50/h (at $3.99/h)** | **$17-25 ($27-40)** | **$43-70 ($68-112)** |

The team's 4-8 H100-hour quote matches the 16k configuration. The 32k configuration costs about three times as
much, but it is the one that keeps the incident turns.

The cheaper lever, not implemented, is to train each trajectory once with loss on every assistant turn, instead of
one example per turn. The per-turn format re-reads each history once for every turn it contains. The same coverage
would then cost a fraction of the tokens. It needs a format change and a decision on how to weight replays' shared
prefixes, so it is left out of this first run.

### 4. Merge, then serve behind an OpenAI-compatible endpoint (vLLM, on the same box)

- **Merge**: `python distill_train.py merge --adapter $OUT/adapter --out $OUT/merged`. This takes about 15 minutes
  and needs about 60 GB of host RAM. A merged checkpoint serves like the original model, and it can use vLLM's
  online fp8 quantisation, which LoRA-on-fp8 serving may not support. fp8 also matches the fp8 that DeepInfra served
  the existing gemma arm with. It halves the weights (28 GB), which leaves about 45 GB of KV cache for 128k-token
  histories.
- **Serve one model at a time** (pin `vllm==0.11.0`):
  ```
  vllm serve $OUT/merged --served-model-name gemma-distill-v1 --quantization fp8 --max-model-len 131072 \
      --gpu-memory-utilization 0.92 --enable-prefix-caching --limit-mm-per-prompt '{"image":0}' \
      --api-key "$VLLM_API_KEY" --port 8000
  ```
  The control arm serves `google/gemma-3-27b-it` with the same flags and `--served-model-name gemma-vllm-base`.
- **Gate**: before any Modal launch, render one val example's prompt, send it to the server and parse the reply with
  `_tool_call_from_tool_code`.

### 5. Point agent-interp-envs at the endpoint (the exact change)

Providers are created in `agent-interp-envs/src/agent_interp_envs/providers/__init__.py` (`_create_provider`).
gemma runs through `OpenRouterProvider` with `text_tools: true`, which renders the tools into the prompt and parses
`tool_code` blocks. It is the only provider that supports text mode (`text_tools` raises for every other provider).
It hardcodes `base_url="https://openrouter.ai/api/v1"` and `OPENROUTER_API_KEY`. vLLM speaks the same chat
completions API, so the change is two optional knobs:

```diff
# providers/__init__.py: _create_provider(...)
+    base_url: str | None = None,
+    api_key_env: str | None = None,
 ...
     if provider == "openrouter":
         return _finish(provider_class(
             ..., text_tools=text_tools,
+            base_url=base_url, api_key_env=api_key_env,
         ))
+    if base_url is not None or api_key_env is not None:
+        raise ValueError("base_url/api_key_env are only supported for openrouter")

# providers/openrouter_provider.py: OpenRouterProvider.__init__(...)
+        base_url: str | None = None,
+        api_key_env: str | None = None,
 ...
-        api_key = require_api_key("OPENROUTER_API_KEY", "openrouter")
+        api_key = require_api_key(api_key_env or "OPENROUTER_API_KEY", "openrouter")
         self.client = OpenAI(
-            base_url="https://openrouter.ai/api/v1",
+            base_url=base_url or "https://openrouter.ai/api/v1",
```

- **No other plumbing is needed.** `checkpoint.provider_kwargs` filters the agent config by `create_provider`'s
  signature (`_PROVIDER_PARAMS`), so `agent.base_url` and `agent.api_key_env` reach the provider on resume.
- **The key is scrubbed after use.** `VLLM_API_KEY` matches the `*API_KEY*` pattern, so `scrub_secrets_from_environ`
  removes it right after construction. Commit the patch in agent-interp-envs, then launch with `--use-current`, as
  `swap_all.sh` already does.
- **Overrides for the distilled arm** (the same sampling as the existing gemma arm, model-card settings):
  ```
  agent.model=gemma-distill-v1,agent.base_url=http://<BOX_IP>:<PORT>/v1,agent.api_key_env=VLLM_API_KEY,agent.provider_preferences=null,agent.reasoning_effort=none,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true,modal.allow_hosts=[<BOX_IP>]
  ```
  - `modal.allow_hosts` replaces the openrouter.ai egress allowlist of the isolated sandbox (`env_registry.modal_run_settings`).
    `resume_modal._step_config` reads the override with `yaml.safe_load`, so `[<BOX_IP>]` becomes a one-element list.
  - `provider_preferences=null` drops OpenRouter's `provider` field. vLLM would ignore it, but it should not be sent.
  - The key travels as a Modal secret holding `VLLM_API_KEY`. Create `vllm-secret-gigi` once, which is free, and
    launch with `MODAL_SECRETS=vllm-secret MODAL_SECRET_SUFFIX=-gigi`. The OpenRouter key never reaches the box.
- **Not verified.** Two points need the first smoke resume: that Modal's CIDR allowlist passes a non-443 port, and
  that plain HTTP to the box is acceptable. The traffic is environment transcripts, with no keys other than the vLLM
  bearer token. A TLS tunnel works too, but its IPs can rotate during a launch.

### 6. Re-run the swap with the distilled gemma

- **Same points, same harness, new folder.** Use `swap_all.sh` as is, with the GEMMA overrides above. The results go
  to `results/raw/swap_all/<env>/<point>/gemma_distill_v1/` (and `gemma_vllm_base/` for the control), which are new
  folders next to `qwen/` and `gemma/`, so nothing existing is touched. The points are those of
  `swap_all_score.PLAN` (`swap_all_plan.json`, `swap_extra_plan.json`, `swap_aird_plan.json`,
  `swap_clean_plan.json`), plus `swap_precommit_plan.json` for precommit_hook.
- **Throttle to the single server.** `swap_all.sh` starts every point with `--detach` within minutes. That is fine
  for OpenRouter, but one vLLM server would queue past the client's 600 s request timeout and lose runs. Run the
  launches in the foreground with `xargs -P 4` (about 40 sandboxes alive at once). Launch from the box, detached, so
  the laptop can sleep. The box needs a Modal token in its `.env` and a checkout holding the `step-K` folders of the
  points.
- **Scoring.** `swap_all_score.py` and `gap_replays.py` read `who` folders, so they need `gemma_distill_v1` and
  `gemma_vllm_base` added as values of `who` (a small change, not made here).
  - **Primary comparison**: swap points whose original run is in a held-out family, from `cheating_families` and the
    family lists in `distill/stratified/manifest.json`.
  - **Secondary comparison**: points in training families. There the model has seen Qwen's own continuation, so this
    measures memorisation, not generalisation.
  - Every rate is given with its n and a Wilson 95% interval, and lost runs are counted apart, never as behaviour.
- **Cost per arm (estimate).**
  - **Serving**: 7,254 gemma turns of about 800 tokens each is about 6 M generated tokens. Prefill is mostly prefix
    cache hits within a point. With tool execution in the sandboxes setting the pace, that is about 3-6 H100-hours,
    or $8-15 at $2.50/h. The gap environments alone (853 continuations) take about half.
  - **Modal sandboxes** (the spar2026 workspace, sandbox rates): $0.0000394 per core-second and $0.0000067 per
    GiB-second ([modal.com/pricing](https://modal.com/pricing)), so about $0.33 per hour for 1 core and 8 GiB. 40
    sandboxes for 3-6 h comes to about $40-80 per arm, all environments. Calibrate this against the real bill of the
    21 Oct swap fleet before approving.
  - **Judge**: $0 for the gap environments, which have mechanical measures. Judged environments cost extra; estimate
    them by turn count (Opus grading is about $1 per long try).
- **Total for both arms** (distilled and control): about 6-12 H100-hours ($15-30) plus about $80-160 of Modal. The
  gap environments alone cost about half of that.

## One-command box script (outline)

Launch once, detached, as soon as the box is reachable:
`setsid nohup bash distill_box.sh > /workspace/logs/box.log 2>&1 < /dev/null &`, then check that `ps -o tty` shows `?`.

```bash
#!/usr/bin/env bash
# distill_box.sh: the whole remaining sequence on the H100 box (CLAUDE.md "Long jobs and GPU boxes").
set -uo pipefail
W=/workspace; LOG=$W/logs; OUT=$W/out/v1; DATA=$W/data/distill/stratified-v1; mkdir -p "$LOG" "$OUT"
set -a; . "$W/.env"; set +a                      # HF_TOKEN, VLLM_API_KEY, MODAL_TOKEN_*: in a file, never in chat or git
HF=model-hermeneutics/gigi-lie-detection-activations
stage () { local n=$1; shift; echo "$(date -u +%FT%TZ) START $n" >> "$LOG/stages.txt"
           "$@" > "$LOG/$n.log" 2>&1; local rc=$?; echo "$(date -u +%FT%TZ) END $n rc=$rc" >> "$LOG/stages.txt"; return $rc; }
up () { stage "upload_$1" hf upload "$HF" "$2" "distill/v1/$1" --repo-type dataset || true; }   # an upload never stops the job

# 0. setup checks: GPU >= 80 GB, disk, RAM, pinned packages, model and data downloaded
stage setup bash -c 'nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /workspace && free -g &&
  pip install torch==2.8.0 transformers==4.57.1 peft==0.17.1 trl==0.24.0 accelerate==1.10.1 datasets==4.2.0 pillow==11.3.0 &&
  hf download google/gemma-3-27b-it && hf download '"$HF"' --repo-type dataset --include "distill/stratified-v1/*" --local-dir /workspace/data' \
  || { echo "SETUP FAILED: box idle, still billing" >> "$LOG/stages.txt"; exit 1; }

# 1. smoke gate: the 4 longest examples, real memory, finite gradients; aborts cleanly
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage smoke python distill_train.py train --data-dir "$DATA" --out "$W/out/smoke" --max-len 32768 --offload-activations --smoke \
  || { up smoke_log "$W/out/smoke/log.json"; echo "SMOKE FAILED: box idle, still billing" >> "$LOG/stages.txt"; exit 1; }
up smoke_log "$W/out/smoke/log.json"

# 2. train, with a periodic log upload as its own failure domain; resume up to 3 times after a crash
( while sleep 1800; do hf upload "$HF" "$OUT/log.json" distill/v1/log.json --repo-type dataset >/dev/null 2>&1; done ) & UPL=$!
for try in 1 2 3; do
  stage "train_$try" python distill_train.py train --data-dir "$DATA" --out "$OUT" --max-len 32768 --offload-activations \
      --eval-every 100 --final-val-sample 1000 --resume && break
done
kill $UPL
up adapter "$OUT/adapter"; up log.json "$OUT/log.json"; up val_per_example.jsonl "$OUT/val_per_example.jsonl"

# 3. merge (independent of the uploads above)
stage merge python distill_train.py merge --adapter "$OUT/adapter" --out "$OUT/merged"

# 4. serve the distilled model and leave it running; the swap launcher (step 6) is a separate, approved stage
pip install vllm==0.11.0 > "$LOG/vllm_install.log" 2>&1
setsid nohup vllm serve "$OUT/merged" --served-model-name gemma-distill-v1 --quantization fp8 --max-model-len 131072 \
  --gpu-memory-utilization 0.92 --enable-prefix-caching --limit-mm-per-prompt '{"image":0}' \
  --api-key "$VLLM_API_KEY" --port 8000 > "$LOG/vllm.log" 2>&1 < /dev/null &
echo "$(date -u +%FT%TZ) DONE: training finished, vLLM serving; box still billing (never stopped automatically)" >> "$LOG/stages.txt"
```

**Rules this follows (CLAUDE.md).**
- The job runs detached and logs every stage to a file.
- The smoke gate aborts cleanly.
- Each artifact goes to HF as a stage as soon as it exists, and the training log goes up every 30 minutes.
- An upload or merge failure does not stop the stages after it.
- **There is no automatic stop: the box is stopped only when the user says so.** When a stage ends or fails, tell the
  user that the box is idle and still billing.
- Check on the run every few tens of minutes.
- Give the user one paste-ready line naming the Vast instance ID.
- Before stopping the box, check whether the swap serving (step 4-6) can reuse it, since the model is already downloaded.

## What was and was not verified (3 Nov 2026, laptop, no GPU, no spend)

- **Verified** with a 17 M-parameter random Gemma3ForConditionalGeneration, built from config with gemma's real
  tokenizer, on the pinned versions (CPU/MPS):
  - `distill_train.py train --smoke` runs end to end: tokenisation, the eval at step 0, 2 optimizer steps, a
    checkpoint, the adapter save, the per-example eval with the adapter on and off, and log.json.
  - `--resume` continues from the last checkpoint and keeps the log history.
  - The chunked target-only loss equals the model's own `labels` loss, to 1e-6.
  - The completion ends in `<end_of_turn>`, and the prompt ends in `<start_of_turn>model\n`.
  - LoRA lands on the text layers only (65,536 trainable parameters = 4 layers x 7 projections).
  - `merge` writes a loadable model.
  - On CPU, reentrant checkpointing with `save_on_cpu` gives the same loss and gradient norm as the non-reentrant
    path.
- **Not verified**:
  - All memory figures are arithmetic, not measurements.
  - The offload path on CUDA: on Apple MPS it gave NaN gradients, and the smoke gate now fails on NaN.
  - The throughput.
  - vLLM's loading of the merged checkpoint with fp8.
  - The provider patch and the Modal allowlist for a raw IP and port.
  - The Modal cost.
