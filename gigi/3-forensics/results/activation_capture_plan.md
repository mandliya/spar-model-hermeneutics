# Capturing activations for NLA and J-space on the new heroes (plan, 10 Nov 2026)

On the 10 Nov call we dropped Llama and made gemma-3 27B and Qwen3.6 27B the heroes. TQ wants to run NLA and J-space on them, and
those read activations, not text. The transcripts alone only allow the text pre-work John suggested (bag of words, or a regression
over the top tokens), which is why the transcripts go to Hugging Face first. For the real thing we replay every hero transcript
through the model itself on a rented GPU box and save what the chosen layers produced. Nothing is generated: the model reads the
exact conversation it had, token for token, and we record its internal state while it reads its own replies.

## What is ready

- `scripts/hf_upload_transcripts.py` puts the transcripts on `model-hermeneutics/gigi-forensics-transcripts`, redacted, with a
  manifest and a README that says which setting each stage ran under.
- `scripts/capture_activations.py` rebuilds the token stream (Qwen3.6 with native tools through its chat template; gemma with the
  harness's own text-mode tool preamble, the same function the provider used), runs one forward pass per conversation in bf16, and
  saves float32 activations per run: the per-reply mean and last token at each chosen layer, optionally every reply token, plus the
  input ids and an index with the reply boundaries. Each run is uploaded as soon as it is written, and runs already in the dataset are
  skipped, so a stopped box resumes where it was.
- `scripts/capture_activations_box.sh` is the whole sequence for a fresh box: install, download the transcripts, a two-transcript
  smoke per model as the gate, then the run. It never stops the box.
- A CPU dry run on the laptop with a tiny stand-in model (Qwen2.5 0.5B) went through two Qwen3.6 and two gemma transcripts from the
  hero8 coding fleet: the reply spans were prefix-consistent with the full render, the tool list came out as the harness built it
  (`execute_command` plus the synthetic `end_task`), and the safetensors file had the expected shapes.

## How big the job is

Counted over the transcripts on disk (characters divided by 3.6 as a rough token count):

| transcripts | runs | tokens the model reads | tokens in its own replies | longest run |
|---|---|---|---|---|
| gemma, no check (19-26 Oct) | 309 | about 1.5M | about 0.5M | about 96k |
| gemma, hard rule + example (9 Nov) | 371 | about 4.1M | about 1.5M | about 157k |
| Qwen3.6, soft check (7-8 Nov) | 382 | about 9.9M | about 5.8M | about 190k |
| Qwen3.6, hard rule + example (9 Nov) | 370 | about 10.2M | about 5.9M | about 189k |

The no-check Qwen3.6 fleet that is running now should land near its soft-check numbers, so plan on about 10M tokens for Qwen3.6
and 2 to 4M for gemma per fleet. A 27B model in bf16 on an H100 reads roughly 1,500 to 3,000 tokens a second in a plain forward pass,
so one Qwen3.6 fleet is one to two hours of compute and one gemma fleet under an hour, plus 15 to 30 minutes per model to download the
weights (about 54 GB each). Call it 3 to 5 GPU hours for both heroes on one fleet each, which at Vast's $1.5 to $2.5 an hour for an
H100 or RTX PRO 6000 is $5 to $15. Runs longer than the cap (65k tokens by default) are skipped and listed; the handful of Qwen3.6 runs
near 190k tokens would need the cap raised and a box with memory to spare, or they stay out.

The storage is the real constraint, not the compute. Per-reply mean and last token at three layers is small: about 12,000 Qwen3.6
replies times 5,120 numbers times 4 bytes times 2 vectors times 3 layers is roughly 1.5 GB per fleet, and gemma's 3,700 replies far
less. Every reply token at three layers in float32 is about 360 GB for one Qwen3.6 fleet, so the box script saves mean and last by
default and turns token-level on (`TOKEN_LEVEL=1`) only for the runs TQ names.

## Open questions for TQ, before the box is rented

1. Which layers? The script defaults to 16, 32 and 48 of 64 for Qwen3.6 and the same indices for gemma's 62. NLA and J-space on Llama
   used one layer (53 of 80, about two thirds of the way up); the matching choice here is around layer 42 for both.
2. Which tokens? Mean and last token of every reply are cheap and always saved. Token-level for everything is hundreds of GB; token-level
   for the cheating runs and their matched clean runs is manageable.
3. Which runs first? The no-check hero fleets are the heroes' own setting, so they come first; the soft-check and hard-rule fleets are
   there if the comparison across settings is wanted.
4. Does J-space need anything beyond the residual stream at a layer (for example the attention output or the MLP output separately)?
   The hook captures the layer's output, which is the residual stream after that layer.

## Two things to check on the box before trusting the run

- The smoke's index.json must say `tools_source` is the environment's own function (for example `tools.build_tools([...])` or
  `tools.get_tools(config)`), not the fallback; and `header_included` should be false for Qwen3.6. If the harness package failed to
  install, the fallback is `execute_command` only and the token stream of the system prompt would be wrong for tasks with more tools.
- Qwen3.6 is a hybrid model (Gated DeltaNet layers). Without the `flash-linear-attention` and `causal-conv1d` kernels transformers uses
  a slow fallback; the box script tries to install them and says so in the log if it could not. Transformers must be recent enough to
  know `qwen3_5` (the script pins 4.57 or newer; the laptop dry run used 5.22).

## The line GG pastes on the box

Rent an RTX PRO 6000, H100 or A100 80GB (never GB10), put the two scripts and a `.env` with `HF_TOKEN=...` in one folder, then:

```
cd /workspace && setsid nohup bash capture_activations_box.sh > capture_box.log 2>&1 &
```

Progress shows in `capture_box.txt` next to the scripts and in the dataset itself under `activations/<model>/<stage>/`. The box keeps
running when it is done; stop it from the Vast console once the dataset has every run.
