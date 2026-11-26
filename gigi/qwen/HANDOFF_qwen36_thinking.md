# Handoff: Qwen3.6-27B, thinking on, through the part-one lie pipeline

For TQ and TQ's coding agent (26 Nov 2026, from GG's side). TQ asked for Qwen3.6-27B to go through the same pipeline as the
other Qwens: rollouts, judge labels and the four did-you-lie follow-ups. It should run **with thinking on**, and **the
reasoning should be kept in the saved rows**, so that NLA and JSpace readers can show or hide it. That gives the lie-detection
gap with Qwen3.6 as the target, and base gemma and classic-distilled gemma as substitutes. GG's side is busy with four trainings, so
this doc is for running it on your side. Ask GG if anything here is unclear.

> **Update, 26 Nov 2026: GG decided to run this on our side**, so TQ doesn't need to. The changes below are made in
> `src/common.py` (`THINKING=1`) and `run_job.sh` (`MODEL_TAG=qwen3.6-27b`). Decoding uses the model card's sampling
> (temperature 1.0, top_p 0.95, top_k 20, min_p 0), seeded per prompt, as in our forensics runs of Qwen3.6. The
> follow-up turns see the earlier answer without its reasoning. Every row carries a `thinking` list with the
> reasoning of each generation behind it.

## Where everything is

- Code: `gigi/qwen/` in this repo. `README.md` there is the full guide: jobs, boxes, what each script does, the earlier
  runs (Qwen3-32B, Qwen3-Next-80B, Qwen3-235B).
- Shared settings: `gigi/qwen/src/common.py` (`MODEL_TAG`, `MODELS`, `load_tokenizer()`, `TEMPERATURE`, `JUDGE_MODEL`).
- One box = one job: `gigi/qwen/run_job.sh <JOB>`. It runs setup, model download, and then for each category a smoke
  test (the gate), the first run, the follow-ups, named_cot, activations, P("A"), the A/B swap and the transcripts. It
  uploads after every stage.
- Results: `gigi/qwen/results/<MODEL_TAG>/`, and on HF in the dataset
  `model-hermeneutics/gigi-lie-detection-activations` under `models/<MODEL_TAG>/boxes/<JOB>/`.
- Ground truth per category: `gigi/1-replication/<category>/` (each has its own README/CLAUDE.md). The dataset's
  `did_lie` label is never used.

## What has to change for Qwen3.6 with thinking on

1. **Add the model.** In `src/common.py`, add `"qwen3.6-27b": "Qwen/Qwen3.6-27B"` to `MODELS` and run with
   `MODEL_TAG=qwen3.6-27b`. Everything then goes to `results/qwen3.6-27b/` and `models/qwen3.6-27b/` on HF, so no
   earlier model's results are touched.
2. **Turn thinking on.** `load_tokenizer()` forces thinking OFF for every render, by prepending
   `{%- set enable_thinking = false %}` to the chat template. For this run, skip that override (for example behind an
   env switch such as `THINKING=1`, so the other models keep running exactly as before).
3. **Give the reasoning room.** Every `00_*.py` script has its own `generate()` with a small `max_new_tokens`, between
   80 and 700 tokens (e.g. `00_alibi.py` 400, `00_unanswerable_claims.py` 400/700, the self-report scripts 80/300).
   With thinking on, the reasoning comes first and would be cut mid-thought. Raise the budget (several thousand tokens)
   and flag any row where `</think>` never appears.
4. **Split reasoning from answer, and save both.** The decoded text will be `<think>...</think>` followed by the answer.
   Save `reasoning` and `response` as separate fields in every row (rollouts and follow-ups). Every parser must read
   `response` only: answer letters, claims, the A/B swap, and the text the judge sees. The judge has only ever seen the
   final answer for the other Qwens, so keep that unless you decide otherwise.
5. **Decide what the model sees of its own earlier reasoning.** The did-you-lie questions come after the model's
   answer. By default Qwen's chat template drops the `<think>` block of earlier assistant turns. So decide whether the
   follow-up turn sees the earlier reasoning (in our forensics training we render Qwen3.6 with `preserve_thinking=True`)
   or not, and record the choice. The same applies to the teacher-forced replays in `src/answer_probs.py`,
   `src/dyl_activations.py` and `04_recapture_fp32.py`, which rebuild the conversation with the saved answer.
6. **Decoding: settle this with GG before the full run.** The pipeline is greedy (`TEMPERATURE = 0.0`, pinned in
   `common.py`), like every earlier model. Qwen's model card advises against greedy decoding in thinking mode because it
   can loop. Using the model card's sampling settings instead would be a deviation from the other models. It needs a
   fixed seed, the settings saved with every row, and a note next to the results.
7. **Activations.** The jobs capture float32 all-layer activations inline, at the same token positions as before. With a
   reasoning block in the turn, decide which positions NLA/JSpace need (end of reasoning, start of answer, the follow-up
   answer token) before the capture runs. Activations stay float32.

## Box and run

- Qwen3.6-27B is about 54 GB in bf16, so one 80 GB+ GPU per job is enough (H100, A100 80GB, RTX PRO 6000). Never use GB10 /
  DGX Spark. Generation is one conversation at a time with HF transformers (a project rule for this pipeline). Qwen3.6
  is a new hybrid architecture (Gated DeltaNet + gated attention), so check first that the box's `transformers` loads it.
  Our forensics runs serve it with vLLM.
- Start with `MODEL_TAG=qwen3.6-27b bash run_job.sh SMOKE3`: 2 conversations through every stage for ascii, alibi and
  MASK known facts. That measures the real speed with thinking on before you quote the full run.
- Then run the jobs J1–J10 across boxes (the job table is in `README.md`). `MONITORS=0` unless the third-person monitors
  are wanted too.
- Cost guide: Qwen3-32B with thinking off took about 9 boxes x 4 hours (~$60 of GPU), plus ~$20 for the judge
  (`anthropic/claude-sonnet-4.5` via OpenRouter, provider pinned). With thinking on, expect it to take several times
  longer; SMOKE3 gives the real figure.

## House rules this pipeline follows

- Keys only from the environment or a git-ignored `.env` (`OPENROUTER_API_KEY`, `HF_TOKEN`). Never in git or chat.
- Open-weight models never run through an API, only on the box. `chat()` refuses the target model.
- Never delete or overwrite anything in a `results/` folder. Scripts append and skip finished rows; superseded files
  are archived.
- Every rate gets its n and a 95% Wilson interval.
- Any deliberate deviation from the earlier models (thinking on, sampling, longer budgets, what the follow-up sees) is
  written down next to the results.
