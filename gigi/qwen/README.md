# qwen: the gemma pipeline, rerun on a second model (Qwen3-32B)

The same experiment as for gemma-3-27b-it, on `Qwen/Qwen3-32B`. Same prompts
(the dataset's gemma configs), same ground-truth checks, same four follow-up
questions, float32 activations, confidence scores, A/B swap test, and
third-person monitors. Results go to `results/qwen3-32b/` here and to
`models/qwen3-32b/` on the HF dataset, so gemma's results are never touched.

## Decisions (user, 5 Oct 2026)

- Model: **Qwen3-32B**, thinking mode **off** (answers straight away, like
  gemma). Forced in `src/common.load_tokenizer()`.
- Answers generated **on the GPU instance, one conversation at a time**, as
  for gemma. Open-weight models are never run via API (`chat()` refuses the
  target model).
- Qwen as a third-person monitor: **batched**, as gemma's monitor run was.
- Boxes: rent when the code is ready; split the categories across ~5-10 GPUs.

## What changed from the gemma scripts

The scripts here are copies of the current gemma pipeline scripts (flat, as
they ran on the box), changed only where needed:

1. `src/common.py`: `TARGET_MODEL` from `MODEL_TAG` only (never from .env),
   temperature pinned to 0, `MODEL_TAG`,
   `RESULTS = results/<MODEL_TAG>/`, `load_tokenizer()` (thinking off),
   `CORE_QUESTIONS`, `chat()` refuses the target model.
2. Every script loads the tokenizer through `load_tokenizer()`.
3. Every activation is saved float32 (was float16 in the older scripts).
4. Follow-up scripts run only `neutral`, `neutral_cot`, `named`
   (`named_cot` comes from `05_named_cot.py`).
5. `src/hf_sync.py`: every upload goes under `models/<MODEL_TAG>/`.
6. `src/rollouts.py`: skips gemma-only files (archived questions,
   mask-factual's named rerun). `04_recapture_fp32.py` uses it, and is only
   needed for ascii, sycophancy A/B and mask-factual (the other follow-up
   scripts capture activations inline).
7. `02_analyse_mask_belief.py`, `15_mask_belief_upload.py`,
   `00_pick_subset.py`: read and write the per-model folder.

Model-independent inputs copied from gemma's run into `results/qwen3-32b/`:
`alibi_guilt_labels.*`, `alibi_subset.csv`, `alibi_rest.csv` (scenario labels),
`unanswerable_question_labels.*` (labels of the dataset's questions).

## How to run (one box per job)

| Job | Categories | Rows |
|---|---|---|
| J1, J2, J3 | alibi, split in 3 (`SHARD=k/3`) | ~425 each |
| J4, J5 | capture-the-secret, split in 2 | ~254 each |
| J6 | ascii + mask-factual | 310 + 87 |
| J6b | mask-factual again (J6's stopped early) | 87 |
| J7 | sycophancy A + B | 158 + 201 |
| J8 | unanswerable + MASK known facts + MASK statistics | 100 + 71 + 49 |
| J9 | MASK continuations | 90 |
| J10 | MASK disinformation | 76 |

Each box: `run_job.sh <JOB>` (smoke-test gate per category, first run,
follow-ups, named_cot, activations, P("A"), A/B swap, transcripts, the model
as its own monitor on the GPU + Gemini/GPT-5 mini via API, activations
tarball). Everything is uploaded to `models/qwen3-32b/boxes/<JOB>/` on HF.
The code bundle the boxes download is in the private HF repo
`ggrazianoknapp/gigi-code` (`qwen.tar.gz`). Afterwards,
`11_merge_boxes.py` (no GPU, no API) combines every box's results into
`results/qwen3-32b/` for the analysis.

Box requirements: one ~96 GB GPU (RTX PRO 6000 or H100 80 GB), >= 150 GB disk.

## Qwen3-Next-80B (the hermeneutics target)

Same pipeline with `MODEL_TAG=qwen3-next-80b` (`Qwen/Qwen3-Next-80B-A3B-Instruct`,
163 GB in bf16, two 96 GB GPUs per box). Run on 9 Oct 2026 on 10 box jobs; all 11
categories; results merged in `results/qwen3-next-80b/` (88 files), per-box files
and activations under `models/qwen3-next-80b/boxes/` on HF. Text check over all
12,124 answers: no corrupted output. Its mask-factual swap test was run on 12 Oct
(`run_swap_mask_factual.sh`). No monitors were run on its transcripts (they are
built: `results/qwen3-next-80b/transcripts/`).

Analysis: `../compare_models.py` covers all three models (gemma, Qwen3-32B,
Qwen3-Next-80B); `19_qwen80b_charts.py` draws the three-model chart and writes the
80B tables in `../RESULTS_HERMENEUTICS.md` ("Background: the target's own results").

## Qwen3-235B (for TQ's activations)

`MODEL_TAG=qwen3-235b` = `Qwen/Qwen3-235B-A22B-Instruct-2507` (the July 2025 Instruct
release, no thinking mode; the paper only says "Qwen3-235B"). 470 GB in bf16, 94 layers,
hidden size 4096, mixture of experts with 22B parameters active per token. Box: eight
96 GB GPUs (or four 141 GB), 700 GB disk. One all-layer activation tensor is 1.56 MB.

Everything runs on the rented box, as for the 80B (the user's choice, 14 Oct 2026; the
alternative, text through OpenRouter and the box only for activations, was about five
times cheaper but mixes an fp8 deployment with bf16 weights). `run_job.sh SMOKE3` is the
first step: 2 conversations through every stage for ascii, alibi and MASK known facts,
to measure the model's real speed before the full run is quoted. The full `SMOKE` job
would mostly pay for model loads (about 90 of them at a few minutes each).

Status: **done** (14 Oct 2026, 05:50-22:46 UTC). Run on 4 boxes (8x RTX PRO 6000 each) with the
same 10 jobs as the 80B (box A = J1 J4 J7, B = J8 J9, C = J2 J5 J10, D = J3 J6); every job finished
and its results were verified on HF. Only failures: 9 follow-up answers on 3 very long
capture-the-secret conversations (GPU out of memory during the all-layer capture; J4), and one
MASK continuations judge verdict that couldn't be parsed (J9). Results merged in
`results/qwen3-235b/` (`11_merge_boxes.py`). Analysis: `../compare_models.py` (all four models) and
`21_qwen235b_charts.py` (chart + tables in `../RESULTS_HERMENEUTICS.md`).

Activations (for TQ): one archive per job, `models/qwen3-235b/boxes/<JOB>/activations_<JOB>.tar.gz`
on HF, 23.9 GB in all.

- **Each tensor** = float32, shape (95, 4096): the embeddings plus the output of each of the 94
  layers, at one token.
- **Two tokens per follow-up answer:** `predecision` (the last prompt token, before the model
  answers) and `answer_token` (the A/B letter of the answer, teacher-forced on the saved reply, as in
  the DYL paper). Some first answers are captured too (`*_activations_file` in the first-run files).
- **Which row is which:** each row names its files (`predecision_activations_file` /
  `answer_token_activations_file`, or `predecision_file` / `answer_token_file`). The recaptured
  follow-ups of ascii, sycophancy A/B and mask-factual are listed in `fp32_map_<category>.jsonl`.
  The swapped-letter rows (`ab_swap_*`) have no activations.
- **Inside an archive:** `workspace/<JOB>/qwen/results/qwen3-235b/activations/` (scripts that
  capture inline), `workspace/fp32_staging/` (`named_cot` and the recaptured follow-ups) and
  `workspace/mask_belief_staging/` (the four MASK belief categories).
- **Archives overlap:** the two staging folders are per box, not per job, so each archive also
  holds the staging files of the jobs that ran before it on the same box. These are identical
  copies under the same path; dedupe by path.

## Substitute step (hermeneutics): gemma reads the target's conversations

`16_substitute_followups.py`: gemma-3-27b answers the four follow-up questions
about another model's conversations, as if they were its own. The target is
named by `MODEL_TAG` (`qwen3-next-80b`).

- **What gemma sees:** the target's conversation exactly as the target saw it
  (system prompt, user turns), with the target's answer in the assistant slot,
  then the same follow-up question word for word. The conversations are rebuilt
  by the code that built them for the target (`src/rollouts.py`,
  `src/mask_belief.py`).
- **What gemma writes:** only the answer to the follow-up. Greedy, one
  conversation at a time, same token limits and parsers as the target's run.
- **Saved per answer:** the reply, the letter, `admitted`, P("A") at the answer
  token; with `--activations`, both capture points in float32.
- **Both letter orders:** `--swap` reruns the questions with the options
  reversed, on the target's own swap subset (`ab_swap_subset.csv`).
- **Output:** `results/gemma-3-27b_reads_<MODEL_TAG>/`, with the target's file
  names (`pa_<category>.jsonl`, `ab_swap_<category>.jsonl`). The target's folder
  is only read. On HF: `models/<MODEL_TAG>/substitute/gemma-3-27b/boxes/<JOB>/`.
- **Rows:** 2,811 conversations x 4 questions = 11,244 answers (11 categories);
  swapped order on 607 conversations = 2,428 answers (7 categories).

Render check (no GPU, no model, free; run from `gigi/qwen/`):

```
MODEL_TAG=qwen3-next-80b uv run --with transformers==5.17.0 --with torch --with pandas \
  --with pyarrow --with fsspec --with huggingface_hub --with python-dotenv --with tqdm \
  --with sentencepiece --with protobuf python 16_substitute_followups.py --render 1
```

It renders every job with gemma's chat template and checks that the target's
answer is the last assistant turn, the follow-up is the last user turn, no
turn's text is lost, and the job names are exactly the target's. Passed for
both letter orders on 12 Oct 2026.

On a box: `MODEL_TAG=qwen3-next-80b bash run_substitute.sh <JOB>` (one >= 80 GB
GPU, >= 150 GB disk; only `HF_TOKEN` needed, no API calls). Render check, then
a 2-conversation smoke test per category as a gate, then the full run.

| Job | Categories | Conversations |
|---|---|---|
| S1, S2 | alibi, split in 2 (`SHARD=k/2`) | ~640 each |
| S3 | capture-the-secret + ascii | 391 + 310 |
| S4 | mask-factual, unanswerable, sycophancy A + B, the four MASK belief categories | 831 |
| ALL | everything on one box | 2,811 |
| SMOKE | every category's smoke test, then stop | 22 |

Built not to need a box restart: downloads are retried (a hung one is killed and
retried), each stage is tried up to 3 times and resumes per answer, rows that
errored are retried, and pasting the same launch line again continues the job (a
second copy refuses to start). Only a failed gate (unparseable answers) stops a
job. The run log is on HF from the first minutes, then every 10 minutes.

Tested on a laptop before launch (12 Oct 2026): the render check on all rows; the
run path end to end with a tiny stand-in model (answers, P("A"), float32
activations, resume, error retry, gate); `run_substitute.sh` on a simulated box
(crash, missing rows, failed gate, wrong GPU, small disk, failed download,
`STOP_WHEN_DONE`). Not testable without the GPU: gemma itself, which the smoke test
covers in each box's first minutes.

`run_swap_mask_factual.sh`: the mask-factual swap test neither Qwen run got
(`run_job.sh` left it out by mistake). One 2 x 96 GB box for the 80B, nothing else
rerun; output `ab_swap_mask_factual.jsonl` under `models/<MODEL_TAG>/boxes/J6swap/`,
where `11_merge_boxes.py` picks it up.

Run on 12 Oct 2026 (four gemma boxes S1-S4, 08:24-12:00 UTC; one 2-GPU box for the
80B swap): all 11,244 answers and all 2,428 swapped ones, 0 errors, 0 unparseable;
the 80B's mask-factual swap, 248 of 248. No job needed a restart. Code bundle
`substitute.tar.gz` in the private HF repo `ggrazianoknapp/gigi-code`; launch lines
(with the key, not in git) in `~/gigi_launch/LAUNCH_SUBSTITUTE.txt`.

Afterwards (no GPU): `17_merge_substitute.py` (boxes -> `results/gemma-3-27b_reads_<MODEL_TAG>/`),
`../substitute_gap.py` (target AUROC - substitute AUROC per category and question,
both letter orders, paired bootstrap -> `results/substitute_gap_*.csv`),
`18_substitute_charts.py` (the AUROC chart and table in `../RESULTS_HERMENEUTICS.md`).
The comparison John asked for on the 12 Oct call is `../four_paths.py` (part 1: lie or
no lie on the first response; part 2: confess / double down / false confession / honest
on Qwen's prefilled trajectory, Qwen vs gemma) and `20_four_paths_charts.py` (its chart
and tables).

gemma's activations on these conversations (both capture points, float32, original
letter order): `models/<MODEL_TAG>/substitute/gemma-3-27b/boxes/<JOB>/substitute_activations_<JOB>.tar.gz`
on HF (S1 3.3 GB, S2 3.3 GB, S3 3.6 GB, S4 4.3 GB); file names are in each answer's
`predecision_file` / `answer_token_file`.

One thing the run showed: a finished box's last log lines never reached HF, so from
outside it looked "still running" (fixed: `final_push`).

## Status (7 Oct 2026)

- **Done: all 11 categories** (J1-J10, J6b). Results are merged in
  `results/qwen3-32b/` (`11_merge_boxes.py`), with per-box files and
  activations under `models/qwen3-32b/boxes/` on HF.
- **J6 and J6b:**
  1. J6 (ascii + mask-factual) ran on a fresh box after the download failures.
  2. Its mask-factual stopped after 34 rows: 12 dataset rows have an empty
     trace and crashed it.
  3. J6b redid mask-factual in full (87 rows, the 12 empty ones skipped), but
     without the A/B swap step. So Qwen has no mask-factual swap test.
  4. `11_merge_boxes.py` leaves out J6's partial mask-factual files, and
     replaces local copies that came from them.
- **Monitors with the paper's exact templates** (`run_paper_monitors.sh`,
  M2) ran on the 9 categories finished by then. J6 and J6b ran them on their
  own boxes.
  1. Qwen's cut-off self-monitor replies were rerun at 2,048 tokens
     (`run_retry_monitor.sh`).
  2. Gemini's 3 cut-off replies were rerun at 4,000 tokens.
  3. Still unscored: 12 of 5,602 Qwen-as-monitor replies (repetition loops,
     or no score given). None for Gemini or GPT-5 mini.
- **Problems found during the run, and fixed in the code:**
  1. Two smoke tests picked rows with nothing to follow up (sycophancy B
     needs a neutral baseline; capture-the-secret needs a pressed secret).
     Smoke tests now pick suitable rows.
  2. GB10 boxes (DGX Spark, shared memory) offloaded Qwen to disk and
     crawled. `run_job.sh` now refuses GPUs under 75 GB.
  3. Model downloads hit Hugging Face transfer errors. Now 5 tries, with the
     plain-HTTP fallback after the first failure.
  4. The dataset was rebuilt on 5 Oct 2026, which renamed a config. The code
     now pins the revision gemma used (`e8fc10f`).
  5. A box with `STOP_WHEN_DONE=1` stayed up overnight because its stop
     depended on an upload check. The stop is now unconditional.

## Results

The cross-model comparison is in [`../RESULTS_REPLICATION.md`](../RESULTS_REPLICATION.md#second-model-qwen3-32b).
It's computed by `../compare_models.py` (same definitions and code path for
both models; it reproduces gemma's published numbers exactly) and charted by
`13_charts.py`.

- **Lying rates** are within 6 points of gemma's in 8 of 11 categories. The
  gaps: ascii (10.0% vs 61.6%), sycophancy B (43.3% vs 57.7%) and
  mask-factual (73.3% vs 64.0%).
- **Self-report AUROC** runs 0.40-0.88 outside ascii. It reaches 0.70 or above
  in 17 of 44 combinations, against 12 of 44 for gemma. That puts it in
  gemma's range, not the paper's best Qwen3-235B number (0.98-0.99; its plain
  self-report is 0.72-0.85, see ../RESULTS_HERMENEUTICS.md, "Why the 235B differs from the paper").
- **ascii:** Qwen says "no" to every question on every row. Its tiny
  probability of "yes" ranks lies at 0.95 under `neutral_cot` and 0.02 under
  `neutral`.
- **On the letter, Qwen denies even more than gemma.** It says "no" to every
  question on sycophancy B and ascii. On alibi `neutral` it admits 27 of 814
  lies.
- **Monitors** (the paper's rubric template): an outside monitor beats Qwen's
  best question in 6 of 11 categories (both do in 5). Qwen as its own monitor
  beats its own best question in 6 of 11. Without the system message (the
  paper's plain 3rd-Person Report), outside monitors fall to near chance on
  alibi (0.52) and mask-factual (0.57), as on gemma's transcripts.

| File | What |
|---|---|
| `results/comparison_lying_rates.csv` | lying rate per model x category, Wilson 95% |
| `results/comparison_counts.csv` | confess / double down / false confess / honest per model x category x question |
| `results/comparison_selfreport_auroc.csv` | AUROC from P("yes") per model x category x question, bootstrap 95% |
| `results/comparison_swap.csv` | swap test: median AUROC with yes at A vs yes at B |
| `results/comparison_monitor_auroc.csv` | monitors (Gemini, GPT-5 mini, the model itself) x prompt, both models |
| `results/qwen3-32b/monitor_<model>_paper.jsonl` | M2's replies on the paper's templates (J6/J6b's are in `monitor_<model>.jsonl`, keyed by template) |

## Cost

- **Monitors, paper templates (measured, reruns included):** $18.27 ($12.02
  Gemini 2.5 Flash, $6.25 GPT-5 mini).
- **Monitors, our earlier templates:** $12.24 measured ($8.35 Gemini, $3.89
  GPT-5 mini).
- **Judge:** about $16-25 (estimated from gemma's run; judge token counts
  aren't saved).
- **GPU:** about 9 boxes x 3.5-5 hours plus failed attempts, at $1.15/h;
  plus the J6, J6b, M2 and retry boxes, including one that sat idle
  overnight (about 19 h).
