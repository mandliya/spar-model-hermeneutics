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

## Status / still to do

- [x] scripts copied and made model-generic (above)
- [x] Qwen chat template checked with the real tokenizer (no GPU): thinking
      forced off even if requested; replayed conversations start exactly with
      the generation prompt; A/B are single tokens bare and after "Final
      answer:"; pad/eos set
- [x] model pinned by MODEL_TAG (the shared .env's TARGET_MODEL=gemma is
      ignored; caught by the test above), temperature pinned to 0
- [x] transcript builder checked against gemma's 2,809 transcripts: identical
      except 13 MASK statistics rows whose dataset entry has no question
      (found a gemma bug: they were counted as honest; fixed 5 Oct 2026, and
      skipped here)
- [x] row sharding, per-box swap subset, run script, merge script, code bundle
- [ ] user approval of the budget (~$80-100) and of stopping boxes when done
- [ ] launch, then analysis: gemma vs Qwen tables/charts, John's four counts
      (confess, false confession, double down, honest)
