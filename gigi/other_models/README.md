# other_models: the gemma pipeline, rerun on a second model (Qwen3-32B)

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

## Status / still to do

- [x] scripts copied and made model-generic (above)
- [x] Qwen chat template checked with the real tokenizer (no GPU): thinking
      forced off even if requested; replayed conversations start exactly with
      the generation prompt; A/B are single tokens bare and after "Final
      answer:"; pad/eos set
- [x] model pinned by MODEL_TAG (the shared .env's TARGET_MODEL=gemma is
      ignored; caught by the test above), temperature pinned to 0
- [ ] third-person: transcript builder for Qwen + Qwen-as-monitor (batched) +
      Gemini/GPT monitors
- [ ] one run script per box (setup, model download, smoke-test gate, the
      category chains, HF uploads) and a one-line launcher to paste into the
      Vast web terminal (pulls this repo, so no laptop needed)
- [ ] measured cost estimate (GPU seconds per row from gemma's saved
      `gpu_seconds`) -> user approval before any spend
- [ ] after the runs: tables/charts comparing gemma and Qwen, John's four
      counts (confess, false confession, double down, honest)
