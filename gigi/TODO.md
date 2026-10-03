# Open items

Kept in git so they survive a lost session. Delete a line when it's done.

## Done

- All 7 categories run end to end: ascii-sandbagging, sycophancy A/B,
  mask-factual, unanswerable, capture-the-secret, alibi (all 1,281 rows since
  1 Oct 2026: 784 lies, 92.7% of guilty suspects).
- 13 follow-up questions per category, including `named_cot` (1,732 rows),
  all in `RESULTS.md` tables, the balanced-accuracy grid and the charts.
- float32 activations for all 7 categories plus every `named_cot` set on
  HF (`model-hermeneutics/gigi-lie-detection-activations`, now public).
  16/16 files verified; the box was stopped once they were confirmed.
- mask-factual first-run activations: all 87 rows on HF (78 recaptured).
- 1 Oct 2026: the project keeps four follow-up questions (`neutral`,
  `neutral_cot`, `named`, `named_cot`). The other nine are archived in
  `archive/2026-10-01_all_13_wordings/`; RESULTS.md, every category doc,
  the charts and the false-confession files now cover the four.
- HF repo reorganised the same way: `activations/<category>.tar.gz` (four
  questions, float32), `first_run/` for every category, retired files under
  `archive/` (28 files moved, copies checksum-verified), new README.
- P("A") on all 19,049 A/B answers -> AUROC per category and wording
  (`answer_probability/README.md`).
- A/B swap test on 626 rows x 11 wordings (`ab_swap/README.md`).
- HF repo tidied: README (from `HF_DATASET_CARD.md`) added, 8,680 stale
  loose root tensors removed, every first-run file moved to `first_run/`.
- 2 Oct 2026: every first-run activation recaptured in float32 (2,173 files,
  cosine >= 0.9999 against the float16 originals, which are in
  `archive/first_run_fp16/` on HF). Every activation on HF is now float32.
- 2-3 Oct 2026: MASK known facts / statistics / continuations / disinformation
  (299 rows, `mask-belief/`): ground truth from gemma's own belief (softened MASK
  belief questions), statistics cutoff 10%. In RESULTS.md, charts and AUROC.

## Still open

- [ ] Next category: to be chosen (every gemma config with usable rows is now covered;
      remaining: n-truths-and-a-lie has only 2 rows).

## Optional (each needs a GPU box and a go-ahead on cost)


## Ruled out, with reasons

- `self-sycophancy` — prompts are visibly corrupted (model output leaked
  into the probe text, duplicate options). Matches the paper's own 46%
  label-instability warning.
- `offpolicy` — traces are from other models, so gemma isn't the actor. It
  measures detecting someone else's lies, which is a different experiment.
