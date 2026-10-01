# Open items

Kept in git so they survive a lost session. Delete a line when it's done.

## Done

- All 7 categories run end to end: ascii-sandbagging, sycophancy A/B,
  mask-factual, unanswerable, capture-the-secret, alibi (500-row subset).
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
- P("A") on all 19,049 A/B answers -> AUROC per category and wording
  (`answer_probability/README.md`).
- A/B swap test on 626 rows x 11 wordings (`ab_swap/README.md`).
- HF repo tidied: README (from `HF_DATASET_CARD.md`) added, 8,680 stale
  loose root tensors removed, every first-run file moved to `first_run/`.

## Still open

- [ ] RUNNING on box 118.163.199.123:18618 (`09_repackage_core4.py`, log
      `repackage.log`): new `activations/<category>.tar.gz` with only the four
      kept questions, plus first-run activations for unanswerable, capture and
      alibi pulled into `first_run/`. Then: move the old tarballs on HF into
      `archive/` and upload the new dataset card.
- [ ] Next deception categories, with the four questions only.

## Optional (each needs a GPU box and a go-ahead on cost)

- [ ] float32 for first-run (pre-follow-up) activations. Ask TQ first.
- [ ] The remaining 781 alibi rows (1,281 total, 500 run).
- [ ] `mask_known_facts`, `mask_statistics`: scoped, not started (need
      fact-checking).

## Ruled out, with reasons

- `self-sycophancy` — prompts are visibly corrupted (model output leaked
  into the probe text, duplicate options). Matches the paper's own 46%
  label-instability warning.
- `offpolicy` — traces are from other models, so gemma isn't the actor. It
  measures detecting someone else's lies, which is a different experiment.
