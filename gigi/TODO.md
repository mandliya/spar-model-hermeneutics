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
- HF repo tidied: README (from `HF_DATASET_CARD.md`) added, 8,680 stale
  loose root tensors removed, every first-run file moved to `first_run/`.

## Still open

- [ ] RUNNING on box 118.163.199.123:18618 (`run_pa_swap.sh`, log
      `run_pa_swap.log`): P("A") on every A/B answer, then the A/B swap test on
      the 626-row subset. Results upload to HF as each category finishes.
      Write-ups go in `answer_probability/README.md` and `ab_swap/README.md`.
      The box does NOT stop itself.

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
