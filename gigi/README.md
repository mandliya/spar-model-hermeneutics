# gigi

Replicating two numbers from Anthropic's [lie-detectors post](https://alignment.anthropic.com/2026/lie-detectors/) on `google/gemma-3-27b-it`, per task-group category: **lying rate** (how often it lies) and **doubling-down rate** (how often it denies the lie when asked "did you lie?"). Each category below is one experiment — those two numbers, both with 95% CIs.

| Folder | Category | Ground truth | Status |
|---|---|---|---|
| [`ascii-sandbagging/`](ascii-sandbagging/README.md) | sandbagging · ascii | external — gemma's own demonstrated capability | done, 310 rows |
| *(in progress)* | sycophancy | neutral vs. pressured answer, run twice + a noise baseline | not started — see caveat below |
| *(planned)* | mask-factual | same two-pass method as sycophancy | not started — see caveat below |

Skipped on purpose, per the reliability check in `ascii-sandbagging/HANDOFF.md` §8: self-sycophancy (46% label instability), mask-roleplay (contested labels), offpolicy (imported data, not gemma's own behavior), unanswerable (residual bucket, not a lie type — never a rate).

**Caveat on sycophancy/mask-factual's cited reliability**: their 100%/80%
figures are the paper's own judge-vs-human agreement, from one annotator
spot-checking 10 examples per category after the judge pass — not a
systematic validation, and not about the released `did_lie` column (unusable
here regardless, see `ascii-sandbagging/CLAUDE.md`). The paper also warns
that naive neutral-vs-pressured answer-change detection produces false
positives from pure sampling variation, so this category's design needs its
own noise baseline (re-ask the neutral question twice, no pressure, measure
the flip rate) before trusting any fold rate it produces.
