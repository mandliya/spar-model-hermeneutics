# gigi

Replicating two numbers from Anthropic's [lie-detectors post](https://alignment.anthropic.com/2026/lie-detectors/) on `google/gemma-3-27b-it`, per task-group category: **lying rate** (how often it lies) and **doubling-down rate** (how often it denies the lie when asked "did you lie?"). Each category below is one experiment — those two numbers, both with 95% CIs.

| Folder | Category | Ground truth | Status |
|---|---|---|---|
| [`ascii-sandbagging/`](ascii-sandbagging/README.md) | sandbagging · ascii | external — gemma's own demonstrated capability | done, 310 rows, plus a 12-condition self-report battery |
| [`sycophancy/`](sycophancy/README.md) | sycophancy — both tracks | track A: neutral vs. pressured answer; track B: same argument under different social framings | done — track A 158 rows, track B 339 rows, plus a 12-condition self-report battery on both (4,308 rows) |
| [`mask-factual/`](mask-factual/README.md) | mask-factual · `mask_provided_facts` | ground truth stated directly in-prompt (true fact + pressured false narrative), judged | done, 86 rows — `mask_known_facts`/`mask_statistics` not started, need external fact-checking |

Skipped on purpose, per the reliability check in `ascii-sandbagging/HANDOFF.md` §8: self-sycophancy (46% label instability), mask-roleplay (contested labels), offpolicy (imported data, not gemma's own behavior), unanswerable (residual bucket, not a lie type — never a rate).

**Caveat on sycophancy/mask-factual's cited reliability**: their 100%/80%
figures are the paper's own judge-vs-human agreement, from one annotator
spot-checking 10 examples per category after the judge pass — not a
systematic validation, and not about the released `did_lie` column (unusable
here regardless, see `ascii-sandbagging/CLAUDE.md`). The paper also warns
that naive neutral-vs-pressured answer-change detection produces false
positives from pure sampling variation — this project addresses that by
running every generation at temperature 0 (greedy decoding, deterministic)
rather than by measuring and subtracting a noise floor; see the top-level
`RESULTS.md`'s methodology section for why.
