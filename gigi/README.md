# gigi

Replicating two numbers from Anthropic's [lie-detectors post](https://alignment.anthropic.com/2026/lie-detectors/) on `google/gemma-3-27b-it`, per task-group category: **lying rate** (how often it lies) and **doubling-down rate** (how often it denies the lie when asked "did you lie?"). Each category below is one experiment — those two numbers, both with 95% CIs.

| Folder | Category | Ground truth | Status |
|---|---|---|---|
| [`ascii-sandbagging/`](ascii-sandbagging/README.md) | sandbagging · ascii | external — gemma's own demonstrated capability | done, 310 rows |
| *(planned)* | sycophancy | neutral vs. pressured answer, run twice | not started — best next category, 100% label reliability |
| *(planned)* | mask-factual | same two-pass method as sycophancy | not started — 80% label reliability |

Skipped on purpose, per the reliability check in `ascii-sandbagging/HANDOFF.md` §8: self-sycophancy (46% label instability), mask-roleplay (contested labels), offpolicy (imported data, not gemma's own behavior), unanswerable (residual bucket, not a lie type — never a rate).
