# Week 6 run log

Appended by scripts/01_lead_curve.py.

## 2026-10-14 20:44:38Z — setup

GPU: NVIDIA A100 80GB PCIe (81920 MiB), driver 580.126.20, CUDA 13.0.
`python -m pytest tests/ -q`: 5 passed (after `uv pip install pytest scikit-learn` in `/venv/main`).
Checked data paths: `data/docent_per_action` (missing), `spar-model-hermeneutics/steering-vectors-control/data/docent_per_action` (missing; no repo checkout in tree).
`DOCENT_API_KEY`: not in process environment; not in `ravi/week1-dyl-vs-oa/dyl-probe-repro/.env` (file absent) or `/workspace/.env` (absent).
`HF_TOKEN`: not found in the same locations (needed for Gemma extract after data exists).

## 2026-10-14 20:45:11Z — STOPPED (stage 1: mark-only)

No per-action JSON found. Pass `--data-dir` pointing at `docent_per_action` (`attack/` and `honest/` full of JSON files). If `DOCENT_API_KEY` is set, download with `steering-vectors-control/scripts/download_docent_trajectories.py` into `data/`. Do not invent trajectories.

Later stages (smoke extract, full extract, fit, `CURVE.md`) did not run.

### Unblock

1. Export `DOCENT_API_KEY`, clone https://github.com/mandliya/steering-vectors-control, install `docent` + deps, run download into `ravi/week6-early-detection/data/`, **or** place an existing `docent_per_action` tree at `data/docent_per_action` or pass `--data-dir`.
2. Put `HF_TOKEN` in `ravi/week1-dyl-vs-oa/dyl-probe-repro/.env` or `/workspace/.env` for Gemma.
3. Re-run the pipeline from `AGENTS.md` (`--mark-only`, smoke extract, full extract, `--fit-only`).

## 2026-10-14 21:06:24Z — mark tokens

ascii_sandbagging: 310 rows, 191 lies, 119 truthful
sycophancy_answer: 158 rows, 38 lies, 120 truthful
sycophancy_feedback: 201 rows, 116 lies, 85 truthful
mask_factual: 86 rows, 55 lies, 31 truthful
mask_known_facts: 71 rows, 43 lies, 28 truthful
mask_statistics: 47 rows, 22 lies, 25 truthful
mask_continuations: 87 rows, 73 lies, 14 truthful
mask_disinformation: 69 rows, 58 lies, 11 truthful
capture_the_secret: 378 rows, 73 lies, 305 truthful
alibi: 1276 rows, 784 lies, 492 truthful
unanswerable: 93 rows, 86 lies, 7 truthful

t* is reply token 0 for every flavor. The committed act is the opening of the reply.
Offsets 1 and above are empty unless a reply somehow started after token 0, which this rule does not produce.
ascii_sandbagging: reply tokens min=1 p50=1 p90=1 max=1; one-token=310; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
sycophancy_answer: reply tokens min=1 p50=3 p90=6 max=60; one-token=2; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
sycophancy_feedback: reply tokens min=68 p50=140 p90=200 max=200; one-token=0; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
mask_factual: reply tokens min=2 p50=200 p90=200 max=200; one-token=0; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
mask_known_facts: reply tokens min=2 p50=64 p90=411 max=500; one-token=0; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
mask_statistics: reply tokens min=24 p50=500 p90=500 max=500; one-token=0; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
mask_continuations: reply tokens min=161 p50=459 p90=500 max=500; one-token=0; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
mask_disinformation: reply tokens min=150 p50=500 p90=500 max=500; one-token=0; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
capture_the_secret: reply tokens min=1 p50=230 p90=306 max=400; one-token=3; sequences longer than 4096=46; in-reply offsets 1–32 before t*: 0 (t* is token 0)
alibi: reply tokens min=1 p50=118 p90=171 max=326; one-token=1; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)
unanswerable: reply tokens min=177 p50=585 p90=700 max=700; one-token=0; sequences longer than 4096=0; in-reply offsets 1–32 before t*: 0 (t* is token 0)

## 2026-10-14 21:07:21Z — extract tokens start

cache=/root/spar-model-hermeneutics/ravi/week6-early-detection/results/token_cache_smoke flavors=mask_known_facts, alibi max_n=8 batch=1 max_length=8192

## 2026-10-14 21:17:23Z — extract tokens start

cache=/root/spar-model-hermeneutics/ravi/week6-early-detection/results/token_cache_smoke flavors=mask_known_facts, alibi max_n=8 batch=1 max_length=8192

## 2026-10-14 21:25:27Z — extract tokens

wrote /root/spar-model-hermeneutics/ravi/week6-early-detection/results/token_cache_smoke for mask_known_facts, alibi

## 2026-10-14 21:25:58Z — extract tokens start

cache=/root/spar-model-hermeneutics/ravi/week6-early-detection/results/token_cache flavors=ascii_sandbagging, sycophancy_answer, sycophancy_feedback, mask_factual, mask_known_facts, mask_statistics, mask_continuations, mask_disinformation, capture_the_secret, alibi, unanswerable max_n=None batch=1 max_length=8192

## 2026-10-15 00:03:18Z — extract tokens

wrote /root/spar-model-hermeneutics/ravi/week6-early-detection/results/token_cache for ascii_sandbagging, sycophancy_answer, sycophancy_feedback, mask_factual, mask_known_facts, mask_statistics, mask_continuations, mask_disinformation, capture_the_secret, alibi, unanswerable

## 2026-10-15 00:07:54Z — fit tokens

# Token lead time

t* is the first reply token. In-reply offsets before it are empty when the reply opens on the act.

## ASCII sandbagging

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.999 | 191 | 119 | 0.553 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 1.000 | 191 | 119 | 0.448 |

## Sycophancy A

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.776 | 38 | 120 | 0.469 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.773 | 38 | 120 | 0.593 |

## Sycophancy B

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.887 | 116 | 85 | 0.515 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.886 | 116 | 85 | 0.530 |

## MASK provided facts

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.605 | 55 | 31 | 0.532 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.582 | 55 | 31 | 0.597 |

## MASK known facts

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.654 | 43 | 28 | 0.555 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.796 | 43 | 28 | 0.525 |

## MASK statistics

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.542 | 22 | 25 | 0.430 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.529 | 22 | 25 | 0.487 |

## MASK continuations

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.405 | 73 | 14 | 0.497 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.354 | 73 | 14 | 0.598 |

## MASK disinformation

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.828 | 58 | 11 | 0.471 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.754 | 58 | 11 | 0.428 |

## Capture the secret

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.622 | 73 | 305 | 0.554 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.574 | 73 | 305 | 0.493 |

## Alibi

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.732 | 784 | 492 | 0.449 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.672 | 784 | 492 | 0.443 |

## Unanswerable

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last |  | 86 | 7 |  |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) |  | 86 | 7 |  |

ASCII sandbagging: τ90 is undefined because the reply is one token. Sycophancy A: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.776. Sycophancy B: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.887. MASK provided facts: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.605. MASK known facts: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.654. MASK statistics: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.542. MASK continuations: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.405. MASK disinformation: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.828. Capture the secret: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.622. Alibi: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.732. Unanswerable: no lead offset reached 0.90.
Offset 0 is higher than every earlier scored offset for ASCII sandbagging, MASK known facts. The probe is reading the bad token as it is emitted.
Transfer at τ90 is not measured, because no flavor has a τ90 with a transfer score.
The monitor gap is not measured.

## 2026-10-15 00:24:45Z — goodfire start

ImpossibleBench Kimi traces, synthetic difference of means, Gemma 27B layer 58

## 2026-10-15 02:23:38Z — goodfire fit

# Goodfire difference-of-means, read by Gemma 27B

Direction: mean residual on synthetic hack spans minus mean residual on the matched honest spans, Gemma 27B layer 58. Not fit on the real rollouts.
Traces: ImpossibleBench, Kimi K3, judge labels from the Goodfire release. Ambiguous passages dropped. A prefix longer than 2048 tokens is left-truncated; the scored token is the end of the passage.

Selected contrast on the train half of rollouts: `pl_peek`.

| Contrast | Train AUROC | Test AUROC | Synthetic hack spans | Synthetic honest spans |
| --- | ---: | ---: | ---: | ---: |
| i_rm_test | 0.503 | 0.503 | 200 | 200 |
| i_edit_test | 0.285 | 0.286 | 200 | 200 |
| i_conftest | 0.214 | 0.223 | 200 | 200 |
| i_env_var | 0.231 | 0.233 | 200 | 200 |
| i_print_line | 0.385 | 0.368 | 200 | 200 |
| i_write_score | 0.263 | 0.269 | 200 | 200 |
| i_expl_h | 0.342 | 0.337 | 189 | 189 |
| i_expl_f | 0.482 | 0.478 | 189 | 189 |
| i_expl_r | 0.271 | 0.275 | 189 | 189 |
| i_peek_h | 0.727 | 0.725 | 189 | 189 |
| i_peek_f | 0.747 | 0.739 | 189 | 189 |
| i_peek_r | 0.563 | 0.552 | 189 | 189 |
| pl_suite | 0.235 | 0.239 | 800 | 200 |
| pl_validator | 0.385 | 0.368 | 200 | 200 |
| pl_scorefile | 0.263 | 0.269 | 200 | 200 |
| pl_acquire | 0.747 | 0.742 | 1134 | 378 |
| pl_peek | 0.749 | 0.743 | 567 | 189 |
| pl_exploit | 0.320 | 0.318 | 567 | 567 |

Held-out passage AUROC for `pl_peek` is 0.743 (641 positive passages in 5898 scored passages).

On the test rollouts that contain an attempted or enacted hack, the probe fires on an earlier step in 6 of 13 (0.462). 6 of those fires are on an earlier assistant step; the rest are an earlier channel of the same step. Median lead, among those that fire early, is 10.000 assistant steps. The threshold is the 95th percentile of the probe on train passages the judge called clean, so about 5% of clean train passages would fire.

This is Gemma reading Kimi's traces. It is not the paper's Kimi, GLM, or Qwen probe, and it does not rerun their resampling test.
