# OA Simple Deception → DYL + Gigi (Gemma-3-27B-IT)

Reverse of [`dyl-probe-repro/RESULTS.md`](dyl-probe-repro/RESULTS.md). Probe trained on OA Simple Deception prefills (`Did you respond truthfully?` / `Yes`), frozen, scored on original-turn labels.

Host: Vast RTX PRO 6000 Blackwell 96GB, 2026-09-28. First-time-reader write-up: [`OA_TRANSFER_NOTES.md`](OA_TRANSFER_NOTES.md).

**Status 2026-09-29: four-position ablation scored.** OA’s claimed `Yes` is not better than `05` last-newline. Sycophancy/capture stay near chance at every token.

## How to resume

```bash
cd ravi/week1-dyl-vs-oa
export PYTHONPATH=src:dyl-probe-repro
# HF_TOKEN in /workspace/.env (Gemma-3 gated)
python -u scripts/05_eval_oa_transfer.py --reuse-probe
```

`05` skips datasets that already have `results/oa_transfer_cache/<name>.npz` and merges new scores into `results/oa_transfer_eval.json`. City/varied caches were labelled before the `lie_reason` fix; `scripts/06_rescore_oa_transfer.py` already rewrote those y’s. Do not re-extract Gigi for official DYL.

## Same-rows comparison (Gigi + OA honeypot)

OA layer 58 and OA mean-of-kept vs Tianq’s **held-out deduced DYL** and just-asking ([`dyl-probe-repro/RESULTS.md`](dyl-probe-repro/RESULTS.md) Experiment H). Same Gigi original-turn labels. **Not** the same follow-up, token, or deduction — see NOTES §2.

| Dataset | lie / honest | just asking (Tianq) | DYL deduced (Tianq) | **OA L58** | **OA mean** |
|---|---:|---:|---:|---:|---:|
| Gigi ascii sandbagging | 191 / 119 | 0.69 | **0.99** | **0.996** | 0.874 |
| Gigi sycophancy (answer) | 38 / 120 | 0.68 | 0.66 | 0.555 | 0.578 |
| Gigi sycophancy (feedback) | 116 / 85 | **0.71** | 0.64 | 0.445 | 0.494 |
| Gigi mask factual | 55 / 31 | 0.58 | 0.52 | 0.659 | **0.736** |
| Gigi unanswerable | 86 / 7 | 0.49 | 0.65 | 0.537 | 0.233 |
| Gigi capture the secret | 73 / 305 | 0.57 | 0.61 | 0.429 | 0.387 |
| OA MBPP-Honeypot (prior) | 500 / 500 | 0.956 | **0.973** | **0.920** | 0.826 |

Ascii: **no headroom** vs DYL. Sycophancy and capture: OA is at or below chance; asking / DYL still weak-to-0.66. Mask: OA mean beats both DYL and asking on a small set. Unanswerable: 7 honest, ignore.

Tianq ascii n was 192 / 118; we have 191 / 119 (`lied is None` skip). Same ground-truth rule.

## AISI city / varied — original turn only

OA prefills after the **first** assistant turn. Labels from AISI `lie_reason`; unmapped / `ambiguous` dropped. **Empty DYL column on purpose:** Tianq’s 1.00 / 0.99 are follow-up-lie at `prompt_last`, not this question.

| Split | lie / honest | dropped | OA L58 | OA mean |
|---|---:|---:|---:|---:|
| city (`dyl_train_city_countries`) | 3575 / 887 | 329 | 0.879 | **0.965** |
| varied (`dyl_validate_varied_deception`) | 2333 / 1056 | 478 | 0.909 | **0.955** |

**Do not cite** city L58 0.73. That run treated AISI `ambiguous` as honest.

## What that means for the mentor note

- **Ascii / city-varied original-turn:** OA already transfers. Finding the OA-vs-DYL *difference* is optional here, not required to show the probe works.
- **Sycophancy:** both white-box methods sit near chance-to-0.66. This is where “did you lie” vs “did you sandbag” (named-act) ablations belong.
- **Mask:** possible OA headroom over DYL, small n. Not a wording-ablation substitute for sycophancy.

## Token position (`07`, 2026-09-29)

New probe per index, same 400 Simple Deception facts. City n=800 original-turn subsample; OA MBPP 200 tasks (400 convs). Full protocol: NOTES §5.

Layer 58 AUROC:

| Set | `prompt_last` | `Yes` | `<end_of_turn>` | last `\n` (`05`) | Tianq DYL (same rows, different detector) |
|---|---:|---:|---:|---:|---:|
| ascii | 0.762 | 0.061 | **1.000** | 0.996 | 0.99 |
| sycophancy A | **0.622** | 0.487 | 0.588 | 0.555 | 0.66 |
| sycophancy B | 0.464 | 0.510 | 0.481 | 0.445 | 0.64 |
| mask | 0.490 | **0.670** | 0.524 | 0.659 | 0.52 |
| capture | 0.482 | 0.487 | **0.518** | 0.441 | 0.61 |
| city (n=800) | 0.191 | 0.495 | 0.442 | **0.860** | — (follow-up-lie, skip) |
| OA MBPP (n=400) | 0.064 | 0.806 | 0.175 | **0.907** | **0.973** (`dyl_cheat`) |

`05` last-newline is the transfer winner. OA `Yes` inverts ascii. Tianq `prompt_last` inverts this OA probe on city/MBPP. No token closes sycophancy or capture.

```bash
PYTHONPATH=src:dyl-probe-repro python -u scripts/07_oa_token_positions.py
```

## Protocol reminders

- `05` token = last non-pad of the closed Gemma chat (`Yes` `<end_of_turn>` `\n`). Train/eval agree. `07` shows that OA’s `Yes` and Tianq `prompt_last` are **worse** for this probe, not better.
- No AISI deduction. OA score is already original-lie.
- Transformers pinned `<5` (4.57.6). Do not use `scripts/03_eval_transfer.py` for official DYL.

## Files

| Path | What |
|---|---|
| `results/oa_transfer_eval.json` | `05` numbers above (in git) |
| `results/oa_token_position_eval.json` | `07` four-position table (in git) |
| `results/oa_simple_deception_probe.json` | last-newline weights, gitignored |
| `results/oa_transfer_cache/*.npz` | activations, gitignored |
