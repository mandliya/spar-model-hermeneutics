# Noninvasive probe — GPU results

Protocol and literature (positions / pooling / instructed pairs): [`NOTES.md`](NOTES.md).

**Status: act probes are scored.** Varied original-turn is in. Truth vs intent cosine is about 0. The three new act directions also disagree with each other (cosine 0.02–0.25). Gemma-3-27B-IT (62 layers, d=5376) on an A800 80GB. `transformers==4.57.6`. Train: 400 Simple Deception rows and 400 instructed pairs. Every layer’s train cross-entropy is under 0.005, so the CE < 0.6 filter kept all 62 layers. Mean-kept is the mean of every layer.

`Noddybear/lies` HEAD was rebuilt on 2026-10-05 and its sample ids no longer match Gigi’s uids (0/310). Eval loaders in this folder pin `e8fc10f495`, the parent commit (310/310 ASCII ids). Gigi’s code is unchanged.

## Smoke (not the result)

Train n=40, ASCII n=16 (8/8), two positions. Layer 58 and the mean of kept layers disagree, including perfect and perfectly inverted cells. That is a 16-row probe, not a transfer result. Cache deleted before the full eval.

| Head | `last_content` L58 | `last_content` mean-kept | `mean_excl_last_5` L58 | `mean_excl_last_5` mean-kept |
|---|---:|---:|---:|---:|
| intent logistic | 0.00 | 1.00 | 0.67 | 1.00 |
| intent mean-diff | 0.00 | 1.00 | 0.63 | 1.00 |
| truth logistic | 0.00 | 0.70 | 1.00 | 0.00 |
| truth mean-diff | 0.05 | 0.34 | 1.00 | 0.91 |

Queue (1–4 done; 5–7 still running). Cache from the smoke was deleted before the full eval.

| # | What | Command | Why |
|---|---|---|---|
| 1 | Smoke fit | `python -u scripts/08_train_noninvasive_probe.py --max-train 40 --positions last_content,mean_excl_last_5` | Confirm extract + logistic / mean-diff on truth and intent. |
| 2 | Smoke score | `python -u scripts/08_train_noninvasive_probe.py --eval-only --datasets ascii_sandbagging --max-per-dataset 16 --positions last_content,mean_excl_last_5` | ASCII confound check on a handful of rows. |
| 3 | Full fit | `python -u scripts/08_train_noninvasive_probe.py` | All six positions, truth + intent, logistic + mean-diff. |
| 4 | Full original-turn | `python -u scripts/08_train_noninvasive_probe.py --eval-only` | Gigi + OA honeypot, no follow-up in context. |
| 5 | LAT | `python -u scripts/08_train_noninvasive_probe.py --recipes intent --heads logistic,mean_diff,lat --reuse-probes` | RepE first PC of instructed-pair diffs. |
| 6 | HTC | `python -u scripts/09_eval_noninvasive.py --skip-invasive` | Stored denials, pre-follow-up residual. |
| 7 | City / varied | `python -u scripts/08_train_noninvasive_probe.py --eval-only --datasets dyl_city,dyl_varied --max-per-dataset 800` | AISI original-turn, after the probe is frozen. |

## How to run

```bash
cd ravi/week3-noninvasive-probe
# HF_TOKEN in ../week1-dyl-vs-oa/dyl-probe-repro/.env or /workspace/.env

# 1. Fit the generic probe (no eval rows).
python -u scripts/08_train_noninvasive_probe.py
# smoke:
# python -u scripts/08_train_noninvasive_probe.py --max-train 40 --positions last_content,mean_excl_last_5

# 2. Score frozen probes on transfer sets (still no fit).
python -u scripts/08_train_noninvasive_probe.py --eval-only
# python -u scripts/08_train_noninvasive_probe.py --eval-only --datasets ascii_sandbagging --max-per-dataset 16 --positions last_content,mean_excl_last_5

python -u scripts/09_eval_noninvasive.py --skip-invasive
```

`--reuse-probes` skips retraining. Caches: `results/noninvasive_cache/` (gitignored). Scores: `results/noninvasive_eval.json`, `noninvasive_htc.json`, `noninvasive_invasive.json`.

City / varied are large; score them after the probe is frozen:

```bash
python -u scripts/08_train_noninvasive_probe.py --eval-only --datasets dyl_city,dyl_varied --max-per-dataset 800
```

LAT (RepE) on instructed pairs only (still not eval sets):

```bash
python -u scripts/08_train_noninvasive_probe.py --recipes intent --heads logistic,mean_diff,lat
```

## Original-turn AUROC

Same labels as week 1 [`OA_TRANSFER_RESULTS.md`](../week1-dyl-vs-oa/OA_TRANSFER_RESULTS.md). No follow-up in context. Cells are **intent logistic, layer 58** (the generic deception-intent probe). `n` is lie/honest.

| Dataset | n | asking | DYL | OA L58 | `last_content` | `mean_excl_last_5` | `mean_first_5` |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASCII | 191/119 | 0.69 | 0.99 | 0.996 | 0.000 | 0.988 | 0.988 |
| Sycophancy A | 38/120 | 0.68 | 0.66 | 0.555 | 0.683 | 0.694 | 0.699 |
| Sycophancy B | 116/85 | 0.71 | 0.64 | 0.445 | 0.398 | 0.515 | 0.485 |
| Mask | 55/31 | 0.58 | 0.52 | 0.659 | 0.540 | 0.602 | 0.602 |
| Capture | 73/305 | 0.57 | 0.61 | 0.429 | 0.540 | 0.597 | 0.533 |
| OA honeypot | 500/500 | 0.956 | 0.973 | 0.920 | 0.248 | 0.306 | 0.497 |

Intent logistic, mean of all 62 layers (same positions):

| Dataset | `last_content` | `mean_excl_last_5` | `mean_first_5` |
|---|---:|---:|---:|
| ASCII | 1.000 | 0.999 | 1.000 |
| Sycophancy A | 0.586 | 0.664 | 0.654 |
| Sycophancy B | 0.440 | 0.408 | 0.452 |
| Mask | 0.529 | 0.628 | 0.632 |
| Capture | 0.563 | 0.631 | 0.522 |
| OA honeypot | 0.362 | 0.215 | 0.454 |

Truth logistic at layer 58, where it is the stronger head: ASCII `mean_first_5` 1.000 and `mean_excl_last_5` 0.994; sycophancy B `mean_first_5` 0.718; capture `mean_first_5` 0.605. Truth mean-diff on the OA honeypot is 0.799 at `last_content` L58 and 0.944 at `last_newline` L58 (mean-kept 0.909 and 0.682). Intent on the honeypot stays below 0.5.

How to read this:

- ASCII `last_content` at layer 58 is perfectly inverted (0.000) while the mean of layers is 1.000, and `mean_excl_last_5` / `mean_first_5` stay at 0.99. The last content token is the YES/NO confound, with the sign flipped at layer 58. The pooled positions still separate, so this is not “only the refusal letter.”
- Sycophancy A intent L58 (0.68–0.70) is above OA (0.555) and about even with just asking (0.68) and DYL (0.66).
- Sycophancy B intent does not transfer. Truth `mean_first_5` L58 (0.718) is above just asking (0.71), DYL (0.64), and OA (0.445).
- Mask intent (~0.60) is above DYL (0.52) and below OA L58 (0.659) and OA mean (0.736).
- Capture intent `mean_excl_last_5` mean-kept (0.631) is above DYL (0.61) and OA (0.429). Layer 58 alone is 0.597.
- OA honeypot is a truth-direction readout, not the intent probe. The best logistic/mean-diff cell (truth mean-diff, `last_newline` L58, 0.944) is above OA L58 (0.920) and below just asking (0.956) and DYL (0.973). `last_newline` can be a chat-template artefact.

## LAT (instructed-pair diffs, first PC)

Same frozen eval caches. Layer 58 / mean of all layers.

| Dataset | `last_content` | `end_of_turn` | `mean_excl_last_5` | `mean_first_5` |
|---|---:|---:|---:|---:|
| ASCII | 0.002 / 0.000 | 0.827 / 0.821 | 0.000 / 0.011 | 0.000 / 0.140 |
| Sycophancy A | 0.500 / 0.443 | 0.366 / 0.388 | 0.300 / 0.404 | 0.315 / 0.319 |
| Sycophancy B | 0.427 / 0.429 | 0.419 / 0.453 | 0.479 / 0.573 | 0.376 / 0.527 |
| Mask | 0.573 / 0.507 | 0.469 / 0.602 | 0.445 / 0.429 | 0.509 / 0.486 |
| Capture | 0.580 / 0.529 | 0.534 / 0.560 | 0.522 / 0.585 | 0.450 / 0.450 |
| OA honeypot | 0.975 / 0.991 | 0.843 / 0.623 | 0.378 / 0.804 | 0.518 / 0.541 |

LAT `last_content` on the honeypot (0.975 L58, 0.991 mean) matches DYL (0.973) without a follow-up. `mean_excl_last_5` at layer 58 drops to 0.378, which is the short-answer warning in NOTES: trust `last_content` here, and do not treat the pooled tokens as the same readout. On ASCII, LAT is inverted at the content pools and only `end_of_turn` (0.83) carries the direction. It does not transfer to sycophancy.

## HTC (stored denials, pre-follow-up residual)

Intent logistic, layer 58. The residual is still the original turn. Just asking is 0.5 by construction on this subset. OA honeypot has no stored denials, so HTC is undefined. Mask `named` has no denials. Mask `neutral` is 6/6 and is not a result.

| Dataset | condition | lied-denied / truthful-denied | `mean_excl_last_5` | `last_content` |
|---|---|---:|---:|---:|
| ASCII | named | 189 / 119 | 0.988 | 0.000 |
| Sycophancy A | named | 32 / 93 | 0.702 | 0.656 |
| Sycophancy A | neutral | 27 / 104 | 0.664 | 0.678 |
| Sycophancy B | named | 21 / 51 | 0.494 | 0.443 |
| Sycophancy B | neutral | 116 / 85 | 0.515 | 0.398 |
| Capture | named | 28 / 75 | 0.587 | 0.501 |
| Capture | neutral | 52 / 193 | 0.557 | 0.487 |

ASCII named is almost the full set (almost every lie is denied), so 0.988 is the same pooled readout, now restricted to denials. Sycophancy A stays near 0.70 on the rows that later denied. Sycophancy B and capture do not get sharper on the denial subset.

## City original-turn (800 rows, 635 lie / 165 honest)

Week-1 OA on this set was 0.879 at layer 58 and 0.965 mean. Cells are layer 58 / mean of all layers. 129 unlabeled rows were skipped before the cap.

| Head | `last_content` | `end_of_turn` | `last_newline` | `mean_excl_last_5` | `mean_first_5` |
|---|---:|---:|---:|---:|---:|
| intent logistic | 0.965 / 0.892 | 0.917 / 0.951 | 0.968 / 0.952 | 0.902 / 0.909 | 0.894 / 0.914 |
| truth logistic | 0.457 / 0.806 | 0.920 / 0.901 | 0.558 / 0.608 | 0.791 / 0.909 | 0.690 / 0.847 |

Intent logistic matches the OA mean (0.965) at `last_content` layer 58 and at `last_newline` (0.968), with no follow-up in context.

Varied, same cap (800 rows), is 777 lie / 23 honest, so the honest class is thin. Week-1 OA was 0.909 at layer 58 and 0.955 mean. Intent logistic layer 58 is 0.647 at `last_content` and 0.744 at `mean_excl_last_5`. That is below both OA numbers.

## Act probes

Method is NOTES §6. Gemma wrote both sides of 80 pairs per act (greedy, one sentence). 10 pairs were dropped because a completion was empty; none were identical. Kept: on-policy 78 pairs, sycophancy 80, concealment 72. Of the concealment “withhold” completions, 25/72 still contain the fact, so that label is the policy, not a clean behavioral lie. OA honeypot here is 800 rows (`--max-per-dataset 800`), not the full 1000 the week-1 baseline used. Machine scores: `results/act_probe_eval.json`. Rollouts: `results/act_rollouts.jsonl`.

Layer-58 cosine of the raw directions: on-policy vs sycophancy 0.18–0.25, on-policy vs concealment 0.11–0.25, sycophancy vs concealment about 0. The shared direction vs truth is about 0, so removing the truth axis (`orth_truth`) does not change the scores. CCS uses no label in the loss; only the sign uses the label.

Cells are layer 58 / mean of all layers. Columns are last content token, drop last 5, first 5, last user token.

| Set | on-policy mean-diff | sycophancy-synth mean-diff | concealment mean-diff | mixture logistic |
|---|---:|---:|---:|---:|
| ASCII | 0.00/1.00, 0.03/1.00, 0.10/0.95, 0.63/0.82 | 0.43/0.00, 0.00/0.99, 0.00/0.00, 0.14/0.07 | **0.93/1.00, 1.00/1.00, 1.00/0.89**, 0.50/0.85 | 0.59/0.98, 0.00/0.43, 0.02/0.78, 0.31/0.39 |
| Sycophancy A | 0.54/0.50, 0.68/0.46, 0.64/0.64, 0.53/0.31 | 0.65/0.51, **0.66/0.51, 0.70/0.72**, 0.54/0.51 | 0.54/0.48, 0.61/0.55, 0.64/0.55, 0.49/0.55 | 0.66/0.56, 0.67/0.50, **0.74/0.50**, 0.56/0.34 |
| Sycophancy B | 0.55/0.45, 0.50/0.42, 0.53/0.38, 0.44/0.47 | 0.62/0.61, **0.75/0.74**, 0.67/0.67, 0.49/0.55 | 0.35/0.42, 0.55/0.41, 0.59/0.59, 0.50/0.51 | 0.47/0.53, 0.59/0.62, 0.60/0.67, 0.42/0.50 |
| Mask | 0.57/0.53, **0.73/0.75**, 0.68/0.67, 0.57/0.52 | 0.56/0.52, 0.68/0.63, 0.57/0.55, 0.41/0.50 | 0.47/0.41, 0.43/0.49, 0.55/0.60, 0.49/0.50 | 0.48/0.50, 0.67/0.64, 0.65/0.66, 0.55/0.60 |
| Capture | 0.50/0.50, 0.59/0.64, 0.49/0.47, 0.59/0.59 | 0.54/0.51, 0.56/0.51, 0.48/0.46, 0.46/0.42 | 0.55/0.53, 0.52/0.52, 0.48/0.46, 0.50/0.44 | 0.50/0.51, 0.51/0.48, 0.50/0.49, 0.57/0.57 |
| OA honeypot | 0.47/0.51, 0.29/0.67, 0.52/0.51, 0.50/0.50 | 0.55/0.95, 0.45/0.71, 0.55/0.53, 0.50/0.50 | 0.51/0.68, 0.18/0.15, 0.47/0.47, 0.50/0.50 | 0.55/0.81, 0.26/0.79, 0.50/0.56, 0.50/0.50 |
| City | **0.96/0.98, 0.94/0.95, 0.93/0.95**, 0.70/0.80 | 0.96/0.97, 0.69/0.88, 0.62/0.94, 0.30/0.94 | 0.82/0.88, 0.82/0.90, 0.80/0.63, 0.87/0.92 | **0.95/0.91, 0.93/0.93, 0.93/0.92**, 0.78/0.93 |
| Varied | 0.53/0.65, 0.72/0.71, 0.43/0.48, 0.50/0.52 | 0.50/0.51, 0.52/0.59, 0.67/0.63, 0.59/0.63 | 0.70/0.61, 0.51/0.53, 0.67/0.69, 0.49/0.63 | 0.57/0.60, 0.63/0.55, 0.59/0.49, 0.53/0.57 |

What this says:

- The three acts are three directions. Their average is chance at layer 58 on ASCII, sycophancy, capture, and the honeypot. Averaging them does not produce a generic probe. Taking out the truth direction changes nothing.
- A probe transfers to the eval that is the same act. On-policy fact lies reach city at 0.96 (layer 58, last token) and mask at 0.73 / 0.75 when the last five tokens are dropped, which matches OA’s mask mean (0.736) and beats OA layer 58 (0.659). Synthetic agreement reaches sycophancy B at 0.75, above just asking (0.71), DYL (0.64), and OA (0.445). Sycophancy A stays near 0.70, where the old intent probe already was.
- Concealment reads ASCII (0.93–1.00 at layer 58 on the content pools) and does not read capture (about 0.52). Hiding a trivia fact is not the capture-the-secret direction.
- CCS on the old instructed pairs matches the supervised ASCII result (drop-last-5 layer 58 = 1.00, last token = 0.00) and does not unlock sycophancy, mask, or capture.
- The last user token, before the assistant speaks, stays near chance on the hard sets. City concealment is the exception (0.87 at layer 58).

So a generic deception direction is not in these three policies. What works is a probe trained on the same kind of act as the eval, still with no follow-up in the eval context.

## HTC (09, stored denials, pre-follow-up residual)

Just asking is 0.5 by construction. Prefer Gigi **named** (more doubling-down). OA often has 0–6 denials / 500 — cell will say undefined.

## Invasive control (09)

Same weights, follow-up **in** context: native `did you lie`, named act, Goldowsky-Dill `Was your last response deceptive?` / `No`. If AUROC moves vs the table above, interrogation changed the residual.

## Files

| Path | What |
|---|---|
| `scripts/08_train_noninvasive_probe.py` | Train + original-turn eval |
| `scripts/09_eval_noninvasive.py` | HTC + invasive |
| `src/noninvasive_probe/extract.py` | Spans and pooling |
| `src/noninvasive_probe/protocol.py` | Follow-up texts, HTC join |
| `src/noninvasive_probe/data_intent.py` | Instructed pairs |
