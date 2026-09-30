# Noninvasive probe — GPU results

Protocol and literature (positions / pooling / instructed pairs): [`NOTES.md`](NOTES.md).

**Status: code ready; GPU numbers not in git yet.** Run on the Vast box (Gemma-3-27B-IT, transformers &lt; 5). Train is Simple Deception / instructed pairs only. Gigi / OA / AISI are eval, never fitted.

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

## Original-turn AUROC (fill after 08)

Same labels as week 1 [`OA_TRANSFER_RESULTS.md`](../week1-dyl-vs-oa/OA_TRANSFER_RESULTS.md). Probe has **no** follow-up in context.

| Dataset | asking | DYL | OA L58 | new `last_content` L58 | new `mean_excl_last_5` L58 | new `mean_first_5` L58 |
|---|---:|---:|---:|---:|---:|---:|
| ASCII | 0.69 | 0.99 | 0.996 | | | |
| Sycophancy A | 0.68 | 0.66 | 0.555 | | | |
| Sycophancy B | 0.71 | 0.64 | 0.445 | | | |
| Mask | 0.58 | 0.52 | 0.659 | | | |
| Capture | 0.57 | 0.61 | 0.429 | | | |
| OA honeypot | 0.956 | 0.973 | 0.920 | | | |

ASCII: if `last_content` ≈ 1.0 and `mean_excl_last_5` ≈ 0.5, that is the YES/NO confound, not a win.

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
