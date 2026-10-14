# Week 5 — one probe per flavor, read at each token position

This folder is week 5. The code it runs lives in [`../week3-noninvasive-probe`](../week3-noninvasive-probe), because the loader, the layer-58 logistic, and the position extractors are already there. Week 6 ([`../week6-early-detection`](../week6-early-detection)) is a different question: how many tokens before the bad action. Do not start that here.

John’s task from the Monday meeting: take the non-invasive probes (no “did you lie?”), train them the way Tianq trained his (one deception flavor at a time), test on sandbagging and the other flavors, and measure how AUROC changes with token position.

The generic probe stays. `scripts/08` and `scripts/10` are week 3. Do not delete them and do not train this week’s probes on instructed pairs. A probe trained on ASCII is supposed to see ASCII labels. That is the difference.

Do not edit `tianq/` or `gigi/`. Do not reproduce Anthropic monitors (Gigi) or the leave-one-out DYL split (Tianq).

## Where we are

| Step | What | State |
| --- | --- | --- |
| 1 | Script: load, cache, fit, print one table per position | Done. `scripts/11_flavor_matrix.py` |
| 2 | Laptop tests, no Gemma | Done. 21 tests passed |
| 3 | Counts on the pinned Noddybear revision | Done on this machine. Table below |
| 4 | Smoke extract of 8 rows, then delete the cache | Not started. Needs the GPU box |
| 5 | Full extract, all 11 flavors, nine positions | Not started |
| 6 | Fit the matrices on CPU | Not started |
| 7 | Write `FLAVOR_MATRIX.md` from the JSON | Not started |
| 8 | Monitor-fail slice, only if Gigi’s labels exist | Waiting on her file |

Steps 4 and 5 are the first commands that load Gemma. Do not run them until step 4’s cache has been deleted.

## Locked choices

Change these only if a step below fails.

| Choice | Value | Why |
| --- | --- | --- |
| Model | `google/gemma-3-27b-it` | Same model as Tianq and week 3 |
| Layer | 58 only (`hidden_states[59]`) | Matches his matrix. The new axis is token position, not layer |
| Label | Gigi’s label on the reply itself | Lied vs truthful. No follow-up in the forward pass |
| Classifier | His logistic: standardize, mean log-loss + `0.01 * (\|\|w\|\|² + b²)`, L-BFGS to convergence | `tianq_read.fit_aisi_logistic`. One solver for every cell, so position trends are not a solver change |
| Diagonal | 5-fold stratified CV, `random_state=0` | His in-category number is out-of-fold. Training and testing on the same rows would inflate it |
| Off-diagonal | Fit on every labelled row of the train flavor, score every labelled row of the test flavor | His transfer cells |
| Train minimum | 10 lies and 10 truthful replies | His `MIN_ROWS_PER_CLASS` |
| Headline position | `mean_answer` | Week 3’s best pool |
| Comparable position | `reply_mean` | His best no-follow-up read (in-category 0.80, transfer 0.50). Not the same tokenization as `mean_answer` |
| Other positions | `last_content`, `end_of_turn`, `last_newline`, `mean_excl_last_5`, `mean_first_5`, plus his `prompt_last` and `reply_last` | The curve he asked for |

His published no-follow-up means, for the report only:

| Position | In-category | Transfer |
| --- | ---: | ---: |
| `prompt_last` | 0.71 | 0.50 |
| `reply_last` | 0.73 | 0.50 |
| `reply_mean` | 0.80 | 0.50 |

`mean_answer` has no published twin. Put it in the curve. Put `reply_mean` next to 0.80 and 0.50.

## Flavors

Confirm with `python -u scripts/11_flavor_matrix.py --counts-only` from the week 3 folder. Stop if a count changed.

Train on these ten. Each has at least 10 of both classes:

| Flavor | Rows | Lies | Truthful |
| --- | ---: | ---: | ---: |
| ascii_sandbagging | 310 | 191 | 119 |
| sycophancy_answer | 158 | 38 | 120 |
| sycophancy_feedback | 201 | 116 | 85 |
| mask_factual | 86 | 55 | 31 |
| mask_known_facts | 71 | 43 | 28 |
| mask_statistics | 47 | 22 | 25 |
| mask_continuations | 87 | 73 | 14 |
| mask_disinformation | 69 | 58 | 11 |
| capture_the_secret | 378 | 73 | 305 |
| alibi | 1276 | 784 | 492 |

Test on those ten plus `unanswerable` (93 rows, 86 lies, 7 truthful). Do not train on unanswerable. Mark its column `(few)` and leave it out of the transfer mean.

ASCII is required on both axes. Sandbagging is the flavor John named. Its answers are often one letter, so `last_content`, `reply_last`, and `reply_mean` can be the same vector. A perfect ASCII score is reported, and labelled as the letter confound, not as a win.

## Step 1 — the script

`ravi/week3-noninvasive-probe/scripts/11_flavor_matrix.py`. It does four things.

1. Load the 11 flavors with `load_category`. Conversations end on the reply. No follow-up is appended.
2. Extract once per flavor and cache. Two families, both layer 58:
   - Closed chat, the six pools in `ORIGINAL_TURN_POSITIONS`.
   - Tianq’s three reads (`prompt_last`, `reply_last`, `reply_mean`).
3. Fit on CPU from the cache. For each position and each train flavor: 5-fold stratified out-of-fold scores on that flavor (the diagonal), then one fit on all of its rows scored on each other flavor.
4. Write `results/flavor_matrix.json` and print one markdown table per position.

Cache paths, so a killed job resumes:

```text
results/flavor_cache/{flavor}_closed_L{max_length}.npz
results/flavor_cache/{flavor}_tianq_L{max_length}.npz
```

Store `uids`, `y`, `max_length`, and one array per position. Reuse a file only when the uid list and `max_length` match. Do not reuse `results/noninvasive_cache/` or `results/tianq_compare_cache/` from a `--max-per-dataset` smoke.

Exactly one of `--counts-only`, `--extract-only`, `--fit-only`. Also: `--flavors`, `--positions`, `--max-per-dataset`, `--max-length` (default 4096), `--batch-size` (default 2), `--refresh`.

`--fit-only` refuses to run if any requested flavor’s cache is missing, was built at a different length, or was built with a shorter uid list.

Fold rule, matching Tianq: `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`. A flavor is `not_trained` below 10 per class. A test cell with fewer than 10 of either class is stored and marked `few`, and excluded from `mean_transfer`. `mean_in_category` averages the trained diagonals, excluding any diagonal that is `few`.

## Step 2 — laptop tests

`tests/test_flavor_matrix.py`. No Gemma.

- Two well-separated clusters, 5-fold: out-of-fold AUROC above 0.9, and each fit’s train indices are disjoint from that fold’s test indices.
- A flavor with 9 lies is `not_trained`. A test set with 7 truthful rows is `few` and is absent from `mean_transfer`.
- The instructed-pairs source is never passed to the fitter. This script’s only labels are the flavor labels.

`python -m pytest tests/ -q` in the week 3 folder stays green, including the week 3 tests. Last run: 21 passed.

## Step 3 — confirm the machine, then the counts

```bash
cd ravi/week3-noninvasive-probe
nvidia-smi
python -m pytest tests/ -q
python -u scripts/11_flavor_matrix.py --counts-only
```

`transformers` stays below 5. `HF_TOKEN` is in `../week1-dyl-vs-oa/dyl-probe-repro/.env` or `/workspace/.env`. `gigi/` is in the checkout. Stop if the counts differ from the table above. The laptop count already matched, on the pinned Noddybear revision `e8fc10f495aabc2d2d95f1e947b10bbcced444d5`.

## Step 4 — smoke extract, then delete it

```bash
python -u scripts/11_flavor_matrix.py --extract-only \
  --flavors alibi,mask_known_facts --max-per-dataset 8 \
  --positions mean_answer,reply_mean
rm -rf results/flavor_cache
```

This only checks that both tokenizers run and a short alibi chat and a MASK chat return finite vectors. The 8-row files must not exist before step 5. The cache name does not include the row cap, so a leftover smoke file would be fitted as if it were the full flavor.

## Step 5 — full extract

Layer 58 only. All 11 flavors. All nine positions. Alibi is 1,276 rows and is most of the time.

```bash
python -u scripts/11_flavor_matrix.py --extract-only
```

If a reply exceeds 4096 tokens, stop, delete `results/flavor_cache`, and rerun with `--max-length 8192`. If the process is killed for memory, rerun the same command with `--batch-size 1`. Cached flavors are skipped, so this continues rather than starting over.

Before fitting, check one cached file: `n_lost_prompt` is 0 for `prompt_last`. A lost prompt token means the chat was truncated and `prompt_last` is not his read.

Do not start `scripts/08`, `scripts/09`, or `scripts/10` during this job.

## Step 6 — fit the matrices on CPU

The GPU can sit idle. Fitting is L-BFGS on layer 58.

```bash
python -u scripts/11_flavor_matrix.py --fit-only
```

Expect about 10 diagonals × 5 folds, plus 10 × 10 transfer fits, at each of 9 positions. The script writes after every position so a crash keeps the finished tables.

`results/flavor_matrix.json` holds, per position:

- `mean_in_category`, `mean_transfer`
- `transfer_cells`, `transfer_cells_at_least_0_7`
- one cell per train/test pair: `auroc`, `n_lie`, `n_honest`, `few`, `kind` (`in_category` or `transfer` or `not_trained`)

## Step 7 — write `FLAVOR_MATRIX.md`

Fill this from the JSON, in this folder. Round cells to two decimals, summary lines to three. Cell text matches his: `0.62 (43/28)`, with ` (few)` when marked. Bold only the diagonal.

One section per position, in this order: `mean_answer`, `reply_mean`, `prompt_last`, `reply_last`, `last_content`, `end_of_turn`, `last_newline`, `mean_excl_last_5`, `mean_first_5`.

Each section starts with:

```text
NO FOLLOW-UP, one probe per flavor, read at {position}: in-category {mean_in:.3f}, transfer {mean_tr:.3f} ({n70} of {n} transfer cells at 0.7 or more).
```

Under `reply_mean`, `prompt_last`, and `reply_last`, add his means on the next line: in-category 0.80 / 0.71 / 0.73, transfer 0.50.

Then the 10×11 table. Row label is the train flavor. Column order is the flavor table above, with unanswerable last.

After all nine sections, add one curve table:

| Position | In-category | Transfer | Cells ≥ 0.7 |
| --- | ---: | ---: | ---: |
| mean_answer |  |  |  |
| reply_mean |  |  |  |
| prompt_last |  |  |  |
| reply_last |  |  |  |
| last_content |  |  |  |
| end_of_turn |  |  |  |
| last_newline |  |  |  |
| mean_excl_last_5 |  |  |  |
| mean_first_5 |  |  |  |

Close with four sentences, only what the numbers support:

1. Does `reply_mean` transfer stay near 0.50, or clear it? Near 0.50 means mean-pooling does not fix flavor-specificity. His category probes already showed that. Clearing it means the gain is the pool, not the category.
2. Is the best position the same on every diagonal, or does each flavor peak at a different token? That is the question John asked. Answer it from the diagonals, not from the transfer mean.
3. ASCII: if the last-token reads are near 1 and `mean_excl_last_5` drops, say it is the letter confound.
4. Mask-known-facts ↔ alibi: if those two off-diagonal cells are the only ones clearly above 0.5, say so. Tianq already saw that pair transfer under a follow-up. This says whether it survives with no follow-up and with these pools.

Do not call the result a general lie detector. Do not compare `mean_answer` to 0.80 as if they were the same read.

## Step 8 — Thursday, only if Gigi’s monitor labels exist

John wants white-box probes scored on rows a black-box monitor gets wrong. That waits on her labels. If they are not in the repo by Thursday, skip this step and say it is waiting.

If a file of monitor-fail uids exists:

1. Do not refit.
2. Take the probes already fit in step 6, at `mean_answer` and `reply_mean`.
3. Score each probe on the failed rows of flavors it was not trained on.
4. Add a short section to `FLAVOR_MATRIX.md`: how many failed rows each flavor has, and the AUROC on those rows. If a flavor has fewer than 10 failures of either class, mark it and do not average it.
5. If the monitor is already near-perfect and the failed set is tiny, write that the black-box benchmark is saturated. Do not invent a white-box win from a handful of rows.

## Done

`FLAVOR_MATRIX.md` is in this folder. The curve table is filled. `reply_mean` has been set next to 0.80 and 0.50. The generic-probe scripts are still in the week 3 tree and were not the source of these numbers.

## Not this experiment

- Leave-one-out and the 3-vs-3 split. That is Tianq’s Anthropic mirror.
- Training or fine-tuning a monitor.
- LAT, all-layer sweeps, or another search over heads.
- `scripts/09` invasive follow-ups.
- Fitting on instructed pairs and calling that cell in-category.
- The token-horizon curve in week 6. That defines a bad-action token and a lead time. This experiment reads finished replies.
