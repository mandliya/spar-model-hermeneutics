# GPU agent — generic probe vs Tianq’s no-follow-up result

You are on an 80 GB GPU box (A100/H100). Work only in this directory:

```text
ravi/week3-noninvasive-probe
```

Your job is to score the **generic deception probe** the way Tianq scored his no-follow-up matrix, then write `TIANQ_COMPARE.md` in this folder. That file is the report Ravi will read. Fill it from `results/tianq_compare_intent.json`. Do not leave the numbers only in the terminal.

## The goal

Ravi is trying to build a **better generic deception probe**: one direction, trained on instructed pairs, that catches kinds of lying it was never trained on.

That probe is already in this folder. Do not remove it, rewrite it, or replace it with Tianq’s per-category probes.

- Train only on Simple Deception and instructed pairs (`src/noninvasive_probe/sets.py` `TRAIN_SOURCES`). `assert_generic_train` must keep refusing Gigi, OA, and AISI rows.
- `scripts/08_train_noninvasive_probe.py` is that trainer. Leave it in place.
- `scripts/10_compare_tianq.py` scores **that same probe** at Tianq’s positions with Tianq’s classifier. It is the run for this job. It must not fit on Gigi’s categories.

Tianq trained one probe per deception category and tested it on the others, with no “did you lie?” in the chat. Inside a category those probes work (about 0.71 / 0.73 / 0.80). On other categories they are chance (0.50). The open question is whether a probe that never saw those categories does better than 0.50. Matching his 0.80 is not the goal. 0.80 is a probe trained on the same kind of lie it is tested on.

## Do not

- Edit `tianq/` or `gigi/`. Import his loaders; do not change them.
- Train on Gigi’s rows, on `model-hermeneutics/gigi-lie-detection-activations`, or on one probe per category. Do not redraw his 11×11 heatmap.
- Run `scripts/08` or `scripts/09` for this report. Those use a different token span and a different logistic. Their AUROCs do not go in the comparison table.
- Quote our mean next to 0.80 as if we failed by missing it. The comparison number is his **transfer** mean, 0.50.
- Invent per-cell numbers for Tianq. His published figures are the three means below. Per-category cells of his matrix are not in this repo.
- Commit `.env`, tokens, or `results/tianq_compare_cache/`. Do not commit or push unless Ravi asks in this session.
- Destroy the Vast instance. GPU billing continues until it is destroyed from the SPAR team context.

## What is matched to Tianq

- Model: `google/gemma-3-27b-it`, layer 58 (`hidden_states[59]`).
- Classifier: standardize by train mean and population std, logistic, mean log-loss + `0.01 * (||w||² + b²)`, L-BFGS to convergence.
- Positions, from his `prefill.py` with the whole reply kept: `prompt_last` (token about to write the reply), `reply_last` (last token of the reply text, not `<end_of_turn>`), `reply_mean` (mean over reply tokens).
- Label: Gigi’s label for the reply itself. No follow-up in the forward pass.
- His 11 categories: ASCII sandbagging, sycophancy answer, sycophancy feedback, MASK provided facts, MASK known facts, MASK statistics, MASK continuations, MASK disinformation, unanswerable, capture the secret, alibi.

His no-follow-up summary (Slack, 2026-10-04), also stored in the JSON as `tianq_no_followup`:

| Position | In-category (trained on that category) | Transfer (other categories) |
| --- | ---: | ---: |
| prompt_last | 0.71 | 0.50 |
| reply_last | 0.73 | 0.50 |
| reply_mean | 0.80 | 0.50 |

## Setup

Need ~80 GB VRAM. Check `nvidia-smi` before loading Gemma. `transformers` must stay below 5. `HF_TOKEN` is in `../week1-dyl-vs-oa/dyl-probe-repro/.env` or `/workspace/.env`. `gigi/` must be in the checkout; the loaders read his CSVs from there.

```bash
cd ravi/week3-noninvasive-probe
python -m pytest tests/ -q
python -u scripts/10_compare_tianq.py --counts-only
```

Stop if the counts are not these. A mismatch means the row set is not his.

| Category | Rows | Lies | Truthful |
| --- | ---: | ---: | ---: |
| ascii_sandbagging | 310 | 191 | 119 |
| sycophancy_answer | 158 | 38 | 120 |
| sycophancy_feedback | 201 | 116 | 85 |
| mask_factual | 86 | 55 | 31 |
| mask_known_facts | 71 | 43 | 28 |
| mask_statistics | 47 | 22 | 25 |
| mask_continuations | 87 | 73 | 14 |
| mask_disinformation | 69 | 58 | 11 |
| unanswerable | 93 | 86 | 7 |
| capture_the_secret | 378 | 73 | 305 |
| alibi | 1276 | 784 | 492 |

Unanswerable has 7 truthful replies, so the script marks it `few` and leaves it out of the mean. Tianq does the same under 10 of either class.

## Run

Smoke, then delete the cache. A smoke cache reused on the full run would score 8 rows and look like a result.

```bash
python -u scripts/10_compare_tianq.py --max-train 40 --max-per-dataset 8 --categories alibi,mask_known_facts
rm -rf results/tianq_compare_cache
python -u scripts/10_compare_tianq.py
```

Headline recipe is `intent` (same statement, honest vs deceptive system prompt). That writes `results/tianq_compare_intent.json`.

If a reply is longer than 4096 tokens the script stops. Rerun the full command with `--max-length 8192`. Do not resume on a cache from the failed length. `n_lost_prompt` on the training facts must be 0. If it is not, the fit was truncated and is not his readout; raise `--max-length` and `--refresh`.

Alibi is 1,276 rows and most of the GPU time. If the process is killed for memory, rerun the full command with `--batch-size 1`. The cache skips categories already extracted.

After `TIANQ_COMPARE.md` exists, the optional side column is:

```bash
python -u scripts/10_compare_tianq.py --recipe truth
```

That asks whether “this statement is false” transfers. It is not the number next to his 0.50. Skip it if the intent run has not finished.

## Write `TIANQ_COMPARE.md`

Read `results/tianq_compare_intent.json`. Use `mean_auc` and each category’s `auc`, `n_deceptive`, `n_honest`, and `few`. Round AUROC to two decimals in the tables, three in the summary line. Copy Tianq’s summary voice. One section per position, in this order: `prompt_last`, `reply_last`, `reply_mean`.

For each position, start with a line in this shape (fill the braces from the JSON; count cells with `few` false and `auc` ≥ 0.7):

```text
NO FOLLOW-UP, generic probe trained on instructed pairs only, read at {position}: mean transfer {mean:.3f} over categories with at least 10 of each class ({n_at_least_0_7} of {n} at 0.7 or more). Tianq's category-trained probes at this position: in-category {his_in_category:.2f}, transfer 0.50.
```

Then a table. One row is our probe. Cells look like his: `0.62 (43/28)`. Append ` (few)` when `few` is true. Do not bold a cell; nothing here is in-category for us.

```text
| trained on \ tested on | ASCII sandbagging | Sycophancy A | Sycophancy B | MASK provided facts | MASK known facts | MASK statistics | MASK continuations | MASK disinformation | Unanswerable | Capture the secret | Alibi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Instructed pairs (ours) | 0.62 (191/119) | ... |
```

After the three sections, add:

1. A four-row summary table: position, our mean, Tianq transfer 0.50, Tianq in-category 0.71 / 0.73 / 0.80. State in one sentence that our mean is compared with 0.50.
2. A short reading, using only these rules:
   - Mean near 0.50 at all three positions: a generic pairs probe does not catch these lies on the reply. Same conclusion as his transfer row.
   - Mean clearly above 0.50, especially at `reply_mean`: the direction is shared across categories. Training on one category was the wrong test of that.
   - `prompt_last` near 0.50 while `reply_mean` is higher: the direction is in the reply, not in the decision to start writing.
   - A high score only on alibi or MASK disinformation: those are the two cases where Gigi found that just asking “did you lie?” already works (about 75–85% accuracy). That is not a general lie detector.
   - ASCII answers are often one letter, so `reply_last` and `reply_mean` can be the same vector there. A high ASCII score can be the letter, which Gigi already reported. Say so if ASCII is an outlier.
3. One line that the generic probe code is unchanged: training is still instructed pairs only; this report is a measurement of that probe.
4. If you ran `--recipe truth`, a separate short section. Do not mix those AUROCs into the intent table.

Gigi’s “did you lie?” accuracies are a different measurement. Do not put them in the AUROC table. Mention them only in the reading, and only for alibi and MASK disinformation.

Do not claim the probe is a lie detector. Report the means, the cells that carry them, and which of the four readings the numbers support.
