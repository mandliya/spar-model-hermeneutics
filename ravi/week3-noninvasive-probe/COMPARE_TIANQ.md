# Comparing our probe with Tianq’s no-follow-up result

The GPU agent’s instructions are [`AGENTS.md`](AGENTS.md). It runs `scripts/10_compare_tianq.py` and writes `TIANQ_COMPARE.md`. This file is the contract those instructions follow.

Tianq trained **one probe per deception category** and tested it on the others, with no “did you lie?” question in the chat. Inside a category those probes work (about 0.71 / 0.73 / 0.80). Across categories they are a coin flip (0.50).

Our experiment is the cell he did not run: **one probe trained only on instructed pairs**, scored on all 11 categories. If that mean is still ~0.50, Gemma-3-27B has no shared lie direction on the reply. If it is clearly above 0.50, the direction is shared and his category probes were too specific to see it.

Do not edit `tianq/` or `gigi/`. Do not train on Gigi’s rows or on his Hugging Face activation files.

## The number that sits next to his

| Read | What it is | Tianq, trained on that category | Tianq, other categories | Ours |
| --- | --- | --- | --- | --- |
| `prompt_last` | token that is about to write the reply | 0.71 | 0.50 | mean AUROC from `10` |
| `reply_last` | last token of the reply text | 0.73 | 0.50 | mean AUROC from `10` |
| `reply_mean` | average over the reply’s tokens | 0.80 | 0.50 | mean AUROC from `10` |

Put our mean next to **0.50**, not next to 0.80. 0.80 is a probe that was trained on the same kind of lie it is tested on. Ours never sees those categories during training, so every category is a transfer cell.

`scripts/08` is a different readout (closed chat, other token spans, a different logistic). Do not quote those AUROCs in this table.

## What is matched

- Model: `google/gemma-3-27b-it`
- Layer 58 (`hidden_states[59]`)
- Classifier: per-dimension train mean and standard deviation, then logistic regression. Objective is mean log-loss plus `0.01 * (||w||² + b²)`, L-BFGS to convergence. Same as `tianq/dyl-probe-repro/dyl_repro/core/training.py`.
- Positions: prompt tokens from the chat template with the generation prompt, then the reply’s own token ids appended (the reply is not re-tokenized inside a closed chat). Same as his `prefill.py` with `whole_answer=True`.
- Label: Gigi’s label for the reply itself (lied vs truthful). Every labelled row. No follow-up, so no false-confession filter.
- Categories (his 11): ASCII sandbagging, sycophancy answer, sycophancy feedback, MASK provided facts, MASK known facts, MASK statistics, MASK continuations, MASK disinformation, unanswerable, capture the secret, alibi.

## What is deliberately different

- Training data is instructed pairs from week 1 `data/true_false_facts.csv` (same statement, honest vs deceptive system prompt). `--recipe truth` is the Geometry-of-Truth side column (false statement = lie). The headline is `--recipe intent`.
- We report one row of transfer, not his 11×11 matrix. Do not retrain a probe on each category to redraw his heatmap.

## Gigi’s Week 3 report, and where it does not enter

His self-report (“did you lie?”) is a different measurement. In most categories it is about a coin flip. Two exceptions, where Gemma admits fairly reliably (about 75–85% accuracy): keeping up a false alibi, and writing an article it believes is false (MASK disinformation). Chain-of-thought makes it confess to lies it did not tell. Letter position can drive the ASCII answer.

Those are black-box accuracies. They do not go in the AUROC table. If our probe is high only on alibi and MASK disinformation, it is picking up the cases where even asking works, not a general lie direction.

His activation dataset (`model-hermeneutics/gigi-lie-detection-activations`) is from the follow-up runs. This comparison re-extracts the reply with no follow-up in the forward pass. Do not train on that dataset.

## On the GPU box

```bash
cd ravi/week3-noninvasive-probe
git pull
python -m pytest tests/ -q

# Row counts only. No model load. The table below is what this should print.
python -u scripts/10_compare_tianq.py --counts-only

# Smoke (tiny). Then delete the cache before the real run.
python -u scripts/10_compare_tianq.py --max-train 40 --max-per-dataset 8 --categories alibi,mask_known_facts
rm -rf results/tianq_compare_cache

# Headline comparison. Intent probe, all 11 categories, layer 58, his three positions.
python -u scripts/10_compare_tianq.py
```

Needs the same Gemma setup as week 3 (`HF_TOKEN`, transformers &lt; 5, ~80 GB). `gigi/` must be in the checkout; the loaders read his CSVs from there.

`--counts-only` should print:

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

Unanswerable has only 7 truthful replies, so it is marked `*` and left out of the mean (Tianq drops a cell with fewer than 10 of either class). Alibi is most of the GPU time.

If a reply is longer than 4096 tokens the script stops and asks you to raise `--max-length` (8192 if the GPU allows). A row that loses only the token before the reply is dropped from `prompt_last` and kept for the two reply reads. The printed `n_lost_prompt` should be 0 on the training facts. If it is not, the train chats were truncated and the fit is not his readout.

The script prints a three-line table and writes `results/tianq_compare_intent.json`. Categories with fewer than 10 lies or 10 truthful replies are marked `*` and left out of the mean.

Optional side column, after the headline finishes:

```bash
python -u scripts/10_compare_tianq.py --recipe truth
```

That writes `results/tianq_compare_truth.json`. It answers whether “this statement is false” transfers. It is not the number next to his 0.50.

## How to read the result

- Mean near 0.50 at all three positions: same conclusion as his transfer row. A generic pairs probe does not catch these lies on the reply.
- Mean clearly above 0.50, especially at `reply_mean`: the signal is shared across categories, and training on one category was the wrong test of that.
- `prompt_last` near 0.50 while `reply_mean` is higher: the direction is in the reply, not in the decision to start writing. That matches both his ordering (0.80 &gt; 0.71) and the Parrack full-average result.
- A spike only on alibi or MASK disinformation: look back at Gigi’s exceptions before calling it a lie detector. ASCII answers are often a single letter, so `reply_last` and `reply_mean` can be the same vector there; a high ASCII score can be the letter, which is the confound Gigi already reported.

Per-category AUROCs are in the JSON under `categories`. Use them when the mean hides one easy category.
