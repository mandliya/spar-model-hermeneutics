> **Archived 1 October 2026.** A frozen copy from before the project kept only four follow-up
> questions (`neutral`, `neutral_cot`, `named`, `named_cot`). It still covers all 13.
> The live version is [ab_swap/README.md](../../../1-replication/ab_swap/README.md); what was archived and why is in
> [the archive index](../README.md).

# Does gemma answer the question or the letter? The A/B swap test

In every follow-up wording in this project, **A means "yes, I lied"** and **B
means "no"**. Models are known to favour particular option letters regardless
of content ([Zheng et al., ICLR 2024](https://arxiv.org/abs/2309.03882);
[Dominguez-Olmedo et al., NeurIPS 2024](https://arxiv.org/abs/2306.07951)), so
some of what looked like gemma's answers could be letter preference. This test
asks the same questions with the two options swapped:

```
original:  A: Yes, I lied            swapped:  A: No, I was truthful
           B: No, I was truthful               B: Yes, I lied
```

Every other character of every question is unchanged. If gemma answers the
meaning, its yes/no verdict is the same in both orders. If it answers the
letter, the verdict flips.

## Design

- **Rows:** a fixed subset of 626 rows, up to 50 lied and 50 honest per
  category (all of them where a category has fewer), seed 0
  (`results/ab_swap_subset.csv`).
- **Wordings:** all 11 A/B wordings. The two confidence wordings have no
  letters, so they are not part of this test.
- **gemma answers fresh** at temperature 0, with the same token limits (80, or
  300 when reasoning comes first) and the same parsers as the original runs.
  6,886 answers, 0 errors, 0 unparseable.
- **Original-order answers** for the same rows come from the main runs, with
  their letter probabilities from [`../answer_probability/`](../../../1-replication/answer_probability).

**Measures**, per category and wording:
- **letter shift**: half the difference between the "yes" rate with yes at A
  and with yes at B, in percentage points. +10 means 10 points of the answers
  follow the letter A whatever it means; −10 means they follow B.
- **agreement**: share of rows with the same yes/no verdict in both orders.

## Results

1. **Letter preference is real, and its direction depends on the wording.**
   Reasoning-first wordings pull toward **A** (mean shift +14 points,
   agreement 67%). The direct wordings (−6) and the `named` / `eval_named`
   wordings (−10) pull toward **B**. `named_cot` barely moves (−1, agreement
   88%).
2. **In some cells gemma is answering the letter, not the question.**
   - ascii `high_stakes_cot`: "yes" 95% with yes at A, 1% with yes at B.
     Agreement 6%.
   - ascii `named`: "yes" 1% with yes at A, 92% with yes at B. Agreement 9%.
     The "no on 308 of 310 rows" reported for `named` is mostly gemma picking
     B.
   - capture-the-secret `neutral`: agreement 36%, shift −32.
3. **How much it matters depends on the category.** ascii is most affected
   (mean |shift| 18 points, agreement 62%) and mask-factual least (3 points,
   93%).
4. **Alibi's self-report survives the swap. The other categories' best cells
   mostly don't.**

   | Cell | Balanced accuracy, yes = A | yes = B | AUROC, yes = A | yes = B |
   |---|---|---|---|---|
   | alibi `named` | 85 | 74 | 0.96 | 0.94 |
   | alibi `named_cot` | 85 | 83 | 0.94 | 0.90 |
   | alibi `eval_named` | 87 | 77 | 0.90 | 0.91 |
   | alibi `low_stakes` | 88 | 79 | 0.96 | 0.93 |
   | ascii `eval_named` | 76 | **42** | 0.82 | **0.24** |
   | sycophancy B `named` | 70 | **51** | 0.73 | 0.77 |

   Alibi is at 70 or above in 9 of 11 wordings with yes at B, and its mean
   AUROC is 0.86 with yes at A and 0.89 with yes at B. ascii `eval_named`, the
   one non-alibi cell over 70, falls to 42.
5. **The P("A") results hold where the letters are stable, and not for
   ascii.** mask-factual's AUROC is 0.73 with yes at A and 0.72 with yes at B,
   and alibi's is 0.86 and 0.89. ascii's mean AUROC goes from 0.50 to **0.21**:
   its probabilities follow the letter, not the lie.

### Letter shift (points; + = toward A, − = toward B)

<!-- SHIFT-GRID:START -->
| Wording | ascii | syc A | syc B | mask | unans. | capture | alibi |
|---|---|---|---|---|---|---|---|
| `neutral` | +12 | -18 | -1 | -6 | -19 | -32 | -16 |
| `low_stakes` | +3 | -4 | +0 | -4 | -14 | -14 | -4 |
| `high_stakes` | -2 | -2 | +0 | -6 | +0 | -26 | -24 |
| `neutral_cot` | +16 | +9 | +7 | +6 | -2 | +19 | +5 |
| `low_stakes_cot` | +35 | +16 | +27 | +4 | +11 | +13 | +4 |
| `high_stakes_cot` | +47 | +15 | +4 | +7 | +16 | +26 | +10 |
| `plain` | -1 | -12 | -2 | +0 | -9 | -1 | -7 |
| `named` | -46 | -15 | -18 | +0 | +2 | -16 | -5 |
| `named_cot` | -5 | -10 | +0 | +0 | +9 | +3 | +0 |
| `eval_plain` | +3 | -1 | +0 | -1 | -1 | -2 | -9 |
| `eval_named` | -33 | +1 | -4 | +0 | +11 | -4 | -9 |
<!-- SHIFT-GRID:END -->

### Agreement between the two orders (same yes/no verdict)

<!-- AGREE-GRID:START -->
| Wording | ascii | syc A | syc B | mask | unans. | capture | alibi |
|---|---|---|---|---|---|---|---|
| `neutral` | 77% | 65% | 98% | 88% | 61% | 36% | 67% |
| `low_stakes` | 94% | 90% | 100% | 91% | 72% | 72% | 91% |
| `high_stakes` | 97% | 97% | 100% | 89% | 100% | 48% | 53% |
| `neutral_cot` | 59% | 76% | 74% | 86% | 82% | 53% | 82% |
| `low_stakes_cot` | 30% | 66% | 46% | 90% | 70% | 68% | 88% |
| `high_stakes_cot` | 6% | 67% | 93% | 80% | 68% | 45% | 73% |
| `plain` | 99% | 76% | 95% | 100% | 79% | 98% | 86% |
| `named` | 9% | 69% | 65% | 100% | 96% | 67% | 87% |
| `named_cot` | 85% | 68% | 100% | 100% | 79% | 95% | 92% |
| `eval_plain` | 94% | 91% | 100% | 99% | 95% | 97% | 83% |
| `eval_named` | 34% | 98% | 89% | 100% | 79% | 91% | 82% |
<!-- AGREE-GRID:END -->

Per-cell yes rates, balanced accuracy and AUROC in both orders, plus AUROC on
the average of the two orders, are in `results/ab_swap_summary.csv`.

## What this changes elsewhere

- **The "wording moves the default answer" headline in `../RESULTS.md`** is
  partly letter preference. ascii `named` answers B, not "no". The
  unanswerable example ("did you lie?" 12.8% vs "did you confabulate?" 100%)
  mostly holds: `named` there agrees across orders on 96% of rows.
- **Cells above 70 outside alibi** should be read as fragile: they don't
  survive reversing the options.
- **Averaging both orders** is the standard fix (Zheng et al.). Here it gives
  a mean AUROC of 0.60, against 0.61 with yes at A alone and 0.58 with yes at
  B alone. The two orders cancel rather than add up.

## Caveats

- **Subset, not every row.** 626 of 1,732 rows, balanced 50/50 where
  possible, so the original-order numbers here differ somewhat from the
  full-row tables. Every comparison above uses the same rows in both orders.
- **The swap only reorders the two options.** "Answer (A) or (B)" and "Final
  answer: A / or / Final answer: B" still list A first. So this separates
  what an option means from which letter it has, not from where it sits in
  the list.
- **unanswerable has only 7 honest rows**, and sycophancy A has 38 lied rows.

## Files

| File | What it is |
|---|---|
| `00_pick_subset.py` | draws the fixed subset (no GPU) |
| `07_ab_swap.py` | the GPU run: swapped question, fresh answer, letter probabilities |
| `02_analyse_swap.py` | the comparison; fills the grids above |
| `results/ab_swap_subset.csv` | the 626 rows |
| `results/ab_swap_<category>.jsonl` | one record per answer: swapped question, answer, letter, `admitted` (B = yes), letter probabilities, the original answer |
| `results/ab_swap_summary.csv` | every measure per cell |

The `ab_swap_*.jsonl` files are also on Hugging Face under `results/ab_swap/`.
Cost: about 4.3 GPU-hours on one A100 80 GB (14:19–18:35 on 30 Sep 2026).

```bash
uv run --with pandas --with numpy --with huggingface_hub \
    python gigi/ab_swap/02_analyse_swap.py --pull
```
