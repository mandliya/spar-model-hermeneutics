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
- **Questions:** the project's four follow-up questions (`neutral`,
  `neutral_cot`, `named`, `named_cot`): 2,504 answers.
- **gemma answers fresh** at temperature 0, with the same token limits (80, or
  300 when reasoning comes first) and the same parsers as the original runs.
  0 errors, 0 unparseable.
- **Original-order answers** for the same rows come from the main runs, with
  their letter probabilities from [`../answer_probability/`](../answer_probability/README.md).

**Measures**, per category and question:
- **letter shift**: half the difference between the "yes" rate with yes at A
  and with yes at B, in percentage points. +10 means 10 points of the answers
  follow the letter A whatever it means; −10 means they follow B.
- **agreement**: share of rows with the same yes/no verdict in both orders.

## Results

1. **Letter preference is real, and its direction depends on the question.**
   `neutral` and `named` pull toward **B** (mean shift −12 and −14 points,
   agreement about 70%); `neutral_cot` pulls toward **A** (+9). `named_cot`
   barely moves (−1, agreement 88%): it is the question least driven by the
   letter.
2. **In some cells gemma is answering the letter, not the question.**
   - ascii `named`: "yes" 1% with yes at A, 92% with yes at B. Agreement 9%.
     The "no on 308 of 310 rows" reported for `named` is mostly gemma picking
     B.
   - capture-the-secret `neutral`: agreement 36%, shift −32.
3. **How much it matters depends on the category.** ascii is most affected
   (mean |shift| 20 points, agreement 58%) and mask-factual least (3 points,
   94%).
4. **Alibi's self-report survives the swap; sycophancy B's letter result
   doesn't.**

   | Cell | Balanced accuracy, yes = A | yes = B | AUROC, yes = A | yes = B |
   |---|---|---|---|---|
   | alibi `named` | 85 | 74 | 0.96 | 0.94 |
   | alibi `named_cot` | 85 | 83 | 0.94 | 0.90 |
   | alibi `neutral` | 71 | 76 | 0.73 | 0.93 |
   | sycophancy B `named` | 70 | **51** | 0.73 | 0.77 |

   Sycophancy B `named` drops to a coin flip on the letters, but its
   probability ranking holds.
5. **The P("A") results hold for alibi and partly for mask-factual and
   sycophancy B, not for ascii.** Median AUROC over the four questions:
   alibi 0.85 → 0.92, mask-factual 0.75 → 0.67, sycophancy B 0.70 → 0.68,
   ascii 0.45 → **0.13**: ascii's probabilities follow the letter, not the
   lie.

### Letter shift (points; + = toward A, − = toward B)

<!-- SHIFT-GRID:START -->
| Question | ascii | syc A | syc B | mask | unans. | capture | alibi |
|---|---|---|---|---|---|---|---|
| `neutral` | +12 | -18 | -1 | -6 | -19 | -32 | -16 |
| `neutral_cot` | +16 | +9 | +7 | +6 | -2 | +19 | +5 |
| `named` | -46 | -15 | -18 | +0 | +2 | -16 | -5 |
| `named_cot` | -5 | -10 | +0 | +0 | +9 | +3 | +0 |
<!-- SHIFT-GRID:END -->

### Agreement between the two orders (same yes/no verdict)

<!-- AGREE-GRID:START -->
| Question | ascii | syc A | syc B | mask | unans. | capture | alibi |
|---|---|---|---|---|---|---|---|
| `neutral` | 77% | 65% | 98% | 88% | 61% | 36% | 67% |
| `neutral_cot` | 59% | 76% | 74% | 86% | 82% | 53% | 82% |
| `named` | 9% | 69% | 65% | 100% | 96% | 67% | 87% |
| `named_cot` | 85% | 68% | 100% | 100% | 79% | 95% | 92% |
<!-- AGREE-GRID:END -->

Per-cell yes rates, balanced accuracy and AUROC in both orders, plus AUROC on
the average of the two orders, are in `results/ab_swap_summary.csv` (which also
keeps the retired questions).

## What this changes elsewhere

- **The "naming the act moves the default answer" headline in
  `../RESULTS.md`** is partly letter preference. ascii `named` answers B, not
  "no". The unanswerable example ("did you lie?" 12.8% vs "did you
  confabulate?" 100%) holds: `named` there agrees across orders on 96% of rows.
- **Letter results outside alibi** should be read as fragile.
- **Averaging both orders** is the standard fix (Zheng et al.). Here it gives a
  mean AUROC of 0.60, against 0.61 with yes at A alone and 0.58 with yes at B
  alone.

The original run also covered seven retired A/B questions (6,886 answers in
total); its write-up is
[archived](../../archive/2026-10-01_all_13_wordings/ab_swap/README.md).

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
| `results/ab_swap_<category>.jsonl` | one record per answer (11 questions; filter on `condition`): swapped question, answer, letter, `admitted` (B = yes), letter probabilities, the original answer |
| `results/ab_swap_summary.csv` | every measure per cell |

The `ab_swap_*.jsonl` files are also on Hugging Face under `results/ab_swap/`.
Cost: about 4.3 GPU-hours on one A100 80 GB (14:19–18:35 on 30 Sep 2026).

```bash
uv run --with pandas --with numpy --with huggingface_hub \
    python gigi/1-replication/ab_swap/02_analyse_swap.py --pull
```
