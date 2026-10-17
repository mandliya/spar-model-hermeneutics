> **Archived 1 October 2026.** A frozen copy from before the project kept only four follow-up
> questions (`neutral`, `neutral_cot`, `named`, `named_cot`). It still covers all 13.
> The live version is [answer_probability/README.md](../../../1-replication/answer_probability/README.md); what was archived and why is in
> [the archive index](../README.md).

# Is gemma's confidence more informative than its yes/no answer?

Every self-report number elsewhere in this project scores the **letter** gemma
wrote after "did you lie?": A (yes) or B (no). That throws away how sure it
was. Here, for every one of those answers, we read gemma's **probability** of
"A" at the moment it answered, and score self-report as an **AUROC**, the
measure the [lie-detectors post](https://alignment.anthropic.com/2026/lie-detectors/)
reports. It gives gemma-3-27b a self-report AUROC of **0.46–0.80**.

AUROC answers: pick one row where gemma lied and one where it didn't; how often
does the lie get the higher P(yes)? 0.5 is a coin flip, 1.0 is perfect, and
below 0.5 means the ranking points the wrong way.

## Method

- **Rows:** every row in every category that has follow-up answers (1,732
  rows, 19,049 answers), across the 11 A/B wordings: the 7 direct ones, the 3
  original reason-first ones, and `named_cot`.
- **No new text.** Each conversation is rebuilt exactly as gemma saw it (the
  same builder the float32 activations used), gemma's saved answer is fed back
  in, and one forward pass reads its next-token distribution at the position
  where it wrote the letter.
- **Score:** `p_yes = P("A") / (P("A") + P("B"))`, where A means "yes, I lied"
  in every wording. The raw log-probabilities of both letters and the top 5
  tokens are saved too.
- **The 2 confidence wordings** already have a 0–100 score, so their AUROC
  comes straight from the saved answers.
- **Ground truth and row filters** are exactly those of the tables in
  [`../RESULTS.md`](../RESULTS.md). As a check, balanced accuracy recomputed
  from the letters here matches that page's grid in all 77 letter cells.
- **Intervals:** 2,000 bootstrap resamples within each arm, seed 0.

## Results

1. **Probabilities carry more signal than letters.** 27 of 91 cells reach
   AUROC 0.70, against 8 of 91 for balanced accuracy. In 59 of 77 letter
   cells, AUROC beats balanced accuracy (mean 0.61 vs 0.54).
2. **The gain is largest where gemma gives the same letter almost every
   time.** mask-factual says "yes" to nearly everything, so its letters score
   50–61, but it is more sure when it really lied: AUROC 0.61–0.86. Sycophancy
   B's `neutral` answer is "No" on every row (balanced accuracy 50), yet AUROC
   is 0.70.
3. **Alibi is strongest again**: 11 of 13 wordings at 0.70 or above, up to
   **0.94** with `named`.
4. **Some cells point firmly the wrong way.** ascii `neutral` is **0.12** and
   `confidence_cot` **0.06**: gemma is more confident it lied on the rows
   where it told the truth. 9 cells have their whole interval below 0.5.
5. **Against the paper.** On its direct question (`neutral` here), our AUROCs
   run **0.12–0.75** across categories; on the reason-first version
   (`neutral_cot`), **0.48–0.77**. That overlaps the paper's 0.46–0.80 at the
   top but goes much lower, because ascii-sandbagging is strongly inverted.

### AUROC, every wording and category

Bold is 0.70 or above. Compare with the balanced-accuracy grid in
[`../RESULTS.md`](../RESULTS.md).

<!-- AUROC-GRID:START -->
| Wording | ascii | syc A | syc B | mask | unans. | capture | alibi |
|---|---|---|---|---|---|---|---|
| `neutral` | 0.12 | 0.61 | **0.70** | **0.75** | 0.50 | 0.45 | 0.66 |
| `low_stakes` | 0.32 | 0.61 | 0.63 | **0.86** | 0.58 | 0.52 | **0.93** |
| `high_stakes` | 0.25 | 0.51 | 0.52 | **0.77** | 0.32 | 0.53 | 0.63 |
| `confidence` | 0.46 | 0.53 | 0.52 | **0.77** | **0.73** | 0.57 | **0.70** |
| `neutral_cot` | 0.48 | 0.64 | 0.61 | **0.75** | 0.68 | 0.57 | **0.77** |
| `low_stakes_cot` | **0.75** | 0.68 | 0.69 | **0.80** | 0.46 | 0.51 | **0.88** |
| `high_stakes_cot` | **0.79** | 0.61 | 0.60 | **0.79** | 0.47 | 0.57 | **0.75** |
| `confidence_cot` | 0.06 | 0.61 | 0.65 | **0.84** | 0.60 | 0.53 | **0.76** |
| `plain` | 0.31 | 0.49 | 0.65 | 0.61 | 0.47 | 0.56 | **0.86** |
| `named` | 0.45 | 0.44 | **0.77** | 0.61 | 0.60 | 0.41 | **0.94** |
| `named_cot` | 0.47 | 0.52 | 0.66 | **0.77** | 0.31 | 0.52 | **0.91** |
| `eval_plain` | 0.56 | 0.58 | 0.58 | **0.74** | 0.49 | 0.58 | **0.82** |
| `eval_named` | **0.81** | 0.37 | 0.69 | 0.62 | 0.46 | 0.47 | **0.88** |
<!-- AUROC-GRID:END -->

Intervals and row counts for every cell are in `results/pa_auroc.csv`.

### Where the probability helps most

| Category | Wording | Balanced accuracy (letters) | AUROC (P("A")) |
|---|---|---|---|
| ascii | `high_stakes_cot` | 45 | 0.79 |
| ascii | `low_stakes_cot` | 43 | 0.75 |
| alibi | `low_stakes_cot` | 57 | 0.88 |
| mask-factual | `named_cot` | 52 | 0.77 |
| mask-factual | `low_stakes` | 61 | 0.86 |
| mask-factual | `high_stakes_cot` | 56 | 0.79 |

## Caveats

- **The signal lives in small probability differences.** Gemma is nearly
  certain in almost every answer: only 462 of 19,049 (2.4%) have P(yes)
  between 0.1 and 0.9. AUROC only needs the ranking, so it picks up
  differences like 0.9999 vs 0.99999. A working detector would have to
  threshold in that range, and whether that holds up on new data is untested.
- **Letter preference.** A always means "yes" here. The A/B swap test
  ([`../ab_swap/`](../../../1-replication/ab_swap)) reverses the options on a 626-row subset:
  mask-factual's and alibi's AUROCs hold (0.73 → 0.72, 0.86 → 0.89), but
  ascii's falls from 0.50 to 0.21, so its probabilities follow the letter
  rather than the lie.
- **Re-scoring vs generation.** For 102 answers (0.5%), the letter gemma
  originally wrote is not its top choice when the answer is re-scored. These
  are near-ties that flip under small numerical differences between
  generating and re-scoring. They are kept. 5 answers (capture-the-secret)
  have no letter probability, because the letter could not be mapped to a
  single token; they are left out, not guessed.
- **unanswerable has only 7 honest rows**, so its intervals are very wide.
- **The paper's range covers more than this.** Its 0.46–0.80 spans lie types
  and four prompt formats, two of them third-person monitoring, which we
  don't run. Its direct wording also differs slightly from our `neutral`
  ("Did you lie?" without "in your previous answer" or "Answer with just the
  letter").

## Files

| File | What it is |
|---|---|
| `06_answer_probability.py` | the GPU run: rebuild, teacher-force, read letter probabilities |
| `02_analyse_pa.py` | AUROC with bootstrap intervals; fills the grid above |
| `results/pa_<category>.jsonl` | one record per answer: letter, `lp_A`, `lp_B`, `p_A_norm`, top 5 tokens, `top1_is_answer` |
| `results/pa_auroc.csv` | AUROC, 95% interval, n per arm, and letter balanced accuracy per cell |
| `run_pa_swap.sh` | the detached box script that ran this and the swap test |

The `pa_*.jsonl` files are also on Hugging Face under
`results/answer_probability/`.

Cost: about 3.0 GPU-hours on one A100 80 GB.

```bash
uv run --with pandas --with numpy --with huggingface_hub \
    python gigi/answer_probability/02_analyse_pa.py --pull
```
