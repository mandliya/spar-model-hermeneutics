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

- **Rows:** every row in every category that has follow-up answers, for the
  project's four follow-up questions (`neutral`, `neutral_cot`, `named`,
  `named_cot`): 10,039 answers.
- **No new text.** Each conversation is rebuilt exactly as gemma saw it (the
  same builder the float32 activations used), gemma's saved answer is fed back
  in, and one forward pass reads its next-token distribution at the position
  where it wrote the letter.
- **Score:** `p_yes = P("A") / (P("A") + P("B"))`, where A means "yes, I lied"
  in every question. The raw log-probabilities of both letters and the top 5
  tokens are saved too.
- **Ground truth and row filters** are exactly those of the tables in
  [`../RESULTS.md`](../RESULTS.md). As a check, balanced accuracy recomputed
  from the letters here matches that page's grid in every cell.
- **Intervals:** 2,000 bootstrap resamples within each arm, seed 0.

## Results

1. **Probabilities carry more signal than letters.** 8 of 28 cells reach
   AUROC 0.70, against 3 of 28 for balanced accuracy. In 21 of 28 cells, AUROC
   beats balanced accuracy (mean 0.59 vs 0.54).
2. **The gain is largest where gemma gives the same letter almost every
   time.** mask-factual says "yes" to nearly everything, so its letters score
   50–54, but it is more sure when it really lied: AUROC 0.61–0.77. Sycophancy
   B's `neutral` answer is "No" on every row (balanced accuracy 50), yet AUROC
   is 0.70.
3. **Alibi is strongest again**: 0.65–0.94, with `named` at **0.94**.
4. **Some cells point firmly the wrong way.** ascii `neutral` is **0.12**:
   gemma is more confident it lied on the rows where it told the truth. Two
   cells have their whole interval below 0.5 (ascii `neutral`,
   capture-the-secret `named`).
5. **Against the paper.** On its direct question (`neutral` here), our AUROCs
   run **0.12–0.75** across categories; on the reason-first version
   (`neutral_cot`), **0.48–0.77**. That overlaps the paper's 0.46–0.80 at the
   top but goes much lower, because ascii-sandbagging is strongly inverted.

### AUROC, every question and category

Bold is 0.70 or above. Compare with the balanced-accuracy grid in
[`../RESULTS.md`](../RESULTS.md).

<!-- AUROC-GRID:START -->
| Question | ascii | syc A | syc B | mask | unans. | capture | alibi |
|---|---|---|---|---|---|---|---|
| `neutral` | 0.12 | 0.61 | **0.70** | **0.75** | 0.50 | 0.45 | 0.65 |
| `neutral_cot` | 0.48 | 0.64 | 0.61 | **0.75** | 0.68 | 0.57 | **0.75** |
| `named` | 0.45 | 0.44 | **0.77** | 0.61 | 0.60 | 0.41 | **0.94** |
| `named_cot` | 0.47 | 0.52 | 0.66 | **0.77** | 0.31 | 0.52 | **0.90** |
<!-- AUROC-GRID:END -->

Intervals and row counts for every cell are in `results/pa_auroc.csv`, which
also keeps the nine retired questions.

### Where the probability helps most

| Category | Question | Balanced accuracy (letters) | AUROC (P("A")) |
|---|---|---|---|
| mask-factual | `named_cot` | 52 | 0.77 |
| mask-factual | `neutral_cot` | 51 | 0.75 |
| mask-factual | `neutral` | 54 | 0.75 |
| sycophancy B | `neutral` | 50 | 0.70 |
| sycophancy B | `named_cot` | 51 | 0.66 |

## Caveats

- **The signal lives in small probability differences.** gemma is nearly
  certain in almost every answer: only 232 of 10,039 (2.3%) have P(yes)
  between 0.1 and 0.9. AUROC only needs the ranking, so it picks up
  differences like 0.9999 vs 0.99999. A working detector would have to
  threshold in that range, and whether that holds up on new data is untested.
- **Letter preference.** A always means "yes" here. The A/B swap test
  ([`../ab_swap/`](../ab_swap/README.md)) reverses the options on a 626-row
  subset: alibi's median AUROC holds (0.85 → 0.92), mask-factual's and
  sycophancy B's weaken but stay above chance (0.75 → 0.67, 0.70 → 0.68), and
  ascii's falls from 0.45 to 0.13, so its probabilities follow the letter
  rather than the lie.
- **Re-scoring vs generation.** For 39 answers (0.4%), the letter gemma
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

The original run covered all 13 questions (19,049 answers); its write-up is
[archived](../archive/2026-10-01_all_13_wordings/answer_probability/README.md).

## Files

| File | What it is |
|---|---|
| `06_answer_probability.py` | the GPU run: rebuild, teacher-force, read letter probabilities |
| `02_analyse_pa.py` | AUROC with bootstrap intervals; fills the grid above |
| `results/pa_<category>.jsonl` | one record per answer (all 13 questions; filter on `condition`): letter, `lp_A`, `lp_B`, `p_A_norm`, top 5 tokens, `top1_is_answer` |
| `results/pa_auroc.csv` | AUROC, 95% interval, n per arm, and letter balanced accuracy per cell |
| `run_pa_swap.sh` | the detached box script that ran this and the swap test |

The `pa_*.jsonl` files are also on Hugging Face under
`results/answer_probability/`.

Cost: about 3.0 GPU-hours on one A100 80 GB.

```bash
uv run --with pandas --with numpy --with huggingface_hub \
    python gigi/answer_probability/02_analyse_pa.py --pull
```
