# Cross-category transfer

**Question.** Does a probe trained on one kind of lie catch the others? Does training on more kinds help (John, 10/5)? Does the answer change with scale (Gemma-27B, Qwen3-Next-80B, Qwen3-235B)?

## Run

- `scripts/run_no_followup_extract.sh` (GPU): Gemma's own replies with no follow-up question
- `scripts/run_qwen_extract.sh` (GPU, `DYL_MODEL=<qwen>`): a Qwen model reading its own conversations, no follow-up and neutral
- `scripts/run_generalization.sh` (CPU): a probe per category and wording, tested on the others, then the matrices
- `scripts/run_no_followup.sh` (CPU): the same with no follow-up, per read position
- `./run.sh compare_diversity` (CPU): held-out AUROC against the number of categories trained on (k-curve)
- `./run.sh compare_diversity_matrices` (CPU): its k = 1 probes as two transfer matrices
- `./run.sh compare_diversity_words` (CPU): the same k-curve with the words alone (TF-IDF) in place of a probe, on the probes' exact training rows

## Files

- `generalization.py`, `generalization_charts.py`, `compare_generalization.py`
- `no_followup.py`, `compare_no_followup.py`
- `diversity.py`, `diversity_charts.py`, `compare_diversity.py`, `compare_diversity_matrices.py`, `compare_diversity_words.py`

## Results

Under `<model>/gigi/results/` on the Hub (Gemma at the top level): `generalization_*`, `no_followup_*`, `diversity_*`, `diversity_matrices_*`. Charts are under `<model>/gigi/charts/`.
