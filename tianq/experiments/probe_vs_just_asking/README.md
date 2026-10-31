# Probe vs just asking

**Question.** On Gigi's categories, does AISI's released "did you lie?" probe beat just asking the model? What does it flag on each kind of answer?

## Run

- `scripts/run_new_gigi_categories.sh` (GPU): extracts the five categories added at Gigi's 0df3751
- `scripts/run_gigi_datasets.sh` (GPU): extracts and scores every category with every follow-up wording
- `./run.sh compare_wordings` (CPU): just asking vs the probe, best held-out wording each, plus the hard-to-catch lies
- `./run.sh compare_outcome_flags` (CPU): the share of each answer group the released probe flags

## Files

- `wordings.py`, `wording_charts.py`, `compare_wordings.py`
- `outcome_flags.py`, `outcome_flag_charts.py`, `compare_outcome_flags.py`
- `dyl_repro/datasets/gigi/transfer.py` (in the pipeline): the per-row probe and just-asking scores both build on

## Results

Under `gigi/results/` on the Hub; the charts are under `gigi/charts/`.
