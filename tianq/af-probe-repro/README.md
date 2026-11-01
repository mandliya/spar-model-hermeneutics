# af-probe-repro

Reproduces our sample-level alignment-faking paper (https://icml.cc/virtual/2026/79535) with the paper's own code, so later AF experiments (`../experiments/alignment_faking_hidden_reasoning/`) stand on a pipeline that is known to match it. The same role `../dyl-probe-repro` plays for AISI's "did you lie" probe.

- `vendor/probe-af/`: the paper's code, verbatim and unmodified (see `vendor/VENDORED.md`)
- `run.sh`: every step, logged and pushed to the Hub (`af-repro/<model>/`, `logs/`)
- `af_repro/paper.py`: the paper's tasks and layers, read off its code; what is left to extract
- `af_repro/check.py`: our results against the CSV the paper committed for each model
- `scripts/overnight.sh`: the driver for one GPU box

## Run

```bash
./run.sh extract hal9000   # GPU: activations at layers 33, 37, 40 for all 18 paper tasks (resumable)
./run.sh probes hal9000    # CPU: the paper's probe grid (LR at C = 1e4, 1, 1e-4 and diff-of-means; last and mean token)
./run.sh check hal9000     # passes if our AUROCs match the paper's
```

Models: `hal9000`, `pacifist` and `llama-70b-3.0`, as in `vendor/probe-af/src/models.py`. They are pinned by the revisions downloaded to the box's `/workspace/hf`, which `run.sh` always reads: `michaelwaves/hal9000` at `24f44ba`, `michaelwaves/pacifist` at `7e7e7e0`, `meta-llama/Meta-Llama-3-70B-Instruct` at `50fd307`.

## Pass condition

Every cell of the paper's headline table (`results/af_comparison_selected.csv`: synthetic AF training on the organism's own AF tasks, at its validation-selected design) is within 0.01 AUROC of the paper's committed value, and 95% of the matched rows are within 0.02, all but one ill-conditioned training set (`gen_misalign`, 50 examples: its CV rows reproduce exactly, its transfers swing), which is reported separately. For HAL9000 those are report_failure 0.9287 and report_score 0.6622 (difference of means, layer 37, last token); for Pacifist, highest_military_bid 0.6624 and report_failure 1.0. The parent model has no AF cells and is judged on its AF probes' rows.
