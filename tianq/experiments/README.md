# Experiments

Each experiment built on the DYL pipeline (`../dyl-probe-repro`) has its own folder:

- `probe_vs_just_asking/`: the released probe vs just asking, on Gigi's categories
- `cross_category_transfer/`: does a probe trained on one kind of lie catch the others? (Gemma, Qwen-80B, Qwen-235B)
- `hermeneutics/`: John's probe faithfulness gap (target minus substitute AUROC)
- `alignment_faking_gate/`: HAL9000 gate check for the distillation idea
- `common/`: shared chart style

Every experiment folder has the same layout:

- `README.md`: the question, how to run it, where the results are
- `*.py`: the analysis, its charts and its `compare_*` entry point
- `scripts/`: the drivers that ran it, kept as run
- `report/`: messages sent, figures and diagrams

## Running

Every step goes through `dyl-probe-repro/run.sh <command>`. It maps the command to its module, puts `dyl-probe-repro` and `tianq` on `PYTHONPATH`, and pushes a raw log to the Hub. Drivers in `scripts/` `cd` into `dyl-probe-repro` first.

Results, activations and logs live on HF `model-hermeneutics/dyl-probe-repro`. Its local mirror is `dyl-probe-repro/scratch/`.
