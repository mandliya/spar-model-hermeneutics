# Part 2: hermeneutics

Qwen3-Next-80B as the target and gemma-3-27b as its stand-in: on the same
prompts, and swapped into Qwen's conversations, does gemma answer the way Qwen
does? Also the Qwen3-235B run for TQ's probe. Results, methodology and
charts: [`RESULTS.md`](RESULTS.md).

- [`analysis/`](analysis/): no GPU, no API.
  - `compare_models.py`: lying rates, self-report AUROC, the swap test and the
    four counts for gemma, Qwen3-32B, Qwen3-Next-80B and Qwen3-235B, into
    `../qwen/results/comparison_*.csv`.
  - `four_paths.py`: John's two parts (lie or no lie; the four paths), into
    `../qwen/results/hermeneutics_part*.csv`.
  - `substitute_gap.py`: the AUROC gap between Qwen about itself and gemma
    reading Qwen's conversations, into `../qwen/results/substitute_gap_*.csv`.
- [`charts/`](charts/): the charts on `RESULTS.md`, drawn by
  `../qwen/18_substitute_charts.py` to `../qwen/21_qwen235b_charts.py`.
- The runs themselves were made by the pipeline in [`../qwen/`](../qwen/README.md)
  (the 80B, the 235B, and gemma reading the 80B's conversations); their data is
  in `../qwen/results/`.

To regenerate everything: `RESULTS.md`, section "Regenerating this page".
