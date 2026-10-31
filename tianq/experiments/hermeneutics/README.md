# Hermeneutics: the probe faithfulness gap

**Question.** Can an open model prefilled with a target model's conversations stand in for the target's own probe? Gap = target probe AUROC minus substitute probe AUROC, per category (John, 10/7; lesswrong.com/posts/CZLhfNGaDqtYxvGJD).

- Targets: Qwen3-Next-80B and Qwen3-235B
- Substitutes: Gemma-3-12B, Gemma-3-27B and Qwen3.5-27B
- Baseline: words only (TF-IDF on the conversation)

## Run

- `scripts/run_hermeneutics.sh` (GPU, `DYL_MODEL=<target>`): every substitute reads the target's conversations, then the scores, diversity charts and gaps. `GAPS=0` stops before the gaps.
- `./run.sh compare_faithfulness` (CPU): in-category scores for the target, or for `DYL_READER`
- `./run.sh compare_faithfulness --gaps <readers>` (CPU): the gaps with paired bootstrap intervals, plus the words-only baseline
- `scripts/gemma_layer_sweep.sh`: Gemma-27B on the 235B conversations at layers 20, 31 and 45 (`DYL_LAYER`). It asks whether Gemma's flat did-you-lie transfer comes from the layer we read.

## The Oct 30 overnight run (`scripts/overnight_boxes/`)

These are the files exactly as they ran, so paths point at the session scratchpad and at `/workspace` on the boxes.

- `box_driver.sh`: the detached driver per box
  - Box A ran the Qwen3-235B self-read
  - Box B ran the AF gate, then the readers on the 235B conversations, then the gaps
- `verify_hub.py`: the completeness check run before any destroy
- `box_download.py`, `start_downloads.sh`, `push_slim.sh`, `setup_env.sh`: model downloads, code push and environment
- `watcher.sh`, `watcher.log`: the local watcher that destroyed verified boxes, with its log

Status logs are on the Hub: `logs/overnight_status_boxA.log`, `logs/overnight_status_boxB.log`.

## Files

- `faithfulness.py`, `faithfulness_charts.py`, `compare_faithfulness.py`

## Results

On the Hub:

- Target scores: `<target>/gigi/results/faithfulness_scores_*`
- Each substitute's scores: `<target>/read_by_<reader>/gigi/results/`
- Gaps: `<target>/gigi/results/faithfulness_gaps.json`
- Chart: `faithfulness_gaps.png`
