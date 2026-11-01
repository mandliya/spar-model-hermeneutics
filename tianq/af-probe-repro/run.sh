#!/usr/bin/env bash
# Entry point for every step of the AF paper reproduction (https://icml.cc/virtual/2026/79535).
# Each step runs the paper's own code (vendor/probe-af, unmodified) and writes only under scratch/.
#
#   ./run.sh extract <model>   # GPU: activations at the paper's layers for the tasks the paper ran on it (resumable)
#   ./run.sh probes <model>    # CPU: the paper's probe grid (vendor scripts/train_test_probes_af.sh) on its tasks
#   ./run.sh check <model>     # CPU: our results against the paper's committed CSV; fails on a mismatch
#   ./run.sh gate-extract <model>  # GPU: the AF gate's activations (../experiments/alignment_faking_hidden_reasoning/extract.py)
#   ./run.sh gate-score all        # CPU: the AF gate itself (../experiments/alignment_faking_hidden_reasoning/score.py)
#   ./run.sh propensity <step>     # one step of ../experiments/alignment_faking_propensity (its <step>.py)
#
# <model> is a key of vendor/probe-af/src/models.py: hal9000, pacifist or llama-70b-3.0 (the gate adds llama-8b-3.1).
# Every invocation keeps a raw log (argv, environment, full output, exit code) in scratch/logs, pushed to
# the Hub's logs/ when it ends; its outputs go to the Hub under af-repro/<model>/. Models are read from
# $HF_HOME/hub at the revisions downloaded there. The .env is dyl-probe-repro's (HF_TOKEN, HF_DATASET_REPO).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/vendor/probe-af"
set -a
# shellcheck disable=SC1091
source <(tr -d '\r' < "$HERE/../dyl-probe-repro/.env")
set +a
# The models are read from the box's pinned cache, never the image's own HF_HOME (which would fetch main
# again, a second full copy). The gate imports the paper's src/ and our experiments/ package.
export HF_HOME=/workspace/hf PYTHONPATH="$HERE:$HERE/../dyl-probe-repro:$HERE/..:$VENDOR${PYTHONPATH:+:$PYTHONPATH}"

step="${1:?usage: ./run.sh <extract|probes|check|gate-extract|gate-score|propensity> <model|step>}"
model="${2:?usage: ./run.sh <extract|probes|check|gate-extract|gate-score|propensity> <model|step>}"
case "$step" in extract | probes | check | gate-extract | gate-score | propensity) ;; *) echo "unknown step: $step" >&2; exit 1 ;; esac

OUT="$HERE/scratch/out/$model"
FEATURES="$OUT/extracted_feats_33-37-40_$model.h5"
RESULTS="$OUT/results_${model}_all_layers.csv"
LOG_DIR="$HERE/scratch/logs"
mkdir -p "$OUT" "$LOG_DIR"
LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ)_af_${step}_${model}.log"
started=$(date +%s)
{
  printf '### %s  host=%s  cwd=%s\n$ ./run.sh %s %s\n--- python\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "$PWD" "$step" "$model"
  python --version
  printf -- '--- nvidia-smi\n'
  if command -v nvidia-smi >/dev/null; then nvidia-smi; else echo "nvidia-smi not present on this host"; fi
  printf -- '--- pip freeze\n'
  python -m pip freeze
  printf -- '--- output\n'
} >>"$LOG" 2>&1
echo "logging to $LOG"

run() {
  # The paper's scripts resolve data/ from their own root and load the tokenizer from a fixed relative
  # cache, ../deception-detection/data/huggingface: point that at this box's model cache.
  mkdir -p "$HERE/vendor/deception-detection/data"
  [[ -e "$HERE/vendor/deception-detection/data/huggingface" ]] || ln -s "$HF_HOME/hub" "$HERE/vendor/deception-detection/data/huggingface"
  case "$step" in
    extract)
      todo=$(python -m af_repro.paper missing "$model" "$FEATURES")
      if [[ -z "$todo" ]]; then echo "every task already extracted in $FEATURES"; return 0; fi
      echo "to extract: $todo"
      # shellcheck disable=SC2086  # the task list, split on purpose
      (cd "$VENDOR" && python scripts/extract_activation.py --model "$model" --huggingface-path "$HF_HOME/hub" \
        --layers 33,37,40 --output "$FEATURES" --gpus "$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)" \
        --tasks $todo)
      left=$(python -m af_repro.paper missing "$model" "$FEATURES")
      [[ -z "$left" ]] || { echo "still missing after extraction: $left" >&2; return 1; }
      ;;
    probes)
      # The training and held-out test tasks the paper's committed results show for this model.
      read -r -a train <<< "$(python -m af_repro.paper train "$model")"
      read -r -a test <<< "$(python -m af_repro.paper test "$model")"
      for feature in last average; do
        for probe in "lr_lbfgs 10000" "lr_lbfgs 1" "lr_lbfgs 1e-4" "diffmean 0"; do
          set -- $probe
          (cd "$VENDOR" && python scripts/train_test_probes.py --probe-type "$1" --regularization "$2" \
            --use-scale true --balance-groups false --layer-idx 33 37 40 \
            --train-feature-type "$feature" --test-feature-type "$feature" --verbose --model-name "$model" \
            --features-file "$FEATURES" --probe-dir "$OUT/probes" --results-csv "$RESULTS"             --train-tasks "${train[@]}" --test-tasks "${test[@]}")
        done
      done
      ;;
    check)
      python -m af_repro.check "$model" "$RESULTS" "$OUT/check_$model.json"
      ;;
    gate-extract)
      python -m experiments.alignment_faking_hidden_reasoning.extract "$model"
      ;;
    gate-score)
      python -m experiments.alignment_faking_hidden_reasoning.score
      ;;
    propensity)
      python -m "experiments.alignment_faking_propensity.$model"
      ;;
  esac
}

set +e
PYTHONUNBUFFERED=1 run 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
printf '### exit=%s  seconds=%s\n' "$status" "$(( $(date +%s) - started ))" | tee -a "$LOG"

# Outputs (activations, probes, results, the check) and the log go to the Hub, success or failure. The gate
# and propensity steps push their own outputs (af-gate/, af-propensity/) as they go.
[[ "$step" == gate-* || "$step" == propensity ]] || python - "$OUT" "$model" <<'PY' || echo "output push failed" >&2
import os, sys
from huggingface_hub import HfApi
out, model = sys.argv[1], sys.argv[2]
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(folder_path=out, path_in_repo=f"af-repro/{model}",
    repo_id=os.environ["HF_DATASET_REPO"], repo_type="dataset", commit_message=f"af-repro: {model} outputs")
PY
python -m dyl_repro.core.logs "$LOG_DIR"
exit "$status"
