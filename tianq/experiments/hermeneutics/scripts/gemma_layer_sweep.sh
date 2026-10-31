#!/usr/bin/env bash
# Box A follow-up (moved from Box B, destroyed early): Gemma-27B reading Qwen3-235B's conversations at three more layers, to see whether its flat
# "did you lie?" transfer is the layer we read. CPU only (activations for every layer are already stored).
cd /workspace/spar-model-hermeneutics/tianq/dyl-probe-repro
source /venv/main/bin/activate
set -a; source <(tr -d '\r' < .env); set +a
export HF_HOME=/workspace/hf PYTHONPATH="$PWD" DYL_MODEL=qwen3-235b DYL_READER=gemma-3-27b-it
ST=/workspace/sweep_status.log
echo "$(date -u +%FT%TZ) sweep started: why = Gemma readers stay flat on did-you-lie transfer while Qwen3.5 tracks the 235B; is it the layer?" >> $ST
for L in 20 31 45; do
  echo "$(date -u +%FT%TZ) started layer $L" >> $ST
  DYL_LAYER=$L ./run.sh compare_diversity > /dev/null 2>&1 && echo "$(date -u +%FT%TZ) ok layer $L" >> $ST || echo "$(date -u +%FT%TZ) FAILED layer $L" >> $ST
done
echo "$(date -u +%FT%TZ) SWEEP DONE" >> $ST
python -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='$ST', path_in_repo='logs/overnight_sweep_boxA.log', repo_id=os.environ['HF_DATASET_REPO'], repo_type='dataset')" > /dev/null 2>&1
