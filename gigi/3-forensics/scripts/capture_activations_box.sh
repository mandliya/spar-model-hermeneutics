#!/usr/bin/env bash
# The whole activation-capture sequence for a fresh Vast box (>= 80 GB GPU: RTX PRO 6000, H100, A100 80GB; never GB10).
# Launch ONCE, detached, and walk away; it never stops the box (the user stops it in the conversation):
#   setsid nohup bash capture_activations_box.sh > capture_box.log 2>&1 &
# Needs, next to this script: .env with HF_TOKEN (the gigi-forensics-transcripts dataset is where results go; the HF
# token also reads google/gemma-3-27b-it, which is gated). Env knobs: MODELS ("qwen3.6-27b gemma-3-27b"), STAGES per model
# (STAGES_QWEN, STAGES_GEMMA), LAYERS (default 16,32,48), MAX_TOKENS (default 65536), REPO.
# Stages: 1 setup (repo clone, deps), 2 transcripts download from HF, 3 smoke on 2 transcripts per model as a gate,
# 4 the run (resumable: skips runs already in the dataset), 5 nothing else: every run is uploaded as it is written.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
LOG="$HERE/capture_box.txt"; say () { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
set -a; . "$HERE/.env"; set +a
: "${HF_TOKEN:?HF_TOKEN missing in .env}"
REPO=${REPO:-model-hermeneutics/gigi-forensics-transcripts}
MODELS=${MODELS:-"qwen3.6-27b gemma-3-27b"}
STAGES_QWEN=${STAGES_QWEN:-"hero9_qwen36 hero9_qwen36_key hero9_qwen36_all"}
STAGES_GEMMA=${STAGES_GEMMA:-"hero9_gemma hero9_gemma_key hero9_gemma_all full_laptop base_gemma"}
LAYERS=${LAYERS:-16,32,48}; MAX_TOKENS=${MAX_TOKENS:-65536}
TOKEN_LEVEL=${TOKEN_LEVEL:-0}   # 1 = every reply token too (hundreds of GB for Qwen3.6); 0 = per-reply mean and last token only
DATA=${DATA:-/workspace/data}; mkdir -p "$DATA"

# 1. setup
say "stage 1: setup"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
if [ ! -d "$HERE/spar-model-hermeneutics" ]; then
  git clone --depth 1 https://github.com/mandliya/spar-model-hermeneutics.git "$HERE/spar-model-hermeneutics" >> "$LOG" 2>&1 || { say "clone failed"; exit 1; }
fi
F="$HERE/spar-model-hermeneutics/gigi/3-forensics"
python3 -m pip install -q -U "transformers>=4.57" torch accelerate safetensors huggingface_hub pyyaml openai tenacity python-dotenv >> "$LOG" 2>&1 || { say "pip failed"; exit 1; }
python3 -m pip install -q flash-linear-attention causal-conv1d >> "$LOG" 2>&1 || say "fla / causal-conv1d kernels not installed: Qwen3.6 (Gated DeltaNet layers) will run on the slow torch path"
python3 -m pip install -q -e "$F/agent-interp-envs" >> "$LOG" 2>&1 || say "agent-interp-envs install failed (tool schemas fall back to execute_command only: CHECK index.json tools_source)"
cp "$HERE/.env" "$F/scripts/.env"

# 2. transcripts
say "stage 2: transcripts from $REPO"
python3 - "$REPO" "$DATA/transcripts" "$STAGES_QWEN $STAGES_GEMMA" >> "$LOG" 2>&1 <<'PY' || { say "download failed"; exit 1; }
import sys
from huggingface_hub import snapshot_download
repo, dest, stages = sys.argv[1], sys.argv[2], sys.argv[3].split()
snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=[f"{s}/**" for s in stages] + ["manifest.csv"])
PY

run_model () {  # $1 model tag, $2 stages, $3 extra args
  python3 "$F/scripts/capture_activations.py" --model-tag "$1" --runs-root "$DATA/transcripts" --stages $2 \
    --envs-root "$F/agent-interp-envs" --configs-root "$F/configs" --layers "$LAYERS" --max-tokens "$MAX_TOKENS" \
    --out "$DATA/activations" --repo "$REPO" $( [ "$TOKEN_LEVEL" = 1 ] || echo --no-token-level ) $3 >> "$LOG" 2>&1
}

for m in $MODELS; do
  stages=$STAGES_QWEN; [ "$m" = gemma-3-27b ] && stages=$STAGES_GEMMA
  # 3. smoke: two transcripts, uploaded, as the gate
  say "stage 3: smoke $m (2 runs)"
  run_model "$m" "$stages" "--limit 2" || { say "SMOKE FAILED for $m: not running the rest for this model"; continue; }
  grep -q "FAILED\|SKIP" <(tail -n 6 "$DATA/activations/capture_$m.log") && say "smoke $m had a skip/fail line, read capture_$m.log before trusting the run"
  # 4. the run
  say "stage 4: full run $m"
  run_model "$m" "$stages" "" && say "$m done" || say "$m ended with errors (see $DATA/activations/capture_$m.log); the box stays up"
done
say "ALL DONE. The box is still running and billing; stop it yourself."
