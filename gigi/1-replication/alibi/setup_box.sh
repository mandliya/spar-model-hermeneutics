#!/usr/bin/env bash
# One-time setup for a fresh box: packages, then the gated model once an
# HF token is present, then STOP. No `set -e` - the stop must always run.
cd /workspace/lies-experiment
source /venv/main/bin/activate
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "torch already on image: $(python -c 'import torch;print(torch.__version__)')"
say "=== installing packages (transformers pinned to 5.17.0 to match the box every other category's activations came from) ==="
pip install -q "transformers==5.17.0" accelerate pandas statsmodels python-dotenv openai tqdm \
  && say "--- packages OK ---" || say "!!! pip install failed"
python -c 'import transformers,pandas,statsmodels,dotenv,openai;print("imports ok, transformers",transformers.__version__)'

say "=== waiting up to 60 min for HF_TOKEN in .env ==="
for i in $(seq 1 60); do grep -qE '^HF_TOKEN=hf_' .env && break; sleep 60; done
if grep -qE '^HF_TOKEN=hf_' .env; then
  TOKEN=$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2-)
  say "token found - downloading google/gemma-3-27b-it (~52 GB)"
  HF_TOKEN="$TOKEN" python -c "from huggingface_hub import snapshot_download; p=snapshot_download('google/gemma-3-27b-it'); print('DOWNLOADED', p)" \
    && say "--- model cached ---" || say "!!! model download FAILED"
else
  say "!!! no HF_TOKEN after 60 min - model NOT downloaded"
fi

say "=== setup done - stopping instance ==="
bash stop_self.sh
