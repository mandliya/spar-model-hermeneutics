#!/usr/bin/env bash
# Fresh box -> gemma third-person monitor, as one detached job:
# packages, model download (3 tries), then 02_monitor_gemma.py (its first two
# batches are a gate), results uploaded to HF as it goes. NO automatic stop.
cd /workspace/third_person
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.hf_home

say "=== setup: packages ==="
pip install -q "transformers==5.17.0" accelerate python-dotenv huggingface_hub \
  && say "--- packages OK ---" || say "!!! pip failed"
python -c 'import transformers, torch; print("transformers", transformers.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'

say "=== setup: gemma-3-27b-it ==="
TOKEN=$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2-)
GOT=0
for attempt in 1 2 3; do
  if HF_TOKEN="$TOKEN" python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('google/gemma-3-27b-it'))"; then GOT=1; break; fi
  say "download attempt $attempt failed, retrying"; sleep 20
done
[ "$GOT" -eq 1 ] || { say "!!! model download failed 3 times - box left running"; exit 1; }
say "--- model cached ---"

say "=== gemma as third-person monitor ==="
python 02_monitor_gemma.py && say "--- monitor OK ---" || say "!!! monitor failed (resumable; results so far on HF)"
say "=== ALL STAGES FINISHED - box is now IDLE and STILL BILLING; stop it by hand ==="
