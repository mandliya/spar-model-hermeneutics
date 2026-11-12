#!/usr/bin/env bash
# distill_box_hero9.sh: training half of distill/PLAN_hero9.md on the H100 box (12 Nov 2026). Launch detached:
#   setsid nohup bash /workspace/distill_box_hero9.sh > /workspace/logs/box.log 2>&1 < /dev/null &
# Stages: setup checks -> smoke gate (aborts cleanly) -> train (resumes up to 3 times) -> uploads. Every stage logs to
# /workspace/logs/<stage>.log and /workspace/logs/stages.txt. Never stops the box. The re-run of the checkpoints in the
# environments (vLLM + Modal) is a separate stage, started after this one.
set -uo pipefail
W=/workspace; LOG=$W/logs; OUT=$W/out/hero9_v1; DATA=$W/data/distill/hero9_v1; mkdir -p "$LOG" "$OUT"
set -a; . "$W/.env"; set +a                      # HF_TOKEN: in a file on the box, never in chat or git
REPO=model-hermeneutics/gigi-forensics-transcripts   # private
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=1
stage () { local n=$1; shift; echo "$(date -u +%FT%TZ) START $n" >> "$LOG/stages.txt"
           "$@" > "$LOG/$n.log" 2>&1; local rc=$?; echo "$(date -u +%FT%TZ) END $n rc=$rc" >> "$LOG/stages.txt"; return $rc; }
up () {   # an upload never stops the job
  python - "$1" "$2" <<'PY' >> "$LOG/uploads.log" 2>&1 || true
import os, sys, pathlib
from huggingface_hub import HfApi
src, dst = sys.argv[1], sys.argv[2]; api = HfApi(token=os.environ["HF_TOKEN"])
kw = dict(repo_id="model-hermeneutics/gigi-forensics-transcripts", repo_type="dataset")
if pathlib.Path(src).is_dir():
    api.upload_folder(folder_path=src, path_in_repo=dst, **kw)
else:
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, **kw)
print("uploaded", src, "->", dst)
PY
}

stage setup bash -c '
  nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /workspace && free -g &&
  pip install -q transformers==4.57.1 peft==0.17.1 trl==0.24.0 accelerate==1.10.1 datasets==4.2.0 pillow==11.3.0 \
      huggingface_hub hf_transfer jinja2 sentencepiece &&
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())" &&
  python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(\"google/gemma-3-27b-it\", token=os.environ[\"HF_TOKEN\"])
snapshot_download(\"'"$REPO"'\", repo_type=\"dataset\", allow_patterns=[\"distill/hero9_v1/*\"], local_dir=\"/workspace/data\", token=os.environ[\"HF_TOKEN\"])
print(\"downloads done\")"' \
  || { echo "$(date -u +%FT%TZ) SETUP FAILED: box idle, still billing" >> "$LOG/stages.txt"; exit 1; }

cd "$W"
stage smoke python distill_train.py train --data-dir "$DATA" --out "$W/out/smoke" --max-len 32768 --offload-activations --smoke \
  || { up "$W/out/smoke/log.json" distill/hero9_v1/runs/smoke_log.json; up "$LOG/smoke.log" distill/hero9_v1/runs/smoke.log
       echo "$(date -u +%FT%TZ) SMOKE FAILED: box idle, still billing" >> "$LOG/stages.txt"; exit 1; }
up "$W/out/smoke/log.json" distill/hero9_v1/runs/smoke_log.json

( while sleep 1800; do up "$OUT/log.json" distill/hero9_v1/runs/v1/log.json; done ) & UPL=$!
for try in 1 2 3; do
  stage "train_$try" python distill_train.py train --data-dir "$DATA" --out "$OUT" --max-len 32768 --offload-activations \
      --keep-checkpoints --save-every 24 --eval-every 24 --final-val-sample 0 --resume && break
done
kill $UPL 2>/dev/null
up "$OUT/log.json" distill/hero9_v1/runs/v1/log.json
up "$OUT/adapter" distill/hero9_v1/runs/v1/adapter
for c in "$OUT"/checkpoint-*; do [ -d "$c" ] && up "$c" "distill/hero9_v1/runs/v1/$(basename "$c")"; done
up "$OUT/val_per_example.jsonl" distill/hero9_v1/runs/v1/val_per_example.jsonl
echo "$(date -u +%FT%TZ) TRAINING DONE: checkpoints uploaded; box idle until the re-run stage, still billing" >> "$LOG/stages.txt"
