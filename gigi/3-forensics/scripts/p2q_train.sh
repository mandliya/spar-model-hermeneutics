#!/usr/bin/env bash
# p2q_train.sh (23 Nov 2026): train Qwen3.6 27B (thinking on) on DeepSeek V4 Pro's runs, on GPU 0 of the phase 2 box
# (phase2/PLAN_deepseek_training.md). As classic_after.sh steps 4-5: smoke (the 4 longest examples), then 2 epochs with
# a saved copy every 0.5 epoch (resumes up to 3 times); each copy goes to HF as soon as it exists; then the loss of
# the untrained model and every copy on the 20% check runs and the held-out tasks' runs. Never stops the box.
#   setsid nohup bash scripts/p2q_train.sh > /workspace/logs/p2q_train.out 2>&1 < /dev/null &
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; W=/workspace; LOG=$W/logs
S=$LOG/stages.txt; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
stop () { say "$* GPU 0 idle, box still billing."; exit 1; }
set -a; . "$W/.env"; set +a                       # HF_TOKEN, never printed
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
PY=$W/venv-train-qwen/bin/python
DATA=$H/distill/p2_deepseekv4_v1; OUT=$W/out/p2_deepseekv4_v1; HFD=distill/p2_deepseekv4_v1
REV=8f33b98937c6031bb49e713dee3e8e185aafd26f
MODEL=$W/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/$REV
up () {   # an upload never stops the job: up <local file or folder> <path in the dataset> [ignore patterns...]
  local src=$1 dst=$2; shift 2
  $PY - "$src" "$dst" "$@" <<'PY' >> "$LOG/p2q_uploads.log" 2>&1
import os, sys, pathlib
from huggingface_hub import HfApi
src, dst, ign = sys.argv[1], sys.argv[2], sys.argv[3:]; api = HfApi(token=os.environ["HF_TOKEN"])
kw = dict(repo_id="model-hermeneutics/gigi-forensics-transcripts", repo_type="dataset")
if pathlib.Path(src).is_dir():
    api.upload_folder(folder_path=src, path_in_repo=dst, ignore_patterns=ign or None, **kw)
else:
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, **kw)
print("uploaded", src, "->", dst)
PY
  local rc=$?; [ $rc = 0 ] || say "UPLOAD FAILED: $src (p2q_uploads.log)"; return $rc
}
cd "$H/scripts"
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  say "START train_smoke (Qwen3.6 student, 32,768 tokens, GPU 0)"
  $PY distill_train.py train --student qwen36 --model "$MODEL" --data-dir "$DATA" --out "$W/out/p2q_smoke" --max-len 32768 \
    --offload-activations --smoke > "$LOG/p2q_train_smoke.log" 2>&1 \
    || { up "$LOG/p2q_train_smoke.log" "$HFD/runs/smoke.log"; stop "TRAINING SMOKE FAILED (p2q_train_smoke.log)."; }
  say "train_smoke OK: $(grep -o 'SMOKE OK.*' "$LOG/p2q_train_smoke.log" | head -1)"
  up "$W/out/p2q_smoke/log.json" "$HFD/runs/smoke_log.json"
fi
mkdir -p "$OUT"
( while sleep 600; do   # every finished copy to HF once (adapter, config, trainer state; not the optimizer)
    for c in "$OUT"/checkpoint-*; do
      [ -f "$c/trainer_state.json" ] && [ ! -f "$c/.uploaded" ] && [ $(( $(date +%s) - $(stat -c %Y "$c/trainer_state.json") )) -gt 120 ] || continue
      up "$c" "$HFD/runs/v1/$(basename "$c")" "optimizer.pt" "scheduler.pt" "rng_state*.pth" && touch "$c/.uploaded" \
        && say "CHECKPOINT READY $(basename "$c") (epoch $($PY -c "import json;print(round(json.load(open('$c/trainer_state.json'))['epoch'],2))")), on HF"
    done
    [ -f "$OUT/log.json" ] && up "$OUT/log.json" "$HFD/runs/v1/log.json"
  done ) & UPL=$!
ok=0
for try in 1 2 3; do
  say "START train_$try (2 epochs, a saved copy every 0.5 epoch)"
  $PY distill_train.py train --student qwen36 --model "$MODEL" --data-dir "$DATA" --out "$OUT" --max-len 32768 \
    --offload-activations --epochs 2 --keep-checkpoints --save-every 0.25 --eval-every 0.25 --final-val-sample 0 --resume \
    > "$LOG/p2q_train_$try.log" 2>&1 && { ok=1; break; }
  say "train_$try ended with an error (p2q_train_$try.log); resuming"
done
kill $UPL 2>/dev/null
[ $ok = 1 ] || { up "$OUT/log.json" "$HFD/runs/v1/log.json"; stop "TRAINING FAILED three times (p2q_train_*.log)."; }
for c in "$OUT"/checkpoint-*; do [ -f "$c/.uploaded" ] || { up "$c" "$HFD/runs/v1/$(basename "$c")" "optimizer.pt" "scheduler.pt" "rng_state*.pth" && touch "$c/.uploaded" && say "CHECKPOINT READY $(basename "$c"), on HF"; }; done
up "$OUT/adapter" "$HFD/runs/v1/adapter"; up "$OUT/log.json" "$HFD/runs/v1/log.json"; up "$OUT/val_per_example.jsonl" "$HFD/runs/v1/val_per_example.jsonl"
say "TRAINING DONE: copies $(ls -d "$OUT"/checkpoint-* | xargs -n1 basename | tr '\n' ' ')"
ADS=$(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n | awk '{n = split($0, a, "-"); printf "%sc%s=%s", (NR > 1 ? "," : ""), a[n], $0}')
say "START checkpoint loss (base + $ADS)"
$PY distill_train.py nll --student qwen36 --model "$MODEL" --data-dir "$DATA" --splits val,test --adapters "$ADS" --out "$OUT/nll" --max-len 32768 \
  > "$LOG/p2q_nll.log" 2>&1 && say "checkpoint loss: $(tail -n 2 "$LOG/p2q_nll.log" | tr '\n' ';')" || say "CHECKPOINT LOSS FAILED (p2q_nll.log)"
up "$OUT/nll" "$HFD/runs/v1/nll"
say "P2Q TRAIN ALL DONE"
