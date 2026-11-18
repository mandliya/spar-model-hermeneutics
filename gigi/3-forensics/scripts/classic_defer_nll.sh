#!/usr/bin/env bash
# classic_defer_nll.sh (18 Nov 2026, GG: "go results faster"): the re-runs come BEFORE the teacher-forced loss, so the
# re-run results arrive ~4 h sooner; same work, same cost. Run detached ON THE BOX while classic_after.sh trains:
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/classic_defer_nll.sh > /workspace/logs/classic_defer_nll.out 2>&1 < /dev/null &
# 1. Waits for classic_after.sh to start its loss step ("distill_train.py nll") and stops it at once. classic_after.sh
#    is built to go on when that step fails, so it goes straight to TRAINING DONE and classic_rerun.sh.
# 2. Waits for the re-runs to finish (classic_rerun.sh's own done line). Then a spend check (measured spend + ~$20 for
#    ~4 h of box must stay under SPEND_LIMIT, default 530, else stop and ask GG), stops the gemma server (the loss needs
#    the GPU), runs the same loss command as classic_after.sh (it resumes per example), uploads it to HF.
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; W=/workspace; LOG=$W/logs; S=$LOG/stages.txt
say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
OUT=$W/out/classic_v1; DATA=$H/distill/classic_v1; HFD=distill/classic_v1; PY=$W/venv-train/bin/python
set -a; . "$W/.env"; set +a                       # HF_TOKEN, never printed
export HF_HOME=$W/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODAL_PROFILE=spar

# 1. stop the loss step as soon as classic_after.sh starts it
say "defer_nll: waiting for classic_after.sh's loss step, to move it after the re-runs (GG, 18 Nov: results first)"
while true; do
  pid=$(pgrep -f "distill_train\.py nll" | head -n 1)
  [ -n "$pid" ] && break
  pgrep -f "scripts/classic_after\.sh" > /dev/null || { say "defer_nll: classic_after.sh ended before its loss step: nothing deferred"; exit 1; }
  sleep 15
done
say "defer_nll: loss step stopped on purpose (moved after the re-runs); classic_after.sh goes on to the re-runs. Its next line, CHECKPOINT LOSS FAILED, is this stop, not an error"
kill "$pid"
for i in $(seq 1 12); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
kill -0 "$pid" 2>/dev/null && kill -9 "$pid"

# 2. after the re-runs: spend check, free the GPU, loss, upload
DONE_RE="Z CLASSIC RERUNS DONE: "                 # the exact line classic_rerun.sh writes; no line of this script matches it
while ! grep -q "$DONE_RE" "$S"; do
  if ! pgrep -f "scripts/classic_(after|rerun)\.sh" > /dev/null; then
    grep -q "$DONE_RE" "$S" && break
    say "defer_nll: the re-runs stopped without finishing (see the lines above): the loss step waits for GG. Box idle, still billing."; exit 1
  fi
  sleep 120
done
cd "$H/agent-interp-envs"
SPEND=$(timeout 600 env -u VIRTUAL_ENV uv run python - <<'PY' 2>> "$LOG/classic_defer_nll.spend.log"
import datetime as dt, json, subprocess
rows = json.loads(subprocess.run(["modal", "billing", "report", "--start", "2026-11-17", "--resolution", "h", "--json"],
                                 capture_output=True, text=True, check=True).stdout)
modal = sum(float(r["cost"]) for r in rows if "/workspace/fo" in r["description"])
box = 4.77 * (dt.datetime.now(dt.timezone.utc) - dt.datetime(2026, 11, 17, 4, 15, tzinfo=dt.timezone.utc)).total_seconds() / 3600
print(f"{modal + box + 15:.0f}")
PY
)
LIMIT=${SPEND_LIMIT:-530}
case "$SPEND" in ''|*[!0-9]*) say "defer_nll SPEND CHECK: could not measure the spend: ask GG before the loss step. Gemma server up. Box idle, still billing."; exit 1;; esac
[ $((SPEND + 20)) -le "$LIMIT" ] || { say "defer_nll SPEND CHECK: \$$SPEND spent + ~\$20 for the loss step is over the \$$LIMIT GG approved: ask GG. Gemma server up. Box idle, still billing."; exit 1; }
say "defer_nll: re-runs done; spend so far \$$SPEND, loss step ~\$20 more (limit \$$LIMIT); stopping the gemma server for it"
pkill -f "[v]llm serve"
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
  [ "$used" -lt 4000 ] && break; sleep 15
done
[ "$used" -lt 4000 ] || { say "defer_nll: the GPU is still in use ($used MiB) after stopping the gemma server: ask GG. Box idle, still billing."; exit 1; }
cd "$H/scripts"
ADS=$(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n | awk '{n = split($0, a, "-"); printf "%sc%s=%s", (NR > 1 ? "," : ""), a[n], $0}')
say "START checkpoint loss, after the re-runs (base + $ADS)"
$PY distill_train.py nll --data-dir "$DATA" --splits val,test --adapters "$ADS" --out "$OUT/nll" --max-len 32768 \
  > "$LOG/classic_nll2.log" 2>&1 && say "checkpoint loss: $(tail -n 2 "$LOG/classic_nll2.log" | tr '\n' ';')" \
  || { say "CHECKPOINT LOSS FAILED (classic_nll2.log). Box idle, still billing."; exit 1; }
$PY - "$OUT/nll" "$HFD/runs/v1/nll" >> "$LOG/classic_uploads.log" 2>&1 <<'PY' && say "checkpoint loss uploaded to HF ($HFD/runs/v1/nll)" || say "UPLOAD FAILED: checkpoint loss (classic_uploads.log)"
import os, sys
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(folder_path=sys.argv[1], path_in_repo=sys.argv[2],
    repo_id="model-hermeneutics/gigi-forensics-transcripts", repo_type="dataset")
PY
say "CLASSIC LOSS DONE (after the re-runs). Box idle, still billing."
