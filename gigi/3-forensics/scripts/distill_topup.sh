#!/usr/bin/env bash
# distill_topup.sh: bring every model x task cell of the distillation re-runs to 20 finished runs (13 Nov 2026).
# On 12-13 Nov the laptop slept on battery and Modal stopped most apps ("ConflictError: App is stopped"): 214 of 600
# runs started. Each round: count finished runs per cell (scored + "no value", from results/distill_v1_eval_per_run.csv),
# park the old launch log in dead/, launch only the missing runs, wait, lean-fetch, re-score. Up to 3 rounds.
#   nohup caffeinate -is bash /abs/.../scripts/distill_topup.sh > /abs/.../results/logs/distill_v1/topup.out 2>&1 &
set -uo pipefail
export DISTILL_V=${DISTILL_V:-v1}; V=$DISTILL_V; MAXC=${MAXC:-2}   # v2 (13 Nov redo, CUDA graphs off): MAXC=4
HERE="$(cd "$(dirname "$0")/.." && pwd)"; L="$HERE/results/logs/distill_$V"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/topup.txt"; }
PY="$HERE/agent-interp-envs/.venv/bin/python"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
STAGES=distill_${V}_base,distill_${V}_c24,distill_${V}_c48,distill_${V}_c72,distill_${V}_c96
fetch_score () {
  (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python ../scripts/distill_fetch_lean.py "$STAGES") >> "$L/topup.txt" 2>&1
  "$PY" "$HERE/scripts/distill_eval.py" score > "$L/score_latest.txt" 2>&1
}
for round in 1 2 3 4 5 6; do
  fetch_score
  NEED=$("$PY" - "$HERE" <<'PY'
import csv, collections, os, pathlib, sys
H = pathlib.Path(sys.argv[1]); done = collections.Counter()
P = H / f"results/distill_{os.environ['DISTILL_V']}_eval_per_run.csv"   # absent before the first run of a new version
for r in (csv.DictReader(open(P)) if P.exists() else []):
    if r["status"] in ("scored", "no value"):
        done[(r["model"], r["env"])] += 1
envs = ["tower_of_hanoi", "eval_tampering", "currency_conversion", "coding", "coding_norules", "lazy_investigation"]
print(" ".join(f"{m}:{e}:{20 - done[(m, e)]}" for m in ["base", "c24", "c48", "c72", "c96"] for e in envs if done[(m, e)] < 20))
PY
)
  [ -z "$NEED" ] && { say "every cell has 20 finished runs"; break; }
  say "round $round: missing $NEED"
  for cell in $NEED; do
    m=${cell%%:*}; rest=${cell#*:}; e=${rest%%:*}; n=${rest##*:}
    f="$HERE/results/logs/distill_${V}_$m/${e}__distill_${V}_$m.log"
    [ -f "$f" ] && { mkdir -p "$(dirname "$f")/dead"; mv "$f" "$(dirname "$f")/dead/$(basename "$f" .log).r$round.$(date +%H%M%S).log"; }
    MAXC=$MAXC GAP=12 MODAL_SECRETS=vllm-secret bash "$HERE/scripts/modal_full.sh" "distill_${V}_$m" "distill_${V}_$m" "$e:$n"
  done
  say "round $round launched"
  for i in $(seq 1 120); do   # up to 10 h
    left=0
    for cell in $NEED; do m=${cell%%:*}; rest=${cell#*:}; e=${rest%%:*}
      f="$HERE/results/logs/distill_${V}_$m/${e}__distill_${V}_$m.log"   # finished, or its launcher lost Modal (13 Nov: a Wi-Fi drop left
      grep -qE "App completed|Stopping app|All runs failed" "$f" 2>/dev/null && continue   # dead fleets the loop waited on for 2 h)
      tail -n 40 "$f" 2>/dev/null | grep -qE "ConnectionError|nodename nor servname|Could not connect to the Modal server" && continue
      left=$((left + 1))
    done
    [ "$left" = 0 ] && break
    [ $((i % 6)) = 1 ] && say "round $round: $left fleets running"
    sleep 300
  done
done
fetch_score
if [ -f /workspace/.env ]; then   # on the Vast box (13 Nov: the launcher moved there): results to the private HF dataset
  ( set -a; . /workspace/.env; set +a; python - "$HERE" <<'UP'
import os, sys, pathlib
from huggingface_hub import HfApi
H = pathlib.Path(sys.argv[1]); api = HfApi(token=os.environ["HF_TOKEN"])
kw = dict(repo_id="model-hermeneutics/gigi-forensics-transcripts", repo_type="dataset")
V = os.environ["DISTILL_V"]
for s in [f"distill_{V}_{m}" for m in ("base", "c24", "c48", "c72", "c96")]:
    api.upload_folder(folder_path=str(H / "results/raw" / s), path_in_repo=f"distill/hero9_v1/eval_{V}/raw/{s}", ignore_patterns=["*.part", "**/fs/**"], **kw)
for f in [f"distill_{V}_eval.csv", f"distill_{V}_eval_per_run.csv"]:
    api.upload_file(path_or_fileobj=str(H / "results" / f), path_in_repo=f"distill/hero9_v1/eval_{V}/{f}", **kw)
print("uploaded to HF")
UP
  ) >> "$L/topup.txt" 2>&1 || say "HF upload failed (results are on the box)"
fi
say "DONE: see results/distill_${V}_eval.csv (box still serving and billing; not committed)"
