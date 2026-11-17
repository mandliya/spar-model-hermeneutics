#!/usr/bin/env bash
# classic_after.sh: data and training half of the classic round, run detached ON THE BOX (distill/PLAN_classic.md
# sections 2-4; 17 Nov 2026). Waits for classic_fleet3.sh's "CLASSIC FLEET DONE", then:
#   1. pick the first 40 complete runs per task by launch order, check that runs stopped at the context window reached
#      the 105,000-character cut, check the five held-out cases (classic_select.py). A failing case stops here: ask GG.
#   2. build distill/classic_v1 (Qwen's own complete runs only, 80/20 by run, five tasks held out), check it, upload it
#   3. stop the Qwen server (training needs the GPU; nothing after this uses Qwen)
#   4. training smoke (the 4 longest examples), then 2 epochs with checkpoints at 0.5 / 1 / 1.5 / 2 epochs (resumes up
#      to 3 times); each checkpoint goes to HF as soon as it exists, for TQ
#   5. teacher-forced loss of base gemma and every checkpoint on the 20% check set and the held-out tasks' runs
#   6. "TRAINING DONE", then classic_rerun.sh (gemma server + re-runs) if it exists.
# Any failed gate stops with a line in /workspace/logs/stages.txt. Never stops the box.
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/classic_after.sh > /workspace/logs/classic_after.out 2>&1 < /dev/null &
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; W=/workspace; LOG=$W/logs
S=$LOG/stages.txt; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
stop () { say "$* Box idle, still billing."; exit 1; }
set -a; . "$W/.env"; set +a                       # HF_TOKEN, never printed
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$W/venv-train/bin/python                       # torch 2.8.0, transformers 4.57.1, peft 0.17.1, trl 0.24.0 (venv_train.sh)
DATA=$H/distill/classic_v1; OUT=$W/out/classic_v1; HFD=distill/classic_v1
HELD=tower_of_hanoi,lazy_investigation,currency_conversion,eval_tampering,norvane
up () {   # an upload never stops the job: up <local file or folder> <path in the dataset> [ignore patterns...]
  local src=$1 dst=$2; shift 2
  $PY - "$src" "$dst" "$@" <<'PY' >> "$LOG/classic_uploads.log" 2>&1
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
  local rc=$?; [ $rc = 0 ] || say "UPLOAD FAILED: $src (classic_uploads.log)"; return $rc
}

DONE_RE="Z CLASSIC FLEET DONE: stages "   # the exact line classic_fleet3.sh writes (this script's own lines must not match)
say "classic_after: waiting for the fleet to finish (classic_fleet3.sh)"
until grep -q "$DONE_RE" "$S"; do
  pgrep -f "[c]lassic_fleet3.sh" > /dev/null || { grep -q "$DONE_RE" "$S" || stop "classic_after: classic_fleet3.sh ended without finishing the fleet: stopping."; }
  sleep 600
done
[ -s "$H/results/logs/classic/stages.txt" ] || stop "classic_after: results/logs/classic/stages.txt is missing: stopping."
STAGES=$(cat "$H/results/logs/classic/stages.txt")
cd "$H/agent-interp-envs"

# 1. selection and held-out check
env -u VIRTUAL_ENV uv run python ../scripts/classic_select.py --stages "$STAGES" > "$LOG/classic_select.log" 2>&1; rc=$?
[ $rc = 3 ] && stop "HELD-OUT CHECK FAILED (results/logs/classic/selection.txt): stopping to ask GG. The Qwen server is still up."
[ $rc = 0 ] || stop "SELECTION FAILED (classic_select.log). The Qwen server is still up."
say "selection: $(grep -E '^[0-9]+ runs kept' "$H/results/logs/classic/selection.txt")"
say "selection: $(grep -E '^Stopped at the context window' "$H/results/logs/classic/selection.txt")"
say "held-out check: all five cases hold (results/logs/classic/selection.txt)"
SEL="$H/results/classic_selected_per_run.csv"
ntask=$(cut -d, -f2 "$SEL" | tail -n +2 | sort -u | wc -l)   # column 2 = environment
[ "$ntask" = 28 ] || stop "SELECTION: $ntask tasks, not 28. The Qwen server is still up."
short=$(cut -d, -f2 "$SEL" | tail -n +2 | sort | uniq -c | awk '$1 < 30 {printf "%s:%s ", $2, $1}')
[ -z "$short" ] || stop "SELECTION: tasks with fewer than 30 complete runs: $short. Stopping to ask GG. The Qwen server is still up."

# 2. training data
env -u VIRTUAL_ENV uv run python ../scripts/distill_data.py build --hero9 --full-csv classic_selected_per_run.csv --no-swap \
  --end-check --keep-context-cut --expect-model qwen3.6-27b-bf16 --trajectories --max-context-chars 105000 \
  --holdout-envs "$HELD" --split 0.8 --seed 0 --max-mb 1000 --out distill/classic_v1 > "$LOG/classic_build.log" 2>&1 \
  || stop "DATA BUILD FAILED (classic_build.log). The Qwen server is still up."
env -u VIRTUAL_ENV uv run python - "$DATA" "$STAGES" "$HELD" > "$LOG/classic_build_check.log" 2>&1 <<'PY' || stop "DATA CHECK FAILED (classic_build_check.log). The Qwen server is still up."
import json, sys, pathlib, collections
d, stages, held = pathlib.Path(sys.argv[1]), sys.argv[2].split(","), set(sys.argv[3].split(","))
man = json.loads((d / "manifest.json").read_text()); bad = []
runs = collections.defaultdict(set)
for s in ("train", "val", "test"):
    for line in open(d / f"{s}.jsonl"):
        e = json.loads(line)
        if e["kind"] != "full": bad.append(f"{s}: kind {e['kind']} in {e['run']}")
        if not any(e["run"].startswith(f"results/raw/{st}/") for st in stages): bad.append(f"{s}: run outside the classic stages: {e['run']}")
        if (e["env"] in held) != (s == "test"): bad.append(f"{s}: {e['env']} in the wrong split")
        runs[s].add(e["run"])
if man.get("runs_not_in_sample"): bad.append(f"runs left out by the size cap: {len(man['runs_not_in_sample'])}")
if runs["train"] & runs["val"]: bad.append("a run in both train and val")
t = man["totals"]
print(f"train {t['train_runs']} runs, val {t['val_runs']} runs, test {t['test_runs']} runs; examples train {t['train']}, val {t['val']}, test {t['test']}")
print(f"tasks: train+val {len({json.loads(l)['env'] for s in ('train', 'val') for l in open(d / f'{s}.jsonl')})}, test {len({json.loads(l)['env'] for l in open(d / 'test.jsonl')})}")
print("excluded runs:", len(man.get("excluded_runs", [])))
for b in bad[:30]: print("BAD", b)
sys.exit(1 if bad else 0)
PY
say "training data: $(head -n 2 "$LOG/classic_build_check.log" | tr '\n' ';') (distill/classic_v1)"
up "$DATA" "$HFD/data"
up "$H/results/logs/classic" "$HFD/logs"
up "$H/results/classic_selected_per_run.csv" "$HFD/classic_selected_per_run.csv"

# 3. the GPU goes to training
pkill -f "[v]llm serve"; for i in $(seq 1 30); do pgrep -f "[v]llm serve" > /dev/null || break; sleep 10; done
pkill -9 -f "[v]llm serve"; sleep 15
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 4000 ] || stop "GPU still holds ${used} MiB after stopping the Qwen server: stopping."
say "Qwen server stopped; GPU free for training"

# 4. training: smoke, then 2 epochs with checkpoints every 0.5 epoch
cd "$H/scripts"
say "START train_smoke"
$PY distill_train.py train --data-dir "$DATA" --out "$W/out/classic_smoke" --max-len 32768 --offload-activations --smoke \
  > "$LOG/classic_train_smoke.log" 2>&1 || { up "$LOG/classic_train_smoke.log" "$HFD/runs/smoke.log"; stop "TRAINING SMOKE FAILED (classic_train_smoke.log)."; }
say "train_smoke OK: $(grep -o 'SMOKE OK.*' "$LOG/classic_train_smoke.log" | head -1)"
up "$W/out/classic_smoke/log.json" "$HFD/runs/smoke_log.json"
mkdir -p "$OUT"
( while sleep 900; do   # every finished checkpoint to HF once (adapter, config, trainer state; not the optimizer)
    for c in "$OUT"/checkpoint-*; do
      [ -f "$c/trainer_state.json" ] && [ ! -f "$c/.uploaded" ] && [ $(( $(date +%s) - $(stat -c %Y "$c/trainer_state.json") )) -gt 120 ] || continue
      up "$c" "$HFD/runs/v1/$(basename "$c")" "optimizer.pt" "scheduler.pt" "rng_state*.pth" && touch "$c/.uploaded" \
        && say "uploaded $(basename "$c") (epoch $($PY -c "import json;print(round(json.load(open('$c/trainer_state.json'))['epoch'],2))")) to HF"
    done
    [ -f "$OUT/log.json" ] && up "$OUT/log.json" "$HFD/runs/v1/log.json"
  done ) & UPL=$!
ok=0
for try in 1 2 3; do
  say "START train_$try"
  $PY distill_train.py train --data-dir "$DATA" --out "$OUT" --max-len 32768 --offload-activations --epochs 2 \
    --keep-checkpoints --save-every 0.25 --eval-every 0.25 --final-val-sample 0 --resume > "$LOG/classic_train_$try.log" 2>&1 && { ok=1; break; }
  say "train_$try ended with an error (classic_train_$try.log); resuming"
done
kill $UPL 2>/dev/null
[ $ok = 1 ] || { up "$OUT/log.json" "$HFD/runs/v1/log.json"; stop "TRAINING FAILED three times (classic_train_*.log)."; }
for c in "$OUT"/checkpoint-*; do [ -f "$c/.uploaded" ] || { up "$c" "$HFD/runs/v1/$(basename "$c")" "optimizer.pt" "scheduler.pt" "rng_state*.pth" && touch "$c/.uploaded"; }; done
up "$OUT/adapter" "$HFD/runs/v1/adapter"; up "$OUT/log.json" "$HFD/runs/v1/log.json"; up "$OUT/val_per_example.jsonl" "$HFD/runs/v1/val_per_example.jsonl"
say "training done: checkpoints $(ls -d "$OUT"/checkpoint-* | xargs -n1 basename | tr '\n' ' ')at epochs $($PY -c "
import json, glob; print(' '.join(str(round(json.load(open(f))['epoch'], 2)) for f in sorted(glob.glob('$OUT/checkpoint-*/trainer_state.json'), key=lambda p: int(p.split('checkpoint-')[1].split('/')[0]))))")"

# 5. teacher-forced loss per checkpoint (independent of the re-runs: a failure here is reported, not fatal)
ADS=$(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n | awk '{n = split($0, a, "-"); printf "%sc%s=%s", (NR > 1 ? "," : ""), a[n], $0}')
say "START checkpoint loss (base + $ADS)"
$PY distill_train.py nll --data-dir "$DATA" --splits val,test --adapters "$ADS" --out "$OUT/nll" --max-len 32768 \
  > "$LOG/classic_nll.log" 2>&1 && say "checkpoint loss: $(tail -n 2 "$LOG/classic_nll.log" | tr '\n' ';')" || say "CHECKPOINT LOSS FAILED (classic_nll.log); the re-runs go ahead"
up "$OUT/nll" "$HFD/runs/v1/nll"

say "TRAINING DONE: checkpoints and losses on HF ($HFD/runs/v1)"
[ -f "$H/scripts/classic_rerun.sh" ] && exec bash "$H/scripts/classic_rerun.sh"
say "no classic_rerun.sh yet: box idle, still billing"
