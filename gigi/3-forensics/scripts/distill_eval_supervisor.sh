#!/usr/bin/env bash
# distill_eval_supervisor.sh: the re-run half of distill/PLAN_hero9.md, unattended, from the laptop (12 Nov 2026).
# The Modal login stays on the laptop (not copied to the rented box), so the laptop must stay awake and plugged in.
#   1. wait for "TRAINING DONE" on the box (or a FAILED line: stop);
#   2. start vLLM on the box, detached (scripts/distill_serve_hero9.sh), and wait until it answers every model;
#   3. write the per-checkpoint configs; gate: one eval_tampering run on base gemma must come back scored;
#   4. launch 20 runs x 6 tasks for every model (base + checkpoints), 2 at a time per task and model;
#   5. wait for the apps, fetch, score (results/distill_v1_eval.csv). Never stops the box. Does not commit.
#   nohup caffeinate -is bash /abs/.../scripts/distill_eval_supervisor.sh > /abs/.../results/logs/distill_v1/supervisor.out 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
L="$HERE/results/logs/distill_v1"; mkdir -p "$L"; say () { echo "$(date '+%F %T') $*" >> "$L/supervisor.txt"; }
IP=5.145.175.103; SSHP=40253; PORT=40764   # Vast 60329886, H200 NVL
PY="$HERE/agent-interp-envs/.venv/bin/python"   # has yaml; the laptop python3 does not
SSH=(ssh -i "$HOME/.ssh/vastai_lies_experiment" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -p "$SSHP" "root@$IP")
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
TASKS="tower_of_hanoi:20 eval_tampering:20 currency_conversion:20 coding:20 coding_norules:20 lazy_investigation:20"
set -a; . "$ROOT/.env"; set +a   # VLLM_API_KEY (for the health check), never printed

say "start: waiting for TRAINING DONE on the box"
while true; do
  st=$("${SSH[@]}" 'cat /workspace/logs/stages.txt' 2>/dev/null)
  echo "$st" | grep -q "TRAINING DONE" && break
  echo "$st" | grep -q "FAILED" && { say "box reports a FAILED stage: stopping (box idle, still billing)"; exit 1; }
  sleep 300
done
say "training done; starting vLLM on the box"
"${SSH[@]}" 'setsid nohup bash /workspace/distill_serve_hero9.sh > /workspace/logs/serve.log 2>&1 < /dev/null & echo started' >> "$L/supervisor.txt" 2>&1
up=0
for i in $(seq 1 60); do   # install + load: up to 60 min
  if curl -sf -m 20 -H "Authorization: Bearer $VLLM_API_KEY" "http://$IP:$PORT/v1/models" > "$L/models.json"; then up=1; break; fi
  sleep 60
done
[ "$up" = 1 ] || { say "vLLM did not come up in 60 min (see /workspace/logs/serve.log on the box): stopping, box idle, still billing"; exit 1; }
MODELS=$("$PY" -c "
import json; ids=[m['id'] for m in json.load(open('$L/models.json'))['data']]
cs=sorted([i for i in ids if i.startswith('c')], key=lambda s: int(s[1:]))   # 'final' = the last checkpoint's weights: not run twice
print(','.join(['base'] + cs))")
say "vLLM up, models: $MODELS"
for m in ${MODELS//,/ }; do   # every model answers one tiny request before any Modal launch
  name=$m; [ "$m" = base ] && name=gemma-base
  r=$(curl -s -m 120 -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" "http://$IP:$PORT/v1/chat/completions" \
      -d "{\"model\":\"$name\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Say ok\"}]}")
  echo "$r" | grep -q '"choices"' || { say "model $name does not answer: $(echo "$r" | head -c 300). Stopping, box idle, still billing"; exit 1; }
done
say "every model answers"

"$PY" "$HERE/scripts/distill_eval.py" configs --host "$IP" --port "$PORT" --models "$MODELS" >> "$L/supervisor.txt" 2>&1
until (cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal secret list 2>/dev/null) | grep -q "vllm-secret-gigi"; do
  say "waiting for the Modal secret vllm-secret-gigi (GG creates it once)"; sleep 600
done
say "gate: one eval_tampering run on base gemma through Modal"
MAXC=1 GAP=15 MODAL_SECRETS=vllm-secret bash "$HERE/scripts/modal_full.sh" distill_v1_gate distill_v1_base eval_tampering:1
G="$HERE/results/logs/distill_v1_gate/eval_tampering__distill_v1_base.log"
for i in $(seq 1 60); do grep -q "App completed\|Stopping app" "$G" 2>/dev/null && break; sleep 30; done
"$PY" "$HERE/scripts/distill_eval.py" fetch --stages distill_v1_gate >> "$L/supervisor.txt" 2>&1
mkdir -p "$HERE/results/raw/distill_v1_gate_check"
"$PY" - "$HERE" >> "$L/supervisor.txt" 2>&1 <<'PY' || { say "GATE FAILED: the base-gemma run through the box did not score (see the gate log). Stopping, box idle, still billing"; exit 1; }
import pathlib, sys
H = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(H / "scripts"))
import full_runs_table as T
runs = sorted(r for r in (H / "results/raw/distill_v1_gate/eval_tampering").rglob("run-*") if r.is_dir())
ok = [r for r in runs if T.status(r) == "finished" and T.MEASURES["eval_tampering"][1](T.last_state(r), r) is not None]
print(f"gate: {len(runs)} run(s), {len(ok)} scored"); sys.exit(0 if ok else 1)
PY
say "gate passed: launching 20 runs x 6 tasks for $MODELS"
for m in ${MODELS//,/ }; do
  MAXC=2 GAP=12 MODAL_SECRETS=vllm-secret bash "$HERE/scripts/modal_full.sh" "distill_v1_$m" "distill_v1_$m" $TASKS
  say "launched distill_v1_$m"
done
STAGES=$(echo "$MODELS" | sed 's/[^,]*/distill_v1_&/g')
for i in $(seq 1 96); do   # up to 8 h
  left=0
  for s in ${STAGES//,/ }; do for f in "$HERE/results/logs/$s"/*__*.log; do
    grep -q "App completed\|Stopping app" "$f" 2>/dev/null || left=$((left + 1))
  done; done
  [ "$left" = 0 ] && break
  [ $((i % 6)) = 1 ] && say "waiting: $left task fleets still running"
  sleep 300
done
for pass in 1 2 3; do "$PY" "$HERE/scripts/distill_eval.py" fetch --stages "$STAGES" >> "$L/supervisor.txt" 2>&1 && break; sleep 120; done
"$PY" "$HERE/scripts/distill_eval.py" score >> "$L/supervisor.txt" 2>&1
say "DONE (box still serving and billing; not committed)"
