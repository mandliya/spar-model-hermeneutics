#!/usr/bin/env bash
# p2q_runs.sh (23 Nov 2026): runs of Qwen3.6 27B (thinking on) on our own server, ON THE LAUNCH BOX, for the DeepSeek
# round (phase2/PLAN_deepseek_training.md): the untrained model ("base", the "before" runs) or a saved copy (c<step>,
# loaded on the server as a LoRA adapter first). The 6 held-out tasks and the 2 trained check tasks, N runs each.
#   setsid nohup bash scripts/p2q_runs.sh base 40 > /workspace/logs/p2q_runs_base.out 2>&1 < /dev/null &
#   GATE=1 bash scripts/p2q_runs.sh base      # one eval_tampering run through Modal, stage distill_p2ds_gate
set -uo pipefail
S=/workspace/logs/p2q.txt; . /workspace/forensics/gigi/3-forensics/scripts/phase2_lib.sh   # H, say, upy, PATH, Modal names
set -a; . /workspace/.env; set +a   # VLLM_API_KEY (the server's password, as in the Modal secret), HF_TOKEN: never printed
IP=207.148.106.151; PORT=40479      # Vast 62121347's public address for the server's port 8000
export DISTILL_V=p2ds DISTILL_HELD_OUT=tower_of_hanoi,lazy_investigation,currency_conversion,eval_tampering,norvane,secret_number
export DISTILL_ENVS=$DISTILL_HELD_OUT,coding,coding_norules
M=${1:?model: base or c<step>}; N=${2:-40}; MAXC=${MAXC:-3}
L=$H/results/logs/p2q_runs; mkdir -p "$L"
curl -sf -m 20 -H "Authorization: Bearer $VLLM_API_KEY" "http://$IP:$PORT/v1/models" > "$L/models_$M.json" \
  || { say "p2q_runs $M: the server does not answer at $IP:$PORT: stopping"; exit 1; }
served=$([ "$M" = base ] && echo qwen36-base || echo "$M")
grep -q "\"$served\"" "$L/models_$M.json" || { say "p2q_runs $M: the server does not serve $served: stopping"; exit 1; }
upy ../scripts/distill_eval.py configs --student qwen36 --base-name qwen36-base --host "$IP" --port "$PORT" --models "$M" \
  > "$L/configs_$M.txt" 2>&1 || { say "p2q_runs $M: CONFIGS FAILED ($L/configs_$M.txt)"; exit 1; }
alive () { pgrep -f "results/raw/$1/$2\$" > /dev/null; }
live_all () { local n=0 f e; for f in "$H/results/logs/$1"/*__*.log; do [ -f "$f" ] || continue; e=$(basename "$f"); alive "$1" "${e%%__*}" && n=$((n + 1)); done; echo $n; }
if [ "${GATE:-0}" = 1 ]; then ST=distill_p2ds_gate; COUNTS="eval_tampering:1"; else
  ST=distill_p2ds_$M; COUNTS=$(for e in ${DISTILL_ENVS//,/ }; do printf "%s:%s " "$e" "$N"; done); fi
say "p2q_runs $M: launching $COUNTS (stage $ST, $MAXC at a time per task)"
MAXC=$MAXC GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" "$ST" "distill_p2ds_$M" $COUNTS >> "$L/launch_$M.txt" 2>&1
sleep 60
while [ "$(live_all "$ST")" -gt 0 ]; do sleep 300; say "p2q_runs $M: $(live_all "$ST") apps still running"; done
upy ../scripts/distill_fetch_lean.py "$ST" >> "$L/fetch.log" 2>&1 || say "p2q_runs $M: FETCH FAILED (fetch.log)"
say "P2Q RUNS DONE $M (stage $ST fetched)"
