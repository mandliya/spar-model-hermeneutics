#!/usr/bin/env bash
# p2q_watch.sh (23 Nov 2026), ON THE LAUNCH BOX: run every saved copy as soon as the server serves it. The GPU box
# loads a copy as c<step> and unloads it again if its check fails (p2q_copies.sh), so a name still listed 6 minutes
# after it first appears has passed. One copy at a time: 8 tasks x N runs (p2q_runs.sh), then the score table.
#   setsid nohup bash scripts/p2q_watch.sh 4 40 > /workspace/logs/p2q_watch.out 2>&1 < /dev/null &
S=/workspace/logs/p2q.txt; . /workspace/forensics/gigi/3-forensics/scripts/phase2_lib.sh
set -a; . /workspace/.env; set +a
WANT=${1:-4}; N=${2:-40}; IP=207.148.106.151; PORT=40479; DONE=/workspace/logs/p2q_copies_done.txt; touch "$DONE"
export DISTILL_V=p2ds DISTILL_HELD_OUT=tower_of_hanoi,lazy_investigation,currency_conversion,eval_tampering,norvane,secret_number
export DISTILL_ENVS=$DISTILL_HELD_OUT,coding,coding_norules
names () { curl -sf -m 20 -H "Authorization: Bearer $VLLM_API_KEY" "http://$IP:$PORT/v1/models" | grep -oE '"id": ?"c[0-9]+"' | grep -oE 'c[0-9]+' | sort -u -t c -k2 -n; }
score () { upy ../scripts/distill_eval.py score > "$H/results/logs/p2q_runs/score_latest.txt" 2>&1 && say "p2q score table updated (results/distill_p2ds_eval.csv)" || say "p2q SCORING FAILED (results/logs/p2q_runs/score_latest.txt)"; }
say "p2q_watch: waiting for $WANT saved copies on the server"
while [ "$(grep -c . "$DONE")" -lt "$WANT" ]; do
  for n in $(names); do
    grep -qx "$n" "$DONE" && continue
    sleep 360; names | grep -qx "$n" || { say "p2q_watch: $n was unloaded (its check failed): skipped"; continue; }
    pgrep -f "[p]2q_runs.sh" > /dev/null && { while pgrep -f "[p]2q_runs.sh" > /dev/null; do sleep 60; done; }
    MAXC=${MAXC:-3} bash "$H/scripts/p2q_runs.sh" "$n" "$N" > "/workspace/logs/p2q_runs_$n.out" 2>&1
    echo "$n" >> "$DONE"; score
  done
  sleep 120
done
say "P2Q ALL COPIES RUN: $(tr '\n' ' ' < "$DONE")"
