#!/usr/bin/env bash
# Every row of charts 3-4 that read "too few to tell" topped up to 60 scored replays per model (26 Oct 2026; the
# 21 Oct swap ran 10 per point, a number we picked, not John). On top of what parallel_checks.sh already has running:
#   coding at N (run-7 K8) and N-1 (run-7 K7): +50 per model; coding_norules at N: +10 per model at each of its 3
#   incident points (20 each, 60 pooled); dashboard at N (K63) and N-1 (K62): +56 Qwen / +55 gemma, spares for the
#   provider losses of 21 Oct; pre-commit at N-2 (q1 K6): +20 per model. Pre-commit at N-1 reaches 60 with
#   parallel_checks.sh alone. New fleets go next to the old ones, so the existing scoring picks them up.
# Logs: results/logs/topup60/; spend before/after in cost.txt.   bash gigi/3-forensics/scripts/topup60.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/topup60"; mkdir -p "$LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
usage () { ( set -a; . "$HERE/../../.env"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
echo "$(date '+%F %T') topup60 start | key spend before: \$$(usage)" >> "$LOG/cost.txt"
M=results/raw/modal; S=results/raw/swap_all
C=$M/gigi-forensics-coding/coding/qwen-qwen3-next-80b-a3b-instruct
CR=$C/2026-10-21_20-43-32-207727; CN=$C/2026-10-21_20-43-32-193139
DB=$M/gigi-forensics-dashboard-perf/dashboard_perf/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-57-49-884559/run-2
DBP=2026-10-19_10-57-49-884559_run-2
Q1=results/raw/screen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-20-35-423567/run-5
# name | results dir | checkpoint | count | overrides
cat > "$LOG/commands.txt" <<EOF
coding_r7K8_qwen|$S/coding/2026-10-21_20-43-32-207727_run-7_K8/qwen|$CR/run-7/step-8|50|
coding_r7K8_gemma|$S/coding/2026-10-21_20-43-32-207727_run-7_K8/gemma|$CR/run-7/step-8|50|$GEMMA
coding_r7K7_qwen|$S/coding/2026-10-21_20-43-32-207727_run-7_K7/qwen|$CR/run-7/step-7|50|
coding_r7K7_gemma|$S/coding/2026-10-21_20-43-32-207727_run-7_K7/gemma|$CR/run-7/step-7|50|$GEMMA
norules_r17K6_qwen|$S/coding_norules/2026-10-21_20-43-32-193139_run-17_K6/qwen|$CN/run-17/step-6|10|
norules_r17K6_gemma|$S/coding_norules/2026-10-21_20-43-32-193139_run-17_K6/gemma|$CN/run-17/step-6|10|$GEMMA
norules_r2K7_qwen|$S/coding_norules/2026-10-21_20-43-32-193139_run-2_K7/qwen|$CN/run-2/step-7|10|
norules_r2K7_gemma|$S/coding_norules/2026-10-21_20-43-32-193139_run-2_K7/gemma|$CN/run-2/step-7|10|$GEMMA
norules_r4K4_qwen|$S/coding_norules/2026-10-21_20-43-32-193139_run-4_K4/qwen|$CN/run-4/step-4|10|
norules_r4K4_gemma|$S/coding_norules/2026-10-21_20-43-32-193139_run-4_K4/gemma|$CN/run-4/step-4|10|$GEMMA
dashboard_K63_qwen|$S/dashboard_costly_failure/${DBP}_K63/qwen|$DB/step-63|56|
dashboard_K63_gemma|$S/dashboard_costly_failure/${DBP}_K63/gemma|$DB/step-63|55|$GEMMA
dashboard_K62_qwen|$S/dashboard_costly_failure/${DBP}_K62/qwen|$DB/step-62|56|
dashboard_K62_gemma|$S/dashboard_costly_failure/${DBP}_K62/gemma|$DB/step-62|55|$GEMMA
precommit_q1_6_qwen|results/raw/swap/precommit_q1/from_step6/qwen|$Q1/step-6|20|
precommit_q1_6_gemma|results/raw/swap_text/precommit_q1/from_step6/gemma|$Q1/step-6|20|$GEMMA
EOF
while IFS='|' read -r name out step n ov; do
  [ -d "$HERE/$step" ] || { echo "$(date '+%F %T') SKIP $name: no checkpoint $step" >> "$LOG/cost.txt"; continue; }
  for try in 1 2 3 4 5 6; do
    ( cd "$ENVS" && MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run --detach scripts/resume_modal.py \
        --step-dir "$HERE/$step" --count "$n" --use-current --max-concurrent 0 --no-judge --timeout 14400 \
        --results-dir "$HERE/$out" ${ov:+--overrides "$ov"} > "$LOG/$name.log" 2>&1 & ); sleep 30
    grep -q "rate limit exceeded" "$LOG/$name.log" || break
    sleep $((15 * try))
  done
  echo "$(date '+%F %T') launched $name x$n" >> "$LOG/cost.txt"
done < "$LOG/commands.txt"
echo "$(date '+%F %T') all launched" >> "$LOG/cost.txt"
