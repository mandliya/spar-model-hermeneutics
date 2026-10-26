#!/usr/bin/env bash
# Two checks from the 22 Oct call, launched together on Modal (26 Oct 2026), detached, from the current tree:
#  A  John's long-history test: gemma takes over 5 long Qwen runs at turn 100 (checkpoint 99) and has to carry the
#     task on for up to 100 more turns (agent.max_steps=200); 10 gemma replays per run, 3 Qwen replays as the
#     baseline. Not graded: the question is whether gemma still operates (scripts/longhist_check.py).
#     Pass, fixed before running: gemma reaches a normal end (final answer, submit/end_session or the step cap) in
#     >= 90% of its replays that were not lost to the provider, with unreadable turns no higher than in its own runs.
#  B  The "unclear" rows of chart 4, topped up: pre-commit hook at N-1 (q1 checkpoint 7, q2 checkpoint 2) and N-2
#     (q1 checkpoint 6), 20 more replays per point and model; dashboard's incident (Qwen run-2, checkpoints 63 and 62),
#     10 more Qwen replays each (most of Qwen's were lost to rate limits on 21 Oct). New fleets go next to the old
#     ones, so the existing scoring picks them up.
# Logs: results/logs/parallel_checks/; spend before/after in cost.txt.   bash gigi/3-forensics/scripts/parallel_checks.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/parallel_checks"; mkdir -p "$LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
usage () { ( set -a; . "$HERE/../../.env"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
echo "$(date '+%F %T') parallel_checks start | key spend before: \$$(usage)" >> "$LOG/cost.txt"
M=results/raw/modal
Q1=results/raw/screen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-20-35-423567/run-5
Q2=results/raw/base_qwen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-24-48-348561/run-18
DB=$M/gigi-forensics-dashboard-perf/dashboard_perf/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-57-49-884559/run-2
DBP=2026-10-19_10-57-49-884559_run-2
# name | results dir | checkpoint | count | overrides
cat > "$LOG/commands.txt" <<EOF
A_dashboard_gemma|results/raw/longhist/dashboard_costly_failure/gemma|$M/gigi-forensics-dashboard-perf/dashboard_perf/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-57-49-884559/run-4/step-99|10|$GEMMA,agent.max_steps=200
A_dashboard_qwen|results/raw/longhist/dashboard_costly_failure/qwen|$M/gigi-forensics-dashboard-perf/dashboard_perf/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-57-49-884559/run-4/step-99|3|agent.max_steps=200
A_gradeyourown_gemma|results/raw/longhist/grade_your_own/gemma|$M/gigi-forensics-grade-your-own/grade_your_own/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-54-43-338056/run-5/step-99|10|$GEMMA,agent.max_steps=200
A_gradeyourown_qwen|results/raw/longhist/grade_your_own/qwen|$M/gigi-forensics-grade-your-own/grade_your_own/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-54-43-338056/run-5/step-99|3|agent.max_steps=200
A_lsrg_gemma|results/raw/longhist/lucky_seed_readable_grader/gemma|$M/gigi-forensics-lucky-seed-readable-grader/lucky_seed_readable_grader/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-56-55-100600/run-5/step-99|10|$GEMMA,agent.max_steps=200
A_lsrg_qwen|results/raw/longhist/lucky_seed_readable_grader/qwen|$M/gigi-forensics-lucky-seed-readable-grader/lucky_seed_readable_grader/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-56-55-100600/run-5/step-99|3|agent.max_steps=200
A_puppeteer_gemma|results/raw/longhist/puppeteer/gemma|$M/gigi-forensics-puppeteer/puppeteer/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-55-45-718688/run-5/step-99|10|$GEMMA,agent.max_steps=200
A_puppeteer_qwen|results/raw/longhist/puppeteer/qwen|$M/gigi-forensics-puppeteer/puppeteer/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-55-45-718688/run-5/step-99|3|agent.max_steps=200
A_refextract_gemma|results/raw/longhist/reference_extraction/gemma|$M/gigi-forensics-aj-data-anon/aj_data_anon/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-58-13-974739/run-9/step-99|10|$GEMMA,agent.max_steps=200
A_refextract_qwen|results/raw/longhist/reference_extraction/qwen|$M/gigi-forensics-aj-data-anon/aj_data_anon/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-58-13-974739/run-9/step-99|3|agent.max_steps=200
B_precommit_q1_7_gemma|results/raw/swap_text/precommit_q1/from_step7/gemma|$Q1/step-7|20|$GEMMA
B_precommit_q1_7_qwen|results/raw/swap/precommit_q1/from_step7/qwen|$Q1/step-7|20|
B_precommit_q2_2_gemma|results/raw/swap_text/precommit_q2/from_step2/gemma|$Q2/step-2|20|$GEMMA
B_precommit_q2_2_qwen|results/raw/swap/precommit_q2/from_step2/qwen|$Q2/step-2|20|
B_precommit_q1_6_gemma|results/raw/swap_text/precommit_q1/from_step6/gemma|$Q1/step-6|20|$GEMMA
B_precommit_q1_6_qwen|results/raw/swap/precommit_q1/from_step6/qwen|$Q1/step-6|20|
B_dashboard_K63_qwen|results/raw/swap_all/dashboard_costly_failure/${DBP}_K63/qwen|$DB/step-63|10|
B_dashboard_K62_qwen|results/raw/swap_all/dashboard_costly_failure/${DBP}_K62/qwen|$DB/step-62|10|
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
