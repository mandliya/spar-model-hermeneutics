#!/usr/bin/env bash
# general_tbench_launch.sh (24 Nov 2026): DeepSeek V4 Pro teacher runs on Terminal-Bench 2 (89 tasks), through Harbor
# with our own shell-tool agent (scripts/general_tbench_agent.py) and task containers on Modal. Design and prices:
# phase2/general_set/TERMINAL_BENCH.md. Runs ON THE LAUNCH BOX, never the laptop (a sleeping laptop loses the runs: the
# loop that calls the model lives in this process). Every paid mode needs GG's yes for that spend first.
#
#   bash scripts/general_tbench_launch.sh setup     # free: Harbor + Modal client in their own venv, the benchmark at a pinned commit
#   bash scripts/general_tbench_launch.sh check     # free: self-test, the first request printed (not sent), Harbor's own dry run
#   Paid modes, each detached and sleep-safe (one line, paste as is):
#   MODE=oracle setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/general_tbench_launch.sh oracle > /workspace/logs/general_tbench_oracle.out 2>&1 < /dev/null &
#       no model: Harbor's reference solutions on ORACLE_TASKS (default: all 89) to find tasks that do not run on Modal. Modal only (~$3).
#   SPEND_CAP=5   ... general_tbench_launch.sh smoke    # 1 run of fix-git and 1 of regex-log with the teacher, then the smoke check
#   SPEND_CAP=60  ... general_tbench_launch.sh pilot    # PILOT_N (2) runs of the 12 PILOT_TASKS; prints cost and length per run
#   SPEND_CAP=<4 x the pilot's projection> N=32 ... general_tbench_launch.sh full   # TARGET (11) complete runs of every task
#
# Settings (environment): N = runs at once (default 8; 32 for the full collection); TARGET = complete runs wanted per
# task; MAX_ROUNDS = top-up rounds (default 3); GT_MAX_STEPS (100), GT_COMMAND_TIMEOUT (600 s), TIMEOUT_MULT (4: each
# task's own time limit times 4, so the step cap ends a run, not the clock); TASKS = comma list to restrict; EXCLUDE = comma list to leave out;
# STAGE = name.
# Resumable: start the same mode again and it launches only the runs still missing (general_tbench_plan.py). Every
# round is a new Harbor job folder; nothing already saved is touched, moved or deleted.
# Money: SPEND_CAP is a hard cap on the real `usage.cost` of every reply of this stage (one ledger file). The agent
# checks it before every request; this script checks it every 5 minutes and then stops Harbor and the Modal app.
# Before launching, the key and the account must hold the cap (phase2_money.py; the workspace budget is John's to check).
# Never stops a box. Heartbeat line every 5 minutes in /workspace/logs/general_tbench.txt.
set -uo pipefail
MODE=${1:-${MODE:-}}; [ -n "$MODE" ] || { sed -n 2,24p "$0"; exit 1; }
H=${H:-/workspace/forensics/gigi/3-forensics}; LOGD=${LOGD:-/workspace/logs}; S=${S:-$LOGD/general_tbench.txt}
ROOT=${GT_ROOT:-/workspace/general_tbench}; VENV=$ROOT/venv; TB=$ROOT/terminal-bench-2
TB_COMMIT=2fd12b88aafdd04a52c298e3940bcb189f9766d6      # harbor-framework/terminal-bench-2, 29 Apr 2026 (89 tasks)
HARBOR_COMMIT=39c2219a41b3caa7a689cef1882263b5a93ae93f  # harbor-framework/harbor 0.27.0, 23 Nov 2026 (the code read for the design)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/conda/bin:$PATH"
export MODAL_PROFILE=${MODAL_PROFILE:-spar} HARBOR_TELEMETRY=0 PYTHONPATH="$H/scripts${PYTHONPATH:+:$PYTHONPATH}"
export GT_TAG=${GT_TAG:-deepseek-v4-pro} GT_MAX_STEPS=${GT_MAX_STEPS:-100} GT_COMMAND_TIMEOUT=${GT_COMMAND_TIMEOUT:-600}
export GT_BENCH_COMMIT=$TB_COMMIT GT_HARBOR_COMMIT=$HARBOR_COMMIT
APP=${GT_MODAL_APP:-gigi-forensics-tbench}; N=${N:-8}; TIMEOUT_MULT=${TIMEOUT_MULT:-4}; MAX_ROUNDS=${MAX_ROUNDS:-3}
PILOT_TASKS=${PILOT_TASKS:-fix-git,cobol-modernization,regex-log,sqlite-with-gcov,openssl-selfsigned-cert,pytorch-model-cli,build-pmars,qemu-startup,compile-compcert,feal-differential-cryptanalysis,mcmc-sampling-stan,hf-model-inference}
mkdir -p "$LOGD" "$ROOT"
say () { echo "$(date -u +%FT%TZ) $*" | tee -a "$S"; }
stop () { say "GENERAL TBENCH STOPPED ($MODE): $*"; exit 1; }
py () { "$VENV/bin/python" "$@"; }
teacher () {   # MODEL and PIN from the same table as the 28-task configs (make_target_configs.TARGETS)
  MODEL=$(cd "$H/scripts" && "$VENV/bin/python" -c "import general_tbench_agent as A; s = A.settings(); print(s['model'], s['pin'])") || stop "could not read the teacher's settings"
  PIN=${MODEL##* }; MODEL=${MODEL%% *}
}

setup () {
  command -v uv > /dev/null || stop "uv is not on this box"
  [ -x "$VENV/bin/harbor" ] || { uv venv -q --python 3.12 "$VENV" && uv pip install -q --python "$VENV/bin/python" \
      "harbor[modal] @ git+https://github.com/harbor-framework/harbor@$HARBOR_COMMIT"; } || stop "could not install Harbor"
  if [ ! -d "$TB/.git" ]; then
    git clone -q https://github.com/harbor-framework/terminal-bench-2 "$TB" && git -C "$TB" checkout -q "$TB_COMMIT" || stop "could not fetch the benchmark"
  fi
  [ "$(git -C "$TB" rev-parse HEAD)" = "$TB_COMMIT" ] || stop "the benchmark is not at the pinned commit"
  n=$(ls "$TB"/*/task.toml | wc -l); [ "$n" = 89 ] || stop "$n tasks found, expected 89"
  say "setup ok: Harbor $("$VENV/bin/harbor" --version 2>/dev/null | tail -n 1), benchmark at ${TB_COMMIT:0:7} with $n tasks"
}

check () {   # free
  teacher
  py "$H/scripts/general_tbench_selftest.py" || stop "self-test failed"
  py "$H/scripts/general_tbench_agent.py" --dry-run --task-dir "$TB/fix-git" > "$LOGD/general_tbench_first_request.txt" || stop "dry run failed"
  grep -q "equal configs/.*: True" "$LOGD/general_tbench_first_request.txt" || stop "the request differs from the 28-task DeepSeek config (see $LOGD/general_tbench_first_request.txt)"
  (cd "$ROOT" && GT_SPEND_CAP=1 GT_LEDGER="$ROOT/dry_ledger.jsonl" "$VENV/bin/harbor" run --dry-run -y -p "$TB" \
      -a general_tbench_agent:ForensicsShellAgent -m "$MODEL" -e modal --ek "app_name=$APP" -k 1 -n 1 -o "$ROOT/dry_jobs" --job-name dry) \
      > "$LOGD/general_tbench_harbor_dry.txt" 2>&1 || stop "Harbor's dry run failed ($LOGD/general_tbench_harbor_dry.txt)"
  say "check ok: self-test passed; first request in $LOGD/general_tbench_first_request.txt; $(tail -n 1 "$LOGD/general_tbench_harbor_dry.txt")"
}

harbor_job () {   # harbor_job JOB_NAME AGENT K TASKS(comma or empty) : one Harbor job in the foreground, watched
  local name=$1 agent=$2 k=$3 tasks=$4 inc=() t hp
  for t in ${tasks//,/ }; do inc+=(-i "$t"); done
  local model=(); [ "$agent" = oracle ] || model=(-m "$MODEL" --agent-timeout-multiplier "$TIMEOUT_MULT")
  [ -e "$JOBS/$name" ] && stop "$JOBS/$name exists already: refusing to write into an old job folder"
  (cd "$ROOT" && exec "$VENV/bin/harbor" run -y -q -p "$TB" -a "$agent" "${model[@]}" -e modal --ek "app_name=$APP" \
      -k "$k" -n "$N" "${inc[@]}" -o "$JOBS" --job-name "$name") >> "$LOGD/general_tbench_$name.log" 2>&1 < /dev/null &
  hp=$!
  while kill -0 $hp 2> /dev/null; do
    for _ in $(seq 30); do kill -0 $hp 2> /dev/null || break; sleep 10; done
    line=$(py "$H/scripts/general_tbench_plan.py" status "$JOBS" --ledger "$GT_LEDGER" --cap "$GT_SPEND_CAP"); rc=$?
    say "$STAGE/$name: $line"
    if [ $rc = 2 ]; then
      say "SPEND CAP reached: stopping Harbor and the Modal app $APP. Ask GG before anything else."
      kill -INT $hp 2> /dev/null; sleep 60; kill -TERM $hp 2> /dev/null
      "$VENV/bin/modal" app stop "$APP" >> "$LOGD/general_tbench_stop.log" 2>&1
      return 2
    fi
  done
  wait $hp; return 0
}

paid_preflight () {   # the key, Modal, the money, the free checks
  [ -x "$VENV/bin/harbor" ] && [ -d "$TB" ] || setup
  check
  if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f /workspace/.env ]; then   # only this one key, never printed; the box's other
    OPENROUTER_API_KEY=$(set -a; . /workspace/.env > /dev/null 2>&1; printf %s "${OPENROUTER_API_KEY:-}")   # keys stay out of Harbor's environment
  fi
  export OPENROUTER_API_KEY
  [ -n "${OPENROUTER_API_KEY:-}" ] || stop "no OPENROUTER_API_KEY on this box (/workspace/.env): the model is called from here"
  "$VENV/bin/modal" profile current > "$LOGD/general_tbench_modal.txt" 2>&1 || stop "Modal profile not working (~/.modal.toml)"
  python3 "$H/scripts/phase2_money.py" --need "$GT_SPEND_CAP" >> "$S" 2>&1; rc=$?
  [ $rc = 2 ] && stop "the key or the account holds less than the cap \$$GT_SPEND_CAP: ask GG / John"
  up=$( . "$H/scripts/phase2_lib.sh" > /dev/null 2>&1; pin_uptime "$MODEL" "$PIN" )
  [ "${up:--1}" -ge 0 ] && [ "$up" -lt 80 ] && stop "$PIN was up only $up% of the last 5 minutes on OpenRouter: try later"
  return 0
}

rounds () {   # rounds TARGET TASKS : launch what is missing, at most MAX_ROUNDS times
  local target=$1 tasks=$2 r k list first
  for r in $(seq "$MAX_ROUNDS"); do
    first=$(ls -d "$JOBS"/"${STAGE}"_r* 2> /dev/null | wc -l)
    py "$H/scripts/general_tbench_plan.py" plan "$JOBS" --tasks-dir "$TB" --target "$target" ${tasks:+--tasks "$tasks"} ${EXCLUDE:+--exclude "$EXCLUDE"} > "$LOGD/general_tbench_${STAGE}_plan.txt" 2>> "$S" || stop "planning failed"
    [ -s "$LOGD/general_tbench_${STAGE}_plan.txt" ] || { say "$STAGE: every task has $target complete runs"; return 0; }
    g=0
    while read -r k list; do
      g=$((g + 1)); say "$STAGE round $((first + 1)).$g: $k runs each of $(echo "$list" | wc -w) tasks, $N at once"
      harbor_job "${STAGE}_r$((first + 1))_$g" general_tbench_agent:ForensicsShellAgent "$k" "${list// /,}" || return 2
    done < "$LOGD/general_tbench_${STAGE}_plan.txt"
  done
  say "$STAGE: $MAX_ROUNDS rounds used; tasks still short are listed by the plan command"
}

report () {
  py "$H/scripts/general_tbench_convert.py" table "$JOBS" > "$LOGD/general_tbench_${STAGE}_table.txt" 2>&1
  cat "$LOGD/general_tbench_${STAGE}_table.txt" >> "$S"
  mkdir -p "$H/results/logs/general_tbench"; cp "$LOGD/general_tbench_${STAGE}_table.txt" "$GT_LEDGER" "$H/results/logs/general_tbench/" 2> /dev/null
}

case "$MODE" in
  setup) setup ;;
  check) [ -x "$VENV/bin/harbor" ] && [ -d "$TB" ] || setup; check ;;
  oracle)
    STAGE=${STAGE:-gt_oracle}; JOBS=$H/results/raw/general_tbench/$STAGE; export GT_LEDGER=$LOGD/general_tbench_${STAGE}_ledger.jsonl GT_SPEND_CAP=1
    [ -x "$VENV/bin/harbor" ] && [ -d "$TB" ] || setup
    teacher
    "$VENV/bin/modal" profile current > "$LOGD/general_tbench_modal.txt" 2>&1 || stop "Modal profile not working (~/.modal.toml)"
    say "START oracle (no model): reference solutions on ${ORACLE_TASKS:-all 89 tasks}, $N at once"
    harbor_job "${STAGE}_$(date -u +%m%d%H%M)" oracle 1 "${ORACLE_TASKS:-}"
    say "ORACLE DONE: $(tail -n 25 "$LOGD"/general_tbench_${STAGE}_*.log | grep -E "Trials|Exceptions|Mean|reward" | tr '\n' ' ')" ;;
  smoke|pilot|full)
    : "${SPEND_CAP:?set SPEND_CAP (dollars): no paid run without a hard cap}"
    STAGE=${STAGE:-gt_$MODE}; JOBS=$H/results/raw/general_tbench/$STAGE
    export GT_SPEND_CAP=$SPEND_CAP GT_LEDGER=$LOGD/general_tbench_${STAGE}_ledger.jsonl
    paid_preflight
    say "START $MODE: $MODEL pinned to $PIN, cap \$$SPEND_CAP, $N runs at once, step cap $GT_MAX_STEPS, command timeout ${GT_COMMAND_TIMEOUT}s, time limits x$TIMEOUT_MULT"
    case "$MODE" in
      smoke) MAX_ROUNDS=1 rounds 1 "${TASKS:-fix-git,regex-log}"
             py "$H/scripts/general_tbench_plan.py" smoke-check "$JOBS" --pin "$PIN" > "$LOGD/general_tbench_${STAGE}_check.txt" 2>&1; rc=$?
             cat "$LOGD/general_tbench_${STAGE}_check.txt" >> "$S"; report
             [ $rc = 0 ] && say "SMOKE DONE: passed" || stop "SMOKE CHECK FAILED ($LOGD/general_tbench_${STAGE}_check.txt): fix before the pilot" ;;
      pilot) MAX_ROUNDS=1 rounds "${PILOT_N:-2}" "${TASKS:-$PILOT_TASKS}"; report
             say "PILOT DONE: $(grep -E "cost:|ended:|solved" "$LOGD/general_tbench_${STAGE}_table.txt" | tr '\n' ';') Projection for 979 runs = 979 x the mean cost above." ;;
      full)  rounds "${TARGET:-11}" "${TASKS:-}"; report
             say "FULL DONE: $(head -n 1 "$LOGD/general_tbench_${STAGE}_table.txt"); $(grep -E "cost:" "$LOGD/general_tbench_${STAGE}_table.txt"). Launch box idle and still billing." ;;
    esac ;;
  *) stop "unknown mode $MODE (setup, check, oracle, smoke, pilot, full)" ;;
esac
