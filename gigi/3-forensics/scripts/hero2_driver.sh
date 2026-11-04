#!/usr/bin/env bash
# Driver for the second pass of the gap-environment solo runs with the end-of-run confirmation on ("hero2", 4 Nov 2026,
# launched by scripts/hero2_launch_solo.sh: stages hero2_llama + hero2_llama_key, hero2_qwen + hero2_qwen_key,
# hero2_gemma + hero2_gemma_key). One detached, sleep-safe process (nohup caffeinate -i; every step logged to
# results/logs/hero2/driver.txt; no questions) that, for each model in turn, Llama first:
#   1. runs scripts/hero_solo_foldin.sh for the model's two hero2 stages (wait for our apps, fetch, redact, grade the
#      judged environments with GRADE_CAP / CEIL, score into results/hero2_full_runs__<model>.csv (+ _per_run), audit
#      into results/hero2_loss_audit__<model>.md, logs in results/logs/hero2/<model>/), and repeats the pass (it is
#      idempotent) until every stage of the model is launched, every fleet fetched and no step failed, or until
#      FOLDIN_PASSES passes / MODEL_HOURS hours are used up (then it moves on and says so);
#   2. after Llama's fold-in (complete or not), starts the swap chain on Llama's hero2 runs:
#      HERO2=1 PLAN_STAGES="hero2_llama hero2_llama_key" REQUIRE_ENVS=<the 8 gap environments> hero_swap_when_ready.sh,
#      which detaches itself, waits for the per-model files, builds results/hero2_swap_plan.json, checks the budget and
#      launches hero_swap.sh in its HERO2 mode (agent.end_confirm=true in every arm). Whatever the chain does (refuses
#      on the spend guard, fails, launches), the driver only logs it and carries on with Qwen and gemma;
#   3. ends with one summary line per model.
# Every fold-in pass and the chain run from the hero_solo_foldin.sh / hero_swap_when_ready.sh scripts unchanged in their
# defaults; nothing under results/ is deleted or overwritten except the per-model hero2 files, which each pass rewrites
# (results/full_runs.csv, hero_full_runs*.csv and hero_loss_audit*.md are never touched). Does NOT commit or push.
#   CEIL=<approved ceiling, $ of the key since 18 Oct> bash gigi/3-forensics/scripts/hero2_driver.sh     detaches, returns at once
#   DRY_RUN=1 bash gigi/3-forensics/scripts/hero2_driver.sh   foreground, no network, no Modal, no grading: one fold-in dry
#            pass per model into a scratch folder and the chain's dry run (USAGE_FAKE="<key left> <credits left>")
# Env: CEIL (required unless DRY_RUN=1), GRADE_CAP ($ per grading pass, default 60), HERO2_MODELS (space list, default
# "llama-3.3-70b qwen3-next-80b gemma-3-27b"), FOLDIN_PASSES (default 12), PASS_SLEEP (seconds between passes, default
# 900), MODEL_HOURS (per model, default 16), WAIT_MINUTES (per fold-in pass, hero_solo_foldin.sh's; default 840),
# REQUIRE_ENVS (the chain's; default the 8 gap environments), HERO2_NO_SWAP=1 (fold-ins only, no chain), plus everything
# hero_swap_when_ready.sh reads (ARMS, MIN_SCORED, SPEND_GUARD_USD, GUARD_FACTOR, ...).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DRY=${DRY_RUN:-0}
MODELS=${HERO2_MODELS:-llama-3.3-70b qwen3-next-80b gemma-3-27b}
PASSES=${FOLDIN_PASSES:-12}; PASS_SLEEP=${PASS_SLEEP:-900}; MODEL_HOURS=${MODEL_HOURS:-16}; GRADE_CAP=${GRADE_CAP:-60}
REQUIRE_ENVS=${REQUIRE_ENVS:-ai_rd,coding,coding_norules,dashboard_costly_failure,grade_your_own,optimize_rendering_engine,optimize_rendering_engine_graded,precommit_hook}
if [ "$DRY" = 1 ]; then
  SCR=$(mktemp -d "${TMPDIR:-/tmp}/hero2_driver_dry.XXXXXX"); L="$SCR/logs"
else
  : "${CEIL:?set CEIL to the approved spend ceiling (\$ of the key since 18 Oct 2026); there is no default}"
  L="$HERE/results/logs/hero2"
fi
mkdir -p "$L"; DRIVER="$L/driver.txt"
say () { echo "$(date '+%F %T') $*" >> "$DRIVER"; [ "$DRY" = 1 ] && echo "$*"; return 0; }
loud () { say "!!!!!!!!!! $*"; say "!!!!!!!!!! $*"; }

# ---- detach (real runs only): the child does the work under caffeinate, this shell returns at once ----
if [ "$DRY" != 1 ] && [ -z "${HERO2_DRIVER_CHILD:-}" ]; then
  if [ -f "$L/driver.pid" ] && kill -0 "$(cat "$L/driver.pid")" 2>/dev/null; then
    echo "hero2_driver: already running (pid $(cat "$L/driver.pid")); log $DRIVER"; exit 0
  fi
  HERO2_DRIVER_CHILD=1 nohup caffeinate -i bash "$0" "$@" >> "$L/driver_stdout.txt" 2>&1 &
  disown 2>/dev/null; echo "hero2_driver: detached (pid $!); log $DRIVER"; exit 0
fi
if [ "$DRY" != 1 ]; then
  if [ -f "$L/driver.pid" ] && [ "$(cat "$L/driver.pid")" != $$ ] && kill -0 "$(cat "$L/driver.pid")" 2>/dev/null; then
    say "another driver is running (pid $(cat "$L/driver.pid")): exiting"; exit 0
  fi
  echo $$ > "$L/driver.pid"; trap 'rm -f "$L/driver.pid"' EXIT
fi
say "start (models: $MODELS; ceiling ${CEIL:-none, dry run}, grade cap $GRADE_CAP; up to $PASSES passes / $MODEL_HOURS h per model$([ "$DRY" = 1 ] && echo "; DRY RUN, outputs in $SCR"))"

stages_of () {   # the two hero2 stages of a model tag
  case "$1" in
    llama-3.3-70b) echo hero2_llama,hero2_llama_key ;;
    qwen3-next-80b) echo hero2_qwen,hero2_qwen_key ;;
    gemma-3-27b) echo hero2_gemma,hero2_gemma_key ;;
    *) echo "" ;;
  esac
}

complete () {   # exit 0 when every stage of the model is launched and every fleet fetched after the apps stopped; prints what is missing
  python3 - "$HERE" "$1" "$2" <<'PY'
import pathlib, re, sys
H, model, stages = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
missing = []
for s in stages:
    d = H / "results" / "logs" / s
    if not d.is_dir():
        missing.append(f"{s}: not launched yet (no results/logs/{s}/)"); continue
    lt = d / "launch.txt"
    if not (lt.exists() and "all launched" in lt.read_text()):
        missing.append(f"{s}: launcher not finished (no 'all launched' in launch.txt)")
    for log in sorted(d.glob(f"*__{model}.log")):
        env = log.stem.split("__")[0]
        o = re.search(r"^Output: results/(\S+?)/?$", log.read_text(errors="replace"), re.M)
        if not o:
            missing.append(f"{s}/{env}: no Output line yet"); continue
        if not (H / "results" / "raw" / s / env / o.group(1) / ".fetched_after_apps_stopped").exists():
            missing.append(f"{s}/{env}: not fetched after the apps stopped")
print("; ".join(missing) if missing else f"complete: {', '.join(stages)} launched and fetched")
sys.exit(1 if missing else 0)
PY
}

foldin () {   # one model: fold-in passes until complete, or out of passes / hours
  local model=$1 stages pass start_line msg deadline rc failed
  stages=$(stages_of "$model")
  [ -n "$stages" ] || { loud "$model: no hero2 stages known for this model tag, skipped"; return 1; }
  deadline=$(( $(date +%s) + MODEL_HOURS * 3600 ))
  for pass in $(seq 1 "$PASSES"); do
    local i=0   # never two fold-ins at once (they would grade the same runs twice)
    while pgrep -f "scripts/hero_solo_foldin.sh" > /dev/null; do
      [ $i = 0 ] && say "$model: another hero_solo_foldin.sh is running, waiting for it"; i=$((i + 1)); sleep 60
    done
    say "$model: fold-in pass $pass (stages $stages)"
    if [ "$DRY" = 1 ]; then
      DRY_RUN=1 HERO_MODEL="$model" HERO_STAGES="$stages" HERO_OUT_DIR="$SCR/$model" HERO_LOG_DIR="$SCR/$model/logs" \
        bash "$HERE/scripts/hero_solo_foldin.sh" > "$L/foldin_${model}_pass$pass.out" 2>&1; rc=$?
      say "$model: dry pass exit $rc: $(grep -E "^(scored|audit)" "$L/foldin_${model}_pass$pass.out" | tr '\n' ';' | cut -c1-300)"
      msg=$(complete "$model" "$stages"); say "$model: $msg"; return 2
    fi
    local plog="$HERE/results/logs/hero2/$model/progress.txt"
    start_line=$([ -f "$plog" ] && wc -l < "$plog" | tr -d ' ' || echo 0)
    HERO_MODEL="$model" HERO_STAGES="$stages" CEIL="$CEIL" GRADE_CAP="$GRADE_CAP" \
      bash "$HERE/scripts/hero_solo_foldin.sh" > "$L/foldin_${model}_pass$pass.out" 2>&1; rc=$?
    failed=$(tail -n +$((start_line + 1)) "$plog" 2>/dev/null | grep -c "STEP FAILED")
    say "$model: pass $pass exit $rc, $failed failed step(s); $(tail -n +$((start_line + 1)) "$plog" 2>/dev/null | grep -E "scored:|audit:|grading:|STEP FAILED|still running" | tail -4 | cut -c21- | tr '\n' ';' | cut -c1-400)"
    if msg=$(complete "$model" "$stages") && [ "$failed" = 0 ]; then
      say "$model: fold-in done ($msg); files results/hero2_full_runs__$model.csv, results/hero2_loss_audit__$model.md"; return 0
    fi
    say "$model: not done yet: $msg$([ "$failed" != 0 ] && echo " (and $failed failed step(s): see $plog)")"
    [ "$(date +%s)" -lt "$deadline" ] || { loud "$model: $MODEL_HOURS h used up after $pass pass(es), moving on; rerun the driver later for what is missing"; return 1; }
    sleep "$PASS_SLEEP"
  done
  loud "$model: $PASSES passes used up, moving on; rerun the driver later for what is missing"; return 1
}

swap_chain () {   # Llama's hero2 swap: the chain detaches itself and waits on its own; the driver only records what it said
  [ "${HERO2_NO_SWAP:-0}" = 1 ] && { say "swap chain skipped (HERO2_NO_SWAP=1)"; return 0; }
  local out="$L/swap_chain_start_$(date '+%Y%m%d-%H%M%S').txt" rc
  if [ "$DRY" = 1 ]; then
    HERO2=1 DRY_RUN=1 PLAN_STAGES="hero2_llama hero2_llama_key" REQUIRE_ENVS="$REQUIRE_ENVS" HERO_SWAP_LOG_DIR="$SCR/chain" \
      USAGE_FAKE="${USAGE_FAKE:-300 500}" bash "$HERE/scripts/hero_swap_when_ready.sh" > "$out" 2>&1; rc=$?
  else
    HERO2=1 PLAN_STAGES="hero2_llama hero2_llama_key" REQUIRE_ENVS="$REQUIRE_ENVS" \
      bash "$HERE/scripts/hero_swap_when_ready.sh" > "$out" 2>&1; rc=$?
  fi
  say "swap chain (HERO2=1, plan stages hero2_llama hero2_llama_key, require $REQUIRE_ENVS): exit $rc: $(grep -vE '^\s*$' "$out" | tail -3 | tr '\n' ';' | cut -c1-500)"
  [ "$DRY" = 1 ] || say "swap chain log: $HERE/results/logs/hero2_swap/chain.txt (it waits, plans, checks the budget and launches by itself; a refusal there does not stop this driver)"
  return 0
}

declare -a SUMMARY=()
for m in $MODELS; do
  foldin "$m"; rc=$?
  case $rc in 0) SUMMARY+=("$m: fold-in complete") ;; 2) SUMMARY+=("$m: dry pass done") ;; *) SUMMARY+=("$m: fold-in INCOMPLETE (see $L/driver.txt)") ;; esac
  [ "$m" = llama-3.3-70b ] && swap_chain
done
say "summary: ${SUMMARY[*]}"
say "done (not committed: the parent session commits; charts: cd agent-interp-envs && uv run --with matplotlib python ../scripts/confirm_effect_charts.py)"
[ "$DRY" = 1 ] && echo "dry-run outputs in $SCR"
exit 0
