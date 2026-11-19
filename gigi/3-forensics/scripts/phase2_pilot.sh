#!/usr/bin/env bash
# phase2_pilot.sh: phase 2, stage 2 for one teacher (19 Nov 2026; phase2/harness_runbook.md sections 3-4). Needs GG's
# yes for the money (DeepSeek V4 Pro: ~$50-100). Runs ON THE LAUNCH BOX, detached, sleep-safe:
#   TAG=deepseek-v4-pro PIN=siliconflow PILOT_CAP=300 \
#     setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/phase2_pilot.sh > /workspace/logs/phase2_pilot.out 2>&1 < /dev/null &
# 1. Preflight: 28 configs for TAG, the harness's usage ledger present, Modal profile and secrets, OpenRouter money
#    (key limit_remaining and account credits >= PILOT_CAP; the workspace budget is John's to check).
# 2. Smoke, one run per kind of loop (memory harness-smoke-per-env-type): coding (submit script ends it), tictactoe (game
#    over), eval_tampering and lazy_investigation (plain final reply), secret_number (its own answer tool), grade_your_own
#    (500 steps: cost and context probe), ai_rd (its own capped agent key). Then phase2_smoke_check.py must pass
#    (finished, reasoning on every turn, a cost on every call, served by the pinned host). Else stop: nothing more runs.
# 3. Pilot: PILOT_N runs (default 2) of each of the 28 tasks, MAXC 2 per task.
# 4. Lean fetch, cost and length per run (run_usage.py, projected to the fleet's 46 / 40 runs per task), how runs ended
#    (classic_counts.py), upload to the private HF dataset, a "PILOT DONE" line. Never stops a box.
# Each run's wall clock: TIMEOUT_SHORT (caps <= 60 steps, default 3 h) / TIMEOUT_LONG (default 12 h): the pilot must not
# cut a run, it measures how long they take (Modal sandboxes allow 24 h).
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; LOGD=/workspace/logs; S=$LOGD/phase2.txt
. "$H/scripts/phase2_lib.sh"
set -a; [ -f /workspace/.env ] && . /workspace/.env; set +a   # HF_TOKEN, maybe OPENROUTER_API_KEY; never printed
TAG=${TAG:-deepseek-v4-pro}; PIN=${PIN:-siliconflow}; PILOT_N=${PILOT_N:-2}; PILOT_CAP=${PILOT_CAP:-300}
SHORT=${SHORT:-p2_$(echo "$TAG" | tr -cd 'a-z0-9' | cut -c1-10)}
SMOKE=${SHORT}_smoke; PILOT=${SHORT}_pilot
TIMEOUT_SHORT=${TIMEOUT_SHORT:-10800}; TIMEOUT_LONG=${TIMEOUT_LONG:-43200}
stop () { say "PILOT STOPPED ($TAG): $*"; exit 1; }

launch () {   # launch STAGE TASK N: one app, MAXC runs at once, the task's own secrets and wall clock
  local st=$1 e=$2 n=$3 cap sec to
  cap=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['agent']['max_steps'])" "$H/configs/${e}__${TAG}.yaml" 2>/dev/null \
        || upy -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['agent']['max_steps'])" "$H/configs/${e}__${TAG}.yaml")
  to=$TIMEOUT_LONG; [ "$cap" -le 60 ] && to=$TIMEOUT_SHORT
  sec=openrouter-secret; [ "$e" = ai_rd ] && sec=openrouter-secret,openrouter-agent   # ai_rd also calls its capped agent key
  MAXC=${MAXC:-2} GAP=12 TIMEOUT_S=$to MODAL_SECRETS=$sec bash "$H/scripts/modal_full.sh" "$st" "$TAG" "$e:$n"
}

# 1. preflight
say "START phase 2 pilot: $TAG pinned to $PIN, smoke then $PILOT_N runs per task, spend cap \$$PILOT_CAP"
n=$(ls "$H"/configs/*__"$TAG".yaml 2>/dev/null | wc -l); [ "$n" = 28 ] || stop "$n configs for $TAG, need 28 (scripts/make_target_configs.py)"
grep -q "USAGE_KEY" "$H/agent-interp-envs/src/agent_interp_envs/providers/openrouter_provider.py" || stop "the harness has no usage ledger (apply harness_patch)"
(cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal profile current) >> "$LOGD/phase2_preflight.log" 2>&1 || stop "Modal profile not working (copy ~/.modal.toml to the box)"
(cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal secret list) > "$LOGD/phase2_secrets.txt" 2>&1
for s in "openrouter-secret$MODAL_SECRET_SUFFIX" "openrouter-agent$MODAL_SECRET_SUFFIX"; do grep -q "$s" "$LOGD/phase2_secrets.txt" || stop "Modal secret $s missing"; done
python3 "$H/scripts/phase2_money.py" --need "$PILOT_CAP" --save-start "$MONEY_START" >> "$S" 2>&1; rc=$?
[ $rc = 2 ] && stop "not enough OpenRouter money for the cap (see the line above): ask GG / John"
[ $rc = 3 ] && say "no OpenRouter key on this box: spend is tracked from the launch logs only (check the key on the laptop first)"

# 2. smoke
for spec in coding:1 tictactoe:1 eval_tampering:1 lazy_investigation:1 secret_number:1 grade_your_own:1 ai_rd:1; do
  MAXC=1 launch "$SMOKE" "${spec%%:*}" "${spec##*:}"
done
say "smoke launched: 7 runs ($SMOKE)"
wait_stages "$SMOKE" "$PILOT_CAP" || stop "spend cap reached during the smoke"
fetch "$SMOKE"
upy ../scripts/phase2_smoke_check.py "$SMOKE" --pin "$PIN" --reasoning-required > "$LOGD/${SMOKE}_check.txt" 2>&1 \
  || stop "SMOKE CHECK FAILED ($LOGD/${SMOKE}_check.txt): fix before any pilot"
say "smoke passed: $(tail -n 1 "$LOGD/${SMOKE}_check.txt"); cost: $(upy ../scripts/run_usage.py "$SMOKE" | tail -n 1)"

# 3. pilot
for cfg in "$H"/configs/*__"$TAG".yaml; do
  e=$(basename "$cfg"); e=${e%%__*}
  while [ "$(live "$SMOKE,$PILOT")" -ge "${PILOT_LIVE_CAP:-40}" ]; do sleep 60; done   # pace launches
  MAXC=$PILOT_N launch "$PILOT" "$e" "$PILOT_N"
done
say "pilot launched: $PILOT_N runs x 28 tasks ($PILOT)"
wait_stages "$SMOKE,$PILOT" "$PILOT_CAP" || stop "spend cap reached during the pilot"

# 4. results
fetch "$PILOT"
upy ../scripts/run_usage.py "$SMOKE,$PILOT" --csv "results/${SHORT}_pilot_usage.csv" --project 46 > "$LOGD/${PILOT}_usage.txt" 2>&1
COUNTS_TAG=$TAG SET_ASIDE_FILE="$H/results/logs/$SHORT/set_aside.tsv" upy ../scripts/classic_counts.py "$PILOT" --target "$PILOT_N" > "$LOGD/${PILOT}_counts.txt" 2>&1
mkdir -p "$H/results/logs/$SHORT"; cp "$LOGD/${SMOKE}_check.txt" "$LOGD/${PILOT}_usage.txt" "$LOGD/${PILOT}_counts.txt" "$H/results/logs/$SHORT/" 2>/dev/null
for st in "$SMOKE" "$PILOT"; do upload_stage "$st" "phase2/$TAG/raw/$st"; done
say "PILOT DONE ($TAG): $(grep -E '^total spent' "$LOGD/${PILOT}_usage.txt"); $(grep -E '^projection' "$LOGD/${PILOT}_usage.txt"); ended: $(tail -n 1 "$LOGD/${PILOT}_counts.txt")"
