#!/usr/bin/env bash
# Llama's last try (7 Nov 2026, John's call of 6 Nov): the 8 main tasks again, same counts as hero2_llama, with the hard
# end rule on (agent.end_task_required=true, patches/end_task_required.patch: a reply without a tool call never ends the
# run, only the end_task tool does; at most 10 reminders per run, then the step cap). Llama 3.3 70B fp16 on CoreWeave,
# as in every hero run. Stages hero4_llama (7 tasks) and hero4_llama_key (ai_rd with the capped agent key). Detached on
# Modal; the laptop only has to stay awake for the ~2 minutes of launching.
#   nohup caffeinate -i bash gigi/3-forensics/scripts/hero4_launch_llama.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
export EXTRA_OV="agent.end_task_required=true,agent.end_task_reprompts=10" MAXC=0 GAP=${GAP:-12}
ENVS="coding:20 coding_norules:20 precommit_hook:20 dashboard_costly_failure:10 grade_your_own:5 optimize_rendering_engine:5 optimize_rendering_engine_graded:5"
mkdir -p results/logs/hero4; say () { echo "$(date '+%F %T') $*" >> results/logs/hero4/launch.txt; }
say "launching llama-3.3-70b with the hard end rule: $ENVS + ai_rd:10"
bash scripts/modal_full.sh hero4_llama llama-3.3-70b $ENVS
MODAL_SECRETS=openrouter-agent bash scripts/modal_full.sh hero4_llama_key llama-3.3-70b ai_rd:10
say "launched"
