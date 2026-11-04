#!/usr/bin/env bash
# Second pass of the solo runs in the 8 gap environments, with the end-of-run confirmation on (agent.end_confirm=true,
# patches/end_confirm.patch), for every model in the hero test: Llama 3.3 70B (fp16, the target), Qwen3-Next-80B and
# gemma-3-27b (both fp8, as in every earlier result). Same run counts as the first pass. Each model's ai_rd runs go with
# the capped agent key (Modal secret openrouter-agent-gigi). All runs start at once and run detached on Modal.
#   nohup caffeinate -i bash gigi/3-forensics/scripts/hero2_launch_solo.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
export EXTRA_OV="agent.end_confirm=true" MAXC=0 GAP=${GAP:-12}
ENVS="coding:20 coding_norules:20 precommit_hook:20 dashboard_costly_failure:10 grade_your_own:5 optimize_rendering_engine:5 optimize_rendering_engine_graded:5"
mkdir -p results/logs/hero2; say () { echo "$(date '+%F %T') $*" >> results/logs/hero2/launch_solo.txt; }
for m in llama-3.3-70b qwen3-next-80b gemma-3-27b; do
  tag=${m%%-*}; [ "$m" = llama-3.3-70b ] && tag=llama; [ "$m" = qwen3-next-80b ] && tag=qwen; [ "$m" = gemma-3-27b ] && tag=gemma
  say "launching $m: $ENVS"
  bash scripts/modal_full.sh hero2_${tag} "$m" $ENVS
  MODAL_SECRETS=openrouter-agent bash scripts/modal_full.sh hero2_${tag}_key "$m" ai_rd:10
  say "launched $m"
done
say "all three models launched"
