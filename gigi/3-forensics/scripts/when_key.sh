#!/usr/bin/env bash
# The two environments that put an OpenRouter key in the agent's own shell (ai_rd: its scorer calls OpenRouter;
# swe_rebench_v2: the agent's shell runs as root) run only with a separate, capped key. This waits for the user to
# write it into the repo's .env (AGENT_OPENROUTER_API_KEY=..., gitignored), stores it as the Modal secret
# openrouter-agent-gigi (never echoed), then launches ai_rd (full runs, 10 per model, run_modal.py) and
# swe_rebench_v2 (10 per model, scripts/swe_modal.py), both on Modal, detached. Spend is bounded by the key's cap.
# Log: results/logs/full_key/run.txt.
#   nohup caffeinate -ims bash gigi/3-forensics/scripts/when_key.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVF="$HERE/../../.env"; LOG="$HERE/results/logs/full_key"; mkdir -p "$LOG"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
say () { echo "$(date '+%F %T') $*" >> "$LOG/run.txt"; }
say "waiting for AGENT_OPENROUTER_API_KEY in .env"
until grep -q '^AGENT_OPENROUTER_API_KEY=sk-or-' "$ENVF" 2>/dev/null; do sleep 60; done
say "key found"
( set -a; . "$ENVF"; set +a; cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal secret create --force \
    openrouter-agent-gigi OPENROUTER_API_KEY="$AGENT_OPENROUTER_API_KEY" > /dev/null 2>&1 ) \
  && say "Modal secret openrouter-agent-gigi stored" || { say "FAILED to store the Modal secret"; exit 1; }
cd "$HERE"
MODAL_SECRETS=openrouter-agent QWEN="ai_rd:10" GEMMA="ai_rd:10" bash scripts/modal_rest.sh full_key && say "ai_rd launched"
for m in qwen3-next-80b gemma-3-27b; do
  ( cd "$HERE/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
      --config "../configs/swe_rebench_v2__$m.yaml" --count 10 --stage full_key > "$LOG/swe_rebench_v2__$m.log" 2>&1 & )
  sleep 30
done
say "swe_rebench_v2 launched"
say "done"
