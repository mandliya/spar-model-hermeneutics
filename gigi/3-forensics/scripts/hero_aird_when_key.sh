#!/usr/bin/env bash
# ai_rd for the hero target (Llama 3.3 70B) needs a separate, capped OpenRouter key: its scorer calls OpenRouter from
# inside the agent's shell, so the main key must never go there. This waits until the user writes the capped key into
# the repo's .env (AGENT_OPENROUTER_API_KEY=sk-or-..., gitignored), stores it as the Modal secret openrouter-agent-gigi
# (never echoed), then launches the 10 ai_rd runs, all at once and detached, as stage hero_solo_key2.
#   nohup caffeinate -i bash gigi/3-forensics/scripts/hero_aird_when_key.sh &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVF="$HERE/../../.env"; LOG="$HERE/results/logs/hero_solo_key2"; mkdir -p "$LOG"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
say () { echo "$(date '+%F %T') $*" >> "$LOG/run.txt"; }
say "waiting for AGENT_OPENROUTER_API_KEY in .env"
until grep -q '^AGENT_OPENROUTER_API_KEY=sk-or-' "$ENVF" 2>/dev/null; do sleep 60; done
say "key found"
( set -a; . "$ENVF"; set +a; cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal secret create --force \
    openrouter-agent-gigi OPENROUTER_API_KEY="$AGENT_OPENROUTER_API_KEY" > /dev/null 2>&1 ) \
  && say "Modal secret openrouter-agent-gigi stored" || { say "FAILED to store the Modal secret"; exit 1; }
cd "$HERE" && MAXC=0 GAP=10 MODAL_SECRETS=openrouter-agent bash scripts/modal_full.sh hero_solo_key2 llama-3.3-70b ai_rd:10 && say "ai_rd launched"
say "done"
