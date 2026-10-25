#!/usr/bin/env bash
# After scripts/fix_rerun.sh's runs finish: fetch, redact, score, re-check, redraw. Free (no API calls).
#   bash gigi/3-forensics/scripts/fix_finish.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
R="env -u VIRTUAL_ENV uv run"
echo "== fetch (A, B)"; $R python ../scripts/fix_fetch.py
echo "== sync (C)"; $R python ../scripts/modal_sync.py gigi-forensics-puppeteer gigi-forensics-revert-or-refactor | tail -5
echo "== redact the key from ai_rd outputs (main key in the agent's shell; README, deviation 7)"
FORENSICS_ENV_FILE="$HERE/../../.env" python3 ../scripts/redact_key.py "$HERE/results/raw/swap_all/ai_rd"   # ai_rd was not rerun (GPT-3.5 gone, 25 Oct); a no-op check
echo "== pre-commit per-run tables (old ones archived)"
mkdir -p "$HERE/results/archive/parser_cut_25oct"
for t in q1 q1m q2; do
  f="$HERE/results/runs_swap_text_precommit_$t.csv"
  [ -f "$f" ] && [ ! -f "$HERE/results/archive/parser_cut_25oct/$(basename "$f")" ] && mv "$f" "$HERE/results/archive/parser_cut_25oct/"
  $R python ../scripts/collect.py "swap_text/precommit_$t" | tail -1
done
echo "== score"; $R python ../scripts/full_runs_table.py | tail -3; $R python ../scripts/swap_all_score.py | tail -1
$R python ../scripts/gemma_endings.py | grep -E "ai_rd|puppeteer|revert"
echo "== validation pass again"; $R python ../scripts/swap_validity.py | tail -60
echo "== charts"; $R --with matplotlib python ../scripts/story_charts.py | tail -40
