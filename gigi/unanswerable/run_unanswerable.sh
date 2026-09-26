#!/usr/bin/env bash
# unanswerable-claims, end to end, on the GPU box.
#
# NOT chained with `set -e`. These three stages have a real dependency chain
# (battery needs the primary run's answers), so a genuine failure must stop
# the chain - but it stops LOUDLY, with the reason in the log, rather than
# dying silently and leaving nothing to show for hours of runtime.
#
# Stage 1 (question labelling) is deliberately absent: it needs no GPU, it is
# already done, and its output is committed.
#
# Usage on the box:
#   cd /workspace/lies-experiment
#   nohup bash run_unanswerable.sh > results/unanswerable_run.log 2>&1 & disown

set -o pipefail
cd "$(dirname "$0")"

stamp () { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

say () { echo "[$(stamp)] $*"; }

require () {
  if [ ! -f "$1" ]; then
    say "!!! ABORT: required input missing: $1"
    exit 1
  fi
}

say "=== unanswerable-claims run starting ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

require results/unanswerable_question_labels.csv

# --- stage 2+3: capability claim + fresh-context behaviour ------------------
say "--- STAGE A: primary run (claim + fresh-context behaviour) ---"
if python 00_unanswerable_claims.py; then
  say "--- STAGE A OK ---"
else
  say "!!! STAGE A FAILED — stopping. The battery depends on its output."
  say "!!! Partial results are in results/unanswerable_claims.jsonl and the"
  say "!!! run is resumable: rerunning skips every row already done."
  exit 1
fi

# --- the gate: are both arms big enough to support a rate? ------------------
say "--- GATE: checking arm sizes before spending GPU on 1,200 battery rows ---"
if python 02_analyse_unanswerable.py --gate; then
  say "--- GATE PASSED ---"
else
  say "!!! GATE FAILED: one arm is too thin for the self-report metrics."
  say "!!! Deliberately NOT running the battery. The primary confabulation"
  say "!!! rate above is still a valid result on its own."
  exit 2
fi

# --- stage 4: the 12-condition battery -------------------------------------
say "--- STAGE B: 12-condition self-report battery ---"
if python 00_unanswerable_selfreport_benchmark.py; then
  say "--- STAGE B OK ---"
else
  say "!!! STAGE B FAILED — partial results kept, run is resumable."
  exit 1
fi

# --- final report ----------------------------------------------------------
say "--- ANALYSIS ---"
python 02_analyse_unanswerable.py || say "!!! analysis failed (data is safe)"

N_PD=$(ls results/activations/*__predecision.pt 2>/dev/null | wc -l | tr -d ' ')
N_AT=$(ls results/activations/*__answer_token.pt 2>/dev/null | wc -l | tr -d ' ')
say "activations on disk: predecision=$N_PD answer_token=$N_AT"

say "=== unanswerable-claims run COMPLETE ==="
