#!/usr/bin/env bash
# P("A") + A/B swap + mask-factual first-run recapture. One detached job.
#
# Stages are independent: a failure in one is logged and the next still runs
# (no `set -e`). Each script uploads its results to HF as it finishes each
# category. There is NO automatic stop: the box is stopped only when the user
# says so. The last line of the log says when it is idle.
cd /workspace/lies-experiment
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }

( while true; do sleep 600; say "heartbeat"; done ) &
HB=$!

say "=== 1/3 mask-factual first-run recapture ==="
python 08_mask_first_run_recapture.py && say "--- 1/3 OK ---" || say "!!! 1/3 FAILED (independent - continuing)"

say "=== 2/3 P(A), all rows, 11 A/B wordings ==="
python 06_answer_probability.py --categories all && say "--- 2/3 OK ---" || say "!!! 2/3 FAILED (independent - continuing)"

say "=== 3/3 A/B swap, 626-row subset, 11 A/B wordings ==="
python 07_ab_swap.py --categories all && say "--- 3/3 OK ---" || say "!!! 3/3 FAILED"

kill $HB 2>/dev/null
say "=== ALL STAGES FINISHED - box is now IDLE and STILL BILLING; stop it by hand ==="
