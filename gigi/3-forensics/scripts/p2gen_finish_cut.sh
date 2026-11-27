#!/usr/bin/env bash
# p2gen_finish_cut.sh (27 Nov 2026), ON THE LAUNCH BOX: finish a training set that the Modal workspace's spend limit cut
# on 26 Nov at 10:29 UTC (Modal stopped every app: the training before its last copy, the tests of the copies being
# run). Three steps, all through p2gen_launch.sh:
#   1. training again: it resumes from the last complete saved copy (the smoke test is skipped: SMOKE_OK is there);
#   2. at the same time, the cut copies are tested again under new names (copy:name, e.g. c46:c46r), because
#      modal_full.sh skips tasks whose launch log exists; move the cut stages out of results/raw/ first;
#   3. when the training says TRAINING DONE, every copy it saved since is tested too.
# Each retest ends with the scores, the table and the upload. Spends Modal money: needs P2GEN_YES=<dv> (GG, 27 Nov:
# "just distill, disregard the limit").
#   P2GEN_YES=p2gen_tbench80 setsid nohup bash scripts/p2gen_finish_cut.sh p2gen_tbench80 c46:c46r c69:c69r \
#     > /workspace/logs/p2gen_tbench80_finish.out 2>&1 < /dev/null &
set -uo pipefail
DV=${1:?training set name}; shift
H=${H:-$(cd "$(dirname "$0")/.." && pwd)}; export H
LOGD=${LOGD:-/workspace/logs}; S=$LOGD/$DV.txt
. "$H/scripts/phase2_lib.sh"                     # say, Modal profile and names
[ "${P2GEN_YES:-}" = "$DV" ] || { echo "spends Modal money: set P2GEN_YES=$DV"; exit 3; }
VOL=${MODAL_VOLUME_PREFIX}p2gen
m () { (cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal "$@"); }
vcat () { m volume get "$VOL" "$1" - 2>/dev/null; }
copies () { vcat "out/$DV/copies_ready.txt" | grep -oE '^c[0-9]+$' | sort -u; }

before=$(copies)
say "finish: $DV has copies $(echo $before); resuming the training and re-testing $*"
bash "$H/scripts/p2gen_launch.sh" train "$DV"
[ $# -gt 0 ] && { bash "$H/scripts/p2gen_launch.sh" retest "$DV" "$@" & }
t=0
until vcat "logs/$DV/stages.txt" | tail -n 5 | grep -qE "TRAINING DONE|TRAINING FAILED"; do
  t=$((t + 1)); [ $t -gt 72 ] && { say "finish: $DV: no TRAINING DONE after 6 h; stopping here (re-tests of the cut copies go on)"; wait; exit 1; }
  sleep 300
done
new=$(comm -13 <(echo "$before") <(copies) | tr '\n' ' ')
if [ -n "${new// /}" ]; then
  say "finish: $DV training done; testing the new copies: $new"
  bash "$H/scripts/p2gen_launch.sh" retest "$DV" $new
else
  say "finish: $DV: the training saved no new copy ($(vcat "logs/$DV/stages.txt" | tail -n 1 | cut -c1-160))"
fi
wait
say "P2GEN FINISH DONE $DV"
