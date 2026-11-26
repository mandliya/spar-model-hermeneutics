#!/usr/bin/env bash
# The end of a lens box's run, without anyone watching: once run.sh's upload_final phase has passed (the finished
# lens is on HF and size-checked), send everything else of the experiment and the box's logs to HF with the size
# check, and destroy the box if it passed.
#   (nohup setsid bash retire_after_upload.sh lens_box > /dev/null 2>&1 < /dev/null &)
# The argument is the label the box's logs get on HF (own_lens_<label>_...), so another box's logs of the same
# experiment are not replaced. If a step of the run failed, the box is left running: someone has to look.
set -uo pipefail
STATUS=/workspace/logs/own_lens_status.txt
say() { echo "$(date -u +%FT%TZ) retire_after_upload: $*" >> "$STATUS"; }
say "waiting for the upload_final phase"
until grep -q "phase upload_final done" "$STATUS"; do
  if grep -q "end fit_.* exit=[1-9]\|end upload_.* exit=[1-9]" "$STATUS"; then say "a step failed; the box is left running"; exit 1; fi
  sleep 60
done
bash "$(dirname "$0")/../classic_distillation/retire_box.sh" own_lens "$1" "the lens is on HF and size-checked; the box's logs are uploaded" \
  >> /workspace/logs/own_lens_retire_box.log 2>&1 || say "the final upload or its check failed; see own_lens_retire_box.log"
