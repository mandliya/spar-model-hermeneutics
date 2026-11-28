#!/usr/bin/env bash
# Start phases of run.sh once other steps have ended, so a GPU goes straight from one job to the next.
#   after.sh "<text of a status line>" ["<another>" ...] -- <run.sh phases...>
# Waits until every text has appeared in /workspace/logs/gender_secret_rerun_status.txt (e.g. "end curriculum_shard_2
# exit"), then runs the phases. The environment (TRAIN_GPU, ...) is passed on.
S=/workspace/logs/gender_secret_rerun_status.txt
while [ "$1" != -- ]; do
  until grep -qF "$1" "$S"; do sleep 10; done
  shift
done
shift
exec bash "$(dirname "$0")/run.sh" "$@"
