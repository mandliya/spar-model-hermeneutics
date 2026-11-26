#!/usr/bin/env bash
# Print where a training step is, from its log, in a few lines. Reads only.
#   bash show_progress.sh train        # or: trial
# Lines: the time; the reconstructor's rebuild quality per round (fve_nrm, on that round's fresh explanations); the
# writer's entropy and its distance from the starting writer (kl) for the last rounds; the last round's seconds; then
# failures, saves on the box, the status file's last line and GPU memory.
LOG=/workspace/logs/classic_nla_$1.log
date -u +"now %T UTC"
grep -o "step [0-9]*: {'train/loss': [0-9.e-]*, 'train/pred_norm_raw.*'train/fve_nrm': [0-9.]*" "$LOG" | sed "s/{.*'train\/fve_nrm': /fve /" | cut -c1-17 | tail -12 | tr '\n' ';'; echo
grep -o "step [0-9]*: {'train/loss': [0-9.e-]*, 'train/pg_loss.*'train/kl_loss': [0-9.]*" "$LOG" | sed "s/{.*'train\/entropy_loss': /entropy /; s/, 'train\/pg_clipfrac.*'train\/kl_loss': / kl /" | cut -c1-42 | tail -3 | tr '\n' ';'; echo
grep -o "perf [0-9]*: {'perf/update_weights.*'perf/step_time': [0-9]*" "$LOG" | sed "s/{.*'perf\/step_time': /seconds /" | tail -1
echo "failures: $(grep -c "Traceback\|out of memory" "$LOG") | saves on the box: $(ls /workspace/nla/runs/gemma-classic-step92/actor 2> /dev/null | grep iter_ | tr '\n' ' ')"
tail -1 /workspace/logs/classic_nla_status.txt | cut -c1-150
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
