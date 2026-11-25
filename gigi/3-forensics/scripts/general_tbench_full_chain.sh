#!/usr/bin/env bash
# general_tbench_full_chain.sh (25 Nov 2026): GG's yes (25 Nov) for the full Terminal-Bench collection, ~$175
# (~$114 OpenRouter at 100 steps + ~$60 Modal, from the 24-run pilot), with a hard OpenRouter cap of $500.
#   - 11 complete runs of each of 87 tasks: all 89 minus qemu-startup and qemu-alpine-ssh, which start a virtual
#     machine and fail on Modal even with the reference solution (oracle check, 24 Nov). GG agreed to drop them.
#   - Step cap 100 (GG, 25 Nov; 50 would have saved ~$26 and cut about 1 run in 5).
#   - The 7 tasks that share a theme with our 28 are collected as well: GG asked for two training sets, one without
#     them (80 tasks) and one with them (87 tasks), built from this one collection (TERMINAL_BENCH.md).
#   - The runs go to the PRIVATE dataset model-hermeneutics/gigi-forensics-transcripts (GG, 25 Nov), never to anything
#     public: the benchmark asks not to be trained on publicly (its canary line).
# Uploads every 2 hours while the collection runs and once at the end. An upload never stops the collection, and a
# stopped collection still gets its final upload. Never stops the box.
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/general_tbench_full_chain.sh > /workspace/logs/general_tbench_full_chain.out 2>&1 < /dev/null &
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; L=$H/scripts/general_tbench_launch.sh; LOGD=/workspace/logs; S=$LOGD/general_tbench.txt
R=model-hermeneutics/gigi-forensics-transcripts; HFD=general/terminal_bench
export PATH="/opt/conda/bin:$PATH"
say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }

upload () {   # every stage folder (oracle, smoke, pilot, full) and the ledgers/tables; refused if any file holds a key shape
  local d=$H/results/raw/general_tbench lg=$H/results/logs/general_tbench
  if grep -rlE 'sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|fw_[A-Za-z0-9]{20,}' "$d" "$lg" > "$LOGD/general_tbench_keyscan.txt" 2> /dev/null; then
    say "UPLOAD REFUSED: a key shape was found ($(wc -l < "$LOGD/general_tbench_keyscan.txt") files, names in general_tbench_keyscan.txt); nothing uploaded"; return 1
  fi
  # HF_TOKEN only in this subshell: the collection itself never sees the box's other keys
  ( set -a; . /workspace/.env > /dev/null 2>&1; set +a
    hf upload "$R" "$d" "$HFD/raw" --repo-type dataset --commit-message "Terminal-Bench runs (DeepSeek V4 Pro), $(date -u +%FT%TZ)" \
      && { [ ! -d "$lg" ] || hf upload "$R" "$lg" "$HFD/logs" --repo-type dataset --commit-message "Terminal-Bench ledgers and tables"; } ) \
    >> "$LOGD/general_tbench_upload.log" 2>&1 \
    && say "uploaded the Terminal-Bench runs to HF (private, $HFD/raw)" || say "UPLOAD FAILED for the Terminal-Bench runs (general_tbench_upload.log); the collection goes on"
}

say "CHAIN full: start (GG's yes 25 Nov: ~\$175, OpenRouter cap \$500; 87 tasks x 11 runs; step cap 100; uploads every 2 h)"
env SPEND_CAP=500 N=32 TARGET=11 EXCLUDE=qemu-startup,qemu-alpine-ssh bash "$L" full > "$LOGD/general_tbench_full.out" 2>&1 < /dev/null &
FP=$!
( while sleep 7200; do kill -0 "$FP" 2> /dev/null || exit 0; upload; done ) &
UP=$!
wait "$FP"; rc=$?
kill "$UP" 2> /dev/null
say "CHAIN full: exit $rc"
upload
say "CHAIN FULL $([ $rc = 0 ] && echo DONE || echo STOPPED) (exit $rc). Launch box idle for Terminal-Bench and still billing."
