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
# Uploads (general_tbench_upload.sh) every 2 hours while the collection runs and once at the end. An upload never stops the collection, and a
# stopped collection still gets its final upload. Never stops the box.
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/general_tbench_full_chain.sh > /workspace/logs/general_tbench_full_chain.out 2>&1 < /dev/null &
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; L=$H/scripts/general_tbench_launch.sh; LOGD=/workspace/logs; S=$LOGD/general_tbench.txt
R=model-hermeneutics/gigi-forensics-transcripts; HFD=general/terminal_bench
export PATH="/opt/conda/bin:$PATH"
say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }

upload () {   # 25 Nov: delegated to general_tbench_upload.sh (one tarball per finished run; the key scan lets through only
              # the fake keys the benchmark itself plants in sanitize-git-repo). The first version of this function
              # uploaded the raw folder as is and was refused on those fake keys at 05:14 and 07:14 UTC.
  bash "$H/scripts/general_tbench_upload.sh" once
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
