#!/usr/bin/env bash
# general_tbench_upload.sh (25 Nov 2026): the Terminal-Bench runs to the PRIVATE dataset
# model-hermeneutics/gigi-forensics-transcripts, under general/terminal_bench/ (GG's yes, 25 Nov). Never anything public:
# the benchmark asks not to be trained on publicly (its canary line).
#
# One .tar.gz per FINISHED run (a run folder with result.json), made once and never remade, so each upload only adds the
# runs finished since the last one. Raw folders hold one file per model call (27,014 files after 440 runs on 25 Nov),
# too many for one HF commit; ~1,000 packed runs are not.
#   general/terminal_bench/raw/<stage>/<job>/<run>.tar.gz   one run, as saved under results/raw/general_tbench/
#   general/terminal_bench/raw/<stage>/<job>/<file>          the job's own small files (config, result, log)
#   general/terminal_bench/logs/                             ledgers and tables (results/logs/general_tbench/)
#
# Key scan before every upload: any key-shaped string refuses the whole upload, EXCEPT the strings the benchmark itself
# plants. sanitize-git-repo asks the model to remove fake keys from a repository, so its runs quote 2 fake hf_ tokens;
# both are in the public benchmark files and neither is a key on this box (checked 25 Nov). The 2-hourly uploads of
# general_tbench_full_chain.sh were refused on them at 05:14 and 07:14 UTC; this script replaces that upload.
#   bash general_tbench_upload.sh once           # one upload now
#   bash general_tbench_upload.sh follow PID     # now, every 2 hours while PID lives, and once more after it ends
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; LOGD=/workspace/logs; S=$LOGD/general_tbench.txt
TB=/workspace/general_tbench/terminal-bench-2; RAW=$H/results/raw/general_tbench; LG=$H/results/logs/general_tbench
STAGE_DIR=/workspace/general_tbench_hf/raw; R=model-hermeneutics/gigi-forensics-transcripts; HFD=general/terminal_bench
export PATH="/opt/conda/bin:$PATH"
say () { echo "$(date -u +%FT%TZ) TB UPLOADER: $*" >> "$S"; }

scan () {   # 0 = no key shape apart from the benchmark's planted ones; 1 = refuse (file names only in general_tbench_keyscan.txt)
  python3 - "$TB" "$LOGD/general_tbench_keyscan.txt" "$RAW" "$LG" <<'PY'
import pathlib, re, sys
pat = re.compile(r"sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|fw_[A-Za-z0-9]{20,}")
def found(p):
    try: return set(pat.findall(p.read_text(errors="ignore")))
    except Exception: return set()
planted = set()
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file(): planted |= found(p)
bad = [str(p) for root in sys.argv[3:] if pathlib.Path(root).exists()
       for p in pathlib.Path(root).rglob("*") if p.is_file() and found(p) - planted]
pathlib.Path(sys.argv[2]).write_text("".join(b + "\n" for b in bad))
sys.exit(1 if bad else 0)
PY
}

pack () {   # one tarball per finished run, made once; the job's own files copied as they are
  local job run out n=0
  for job in "$RAW"/*/*/; do
    job=${job%/}; out=$STAGE_DIR/${job#$RAW/}; mkdir -p "$out"
    find "$job" -maxdepth 1 -type f -exec cp -p {} "$out/" \;
    for run in "$job"/*/; do
      run=${run%/}
      [ -f "$run/result.json" ] && [ ! -f "$out/$(basename "$run").tar.gz" ] || continue
      tar -czf "$out/.$(basename "$run").tar.gz.part" -C "$job" "$(basename "$run")" \
        && mv "$out/.$(basename "$run").tar.gz.part" "$out/$(basename "$run").tar.gz" && n=$((n + 1))
    done
  done
  echo "$n"
}

upload () {
  if ! scan; then
    say "UPLOAD REFUSED: a key shape that is not one of the benchmark's planted strings was found ($(wc -l < "$LOGD/general_tbench_keyscan.txt") files, names in general_tbench_keyscan.txt); nothing uploaded"; return 1
  fi
  local n; n=$(pack)
  ( set -a; . /workspace/.env > /dev/null 2>&1; set +a   # HF_TOKEN in this subshell only
    hf upload "$R" "$STAGE_DIR" "$HFD/raw" --repo-type dataset --exclude "*.part" --commit-message "Terminal-Bench runs (DeepSeek V4 Pro), $(date -u +%FT%TZ)" \
      && { [ ! -d "$LG" ] || hf upload "$R" "$LG" "$HFD/logs" --repo-type dataset --commit-message "Terminal-Bench ledgers and tables"; } ) \
    >> "$LOGD/general_tbench_upload.log" 2>&1 \
    && say "uploaded to HF (private, $HFD/raw): $n newly packed runs, $(find "$STAGE_DIR" -name "*.tar.gz" | wc -l) runs there in all" \
    || say "UPLOAD FAILED (general_tbench_upload.log); the collection goes on"
}

case "${1:-once}" in
  once)   upload ;;
  follow) P=${2:?the PID to follow}; upload; n=0
          while kill -0 "$P" 2> /dev/null; do sleep 300; n=$((n + 1)); [ $((n % 24)) = 0 ] && kill -0 "$P" 2> /dev/null && upload; done
          upload; say "final upload after the collection ended" ;;
  *)      echo "usage: $0 once | follow PID"; exit 1 ;;
esac
