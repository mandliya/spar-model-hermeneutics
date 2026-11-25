#!/usr/bin/env bash
# phase2_finish.sh: the closing steps of phase2_fleet.sh, for a fleet its spend guard stopped after the queue was empty
# (25 Nov 2026: the GPT-OSS fleet's guard, which counts the whole OpenRouter key, fired at $501 of $500 during the
# set-aside round because Terminal-Bench runs on the same key; the fleet's own ledger was $87). NO MODEL CALLS: lean
# fetch of the set-aside stages (the main stages were fetched before that round), final counts, the shortfall a top-up
# would need (printed, never launched: top-ups spend money and need GG's yes), the per-run table, cost per run, and the
# upload of every stage and the logs to the private HF dataset. Runs ON THE LAUNCH BOX, detached:
#   TAG=gpt-oss-120b setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/phase2_finish.sh > /workspace/logs/phase2_finish.out 2>&1 < /dev/null &
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; LOGD=/workspace/logs; S=$LOGD/phase2.txt
. "$H/scripts/phase2_lib.sh"
set -a; [ -f /workspace/.env ] && . /workspace/.env; set +a   # HF_TOKEN; never printed
TAG=${TAG:?set TAG}; SHORT=${SHORT:-p2_$(echo "$TAG" | tr -cd 'a-z0-9' | cut -c1-10)}
Q=$LOGD/${SHORT}_queue.txt; SA=$H/results/logs/$SHORT/set_aside.tsv
STAGES=$(cut -d' ' -f1 "$Q.done" 2>/dev/null | awk '!s[$0]++' | paste -sd, -)
[ -n "$STAGES" ] || { say "FINISH STOPPED ($TAG): no stages in $Q.done"; exit 1; }
[ "$(grep -c . "$Q" 2>/dev/null)" = 0 ] || { say "FINISH STOPPED ($TAG): $Q still has chunks queued"; exit 1; }
say "FINISH ($TAG, no model calls): fetch the set-aside stages, final counts, per-run table, cost per run, HF upload"

for st in ${STAGES//,/ }; do case "$st" in *_s_*|*_t1_*|*_t2_*) fetch "$st";; esac; done
CNT () { COUNTS_TAG=$TAG SET_ASIDE_FILE=$SA upy ../scripts/classic_counts.py "$STAGES" "$@"; }
CNT > "$H/results/logs/$SHORT/counts_final.txt" 2>> "$LOGD/phase2_fetch.log"
say "final counts: $(tail -n 1 "$H/results/logs/$SHORT/counts_final.txt")"
say "shortfall to 40 complete per task (plus a margin of 2), NOT launched: $(CNT --shortfall 2>> "$LOGD/phase2_fetch.log")"

HERO=1 HERO_ALL_ENVS=1 HERO_MODEL=$TAG HERO_STAGES="$STAGES" HERO_OUT_PREFIX=${SHORT}_full_runs__$TAG \
  upy ../scripts/full_runs_table.py --hero >> "$LOGD/phase2_table.log" 2>&1 && say "per-run table written" || say "PER-RUN TABLE FAILED (phase2_table.log)"
upy ../scripts/run_usage.py "$STAGES" --csv "results/${SHORT}_usage.csv" > "$H/results/logs/$SHORT/usage.txt" 2>&1
say "fleet cost: $(grep -E '^total spent' "$H/results/logs/$SHORT/usage.txt")"
keyscan () {   # 0 = no key shape in what would be uploaded; 1 = found (file NAMES only in ${SHORT}_keyscan.txt, never a match)
  python3 - "$H" "$STAGES" "$SHORT" "$LOGD/${SHORT}_keyscan.txt" <<'PY'
import pathlib, re, sys
H, stages, short, out = pathlib.Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3], pathlib.Path(sys.argv[4])
pat = re.compile(rb"sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|fw_[A-Za-z0-9]{20,}")
dirs = [H / "results/raw" / st for st in stages] + [H / "results/logs" / short]
hits = [str(p) for d in dirs for p in d.rglob("*") if p.is_file() and "/fs/" not in str(p) and pat.search(p.read_bytes())]
out.write_text("".join(h + "\n" for h in hits))
sys.exit(1 if hits else 0)
PY
}
keyscan || { say "UPLOAD REFUSED ($TAG): a key shape was found in $(wc -l < "$LOGD/${SHORT}_keyscan.txt") files (names in ${SHORT}_keyscan.txt); nothing uploaded"; exit 1; }
say "key-shape scan: none found"
for st in ${STAGES//,/ }; do upload_stage "$st" "phase2/$TAG/raw/$st"; done
hf upload model-hermeneutics/gigi-forensics-transcripts "$H/results/logs/$SHORT" "phase2/$TAG/logs" --repo-type dataset >> "$LOGD/phase2_upload.log" 2>&1 \
  && say "uploaded results/logs/$SHORT to HF" || say "UPLOAD FAILED for results/logs/$SHORT"
echo "$STAGES" > "$H/results/logs/$SHORT/stages.txt"
say "FINISH DONE ($TAG): stages $STAGES"
