#!/usr/bin/env bash
# p2gen_launch.sh (24 Nov 2026), ON THE LAUNCH BOX (never the laptop: a sleeping Mac kills a Modal launch): every stage of
# the general-set round for ONE training set, on Modal (phase2/general_set/PIPELINE.md). The three sets run side by side:
# start this script once per set (p2gen_toolscale, p2gen_wildchat, p2gen_tbench).
#   bash scripts/p2gen_launch.sh data   <dv> <folder with train.jsonl and val.jsonl>   upload the set to the Modal volume
#   bash scripts/p2gen_launch.sh check  <dv>          a CPU-only function: does the workspace start our functions? (cents)
#   bash scripts/p2gen_launch.sh smoke  <dv>          the training smoke test only (4 longest examples, ~10 min of GPU)
#   bash scripts/p2gen_launch.sh train  <dv>          smoke, 2 passes with a copy every half pass, the copies' loss
#   bash scripts/p2gen_launch.sh tests  <dv> [runs]   as each copy is saved: its own server, 8 tasks x 40 runs, the scores
#   bash scripts/p2gen_launch.sh score  <dv>          the score files and the table (reuses p2ds's "before" column)
#   bash scripts/p2gen_launch.sh upload <dv>          copies, runs and logs to the private HF dataset (never the data)
#   bash scripts/p2gen_launch.sh status <dv>          the last lines of the set's stage log, and its servers
#   bash scripts/p2gen_launch.sh stop-server <dv> <name>
#   bash scripts/p2gen_launch.sh retest <dv> <copy[:name]>...  test saved copies again (a cut test: c69:c69r), then score, upload
#   bash scripts/p2gen_launch.sh all    <dv> <folder>     data, check, train (in the background), tests, score, upload
# Detached, the way every long job runs:
#   P2GEN_YES=p2gen_toolscale setsid nohup bash scripts/p2gen_launch.sh all p2gen_toolscale /workspace/p2gen_data/toolscale \
#     > /workspace/logs/p2gen_toolscale_launch.out 2>&1 < /dev/null &
# MONEY: check, smoke, train, tests and all start Modal GPUs or sandboxes. They refuse to start unless P2GEN_YES=<dv> is
# set, which GG sets after saying yes to that set's cost (PIPELINE.md has the estimate). DRY_RUN=1 prints every command
# instead of running it and needs no yes.
# Nothing here creates a Modal secret or touches credentials: the functions mount the existing vllm-secret-gigi, the
# model is public, and HF_TOKEN for the upload is read from /workspace/.env on the box as before.
set -uo pipefail
CMD=${1:?stage}; DV=${2:?training set name, e.g. p2gen_toolscale}
[[ "$DV" =~ ^p2gen_[a-z0-9]+$ ]] || { echo "name $DV: expected p2gen_<letters and digits>"; exit 1; }
DRY=${DRY_RUN:-0}
H=${H:-$(cd "$(dirname "$0")/.." && pwd)}; export H
[ "$DRY" = 1 ] && [ ! -d /workspace ] && export LOGD=${LOGD:-${TMPDIR:-/tmp}/p2gen_dry}
LOGD=${LOGD:-/workspace/logs}; export LOGD; S=$LOGD/$DV.txt
. "$H/scripts/phase2_lib.sh"
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }   # HF_TOKEN, VLLM_API_KEY: never printed
VOL=${MODAL_VOLUME_PREFIX}p2gen; APP=$H/scripts/p2gen_modal.py
TASKS=tower_of_hanoi,lazy_investigation,currency_conversion,eval_tampering,norvane,secret_number,coding,coding_norules
export DISTILL_V=$DV DISTILL_ENVS=$TASKS DISTILL_HELD_OUT=$TASKS
PAR=${PAR:-4}                 # copies tested at the same time, each on its own server
SERVE_HOURS=${SERVE_HOURS:-10} # a server ends by itself after this long, whatever happens to the launcher
HFD=distill/$DV
run () { if [ "$DRY" = 1 ]; then printf 'DRY RUN:'; printf ' %q' "$@"; echo; else "$@"; fi; }
m () { (cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal "$@"); }
paid () { [ "$DRY" = 1 ] || [ "${P2GEN_YES:-}" = "$DV" ] || { echo "The stage '$CMD' spends money on Modal. Set P2GEN_YES=$DV once GG has said yes to this set's cost."; exit 3; }; }
vcat () { m volume get "$VOL" "$1" - 2>/dev/null; }
copy_state () {   # copy_state <server file's text> <copy>: "served", "failed" (its load check said no) or "waiting"
  python3 -c 'import json, sys
t = sys.argv[1]
try: d = json.loads(t[t.find("{"):t.rfind("}") + 1])
except ValueError: d = {}
n = sys.argv[2]
print("served" if n in (d.get("served") or []) and d.get("status") == "up" else "failed" if n in (d.get("gate") or {}) else "waiting")' "$1" "$2"; }
detached () {   # detached <log> <function> <args...>: one `modal run --detach` per function, which then survives this box
  local log=$1; shift
  if [ "$DRY" = 1 ]; then printf 'DRY RUN: (detached, log %s) modal run --detach %s' "$log" "$APP::$1"; shift; printf ' %q' "$@"; echo; return; fi
  ( cd "$H/agent-interp-envs" && nohup env -u VIRTUAL_ENV uv run modal run --detach "$APP::$1" "${@:2}" > "$log" 2>&1 < /dev/null & )
}

do_data () {
  local d=${1:?folder with train.jsonl and val.jsonl}
  for f in train.jsonl val.jsonl; do [ -s "$d/$f" ] || { echo "$d/$f is missing"; exit 1; }; done
  python3 - "$d" <<'PY' || exit 1
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
for f in ("train.jsonl", "val.jsonl"):
    n = 0
    for line in open(d / f):
        e = json.loads(line); n += 1
        assert e.get("format") == "qwen36" and "tools" in e and e["messages"][-1]["role"] == "assistant", f"{f} line {n}: not the p2ds training format"
    print(f"{f}: {n} examples in the p2ds training format ({'one per run, with train_turns' if 'train_turns' in e else 'one per turn'})")
PY
  if grep -lE 'sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}' "$d"/*.jsonl > /dev/null 2>&1; then
    echo "A key shape was found in the data files: not uploading. Look before anything else."; exit 1; fi
  for f in train.jsonl val.jsonl manifest.json; do [ -f "$d/$f" ] && run m volume put --force "$VOL" "$d/$f" "data/$DV/$f"; done
  say "data: $DV uploaded to the Modal volume $VOL (data/$DV). It is not in git and not on HF."
}
do_train () {
  paid; detached "$LOGD/${DV}_train.out" train --dv "$DV" ${EXTRA_NLL:+--extra-nll "$EXTRA_NLL"} "$@"
  say "train: launched on Modal (GPU ${P2GEN_GPU:-H200}); progress in logs/$DV/stages.txt on the volume (p2gen_launch.sh status $DV)"
}
one_copy () {   # one saved copy: its own server, the 320 runs, the server stopped, the scores
  local n=$1 runs=$2 nm=${3:-$1} t=0 st   # nm: server and stage name (a re-test, 27 Nov: c69 -> c69r)
  detached "$LOGD/${DV}_serve_$nm.out" serve --dv "$DV" --copies "$n" --name "$nm" --hours "$SERVE_HOURS"
  [ "$DRY" = 1 ] && { run env STAGE_AS=$nm bash "$H/scripts/p2gen_runs.sh" "$DV" "$n" "$runs" "$nm"; run m volume put "$VOL" /dev/null "serve/$DV/$nm.stop"; return; }
  until st=$(vcat "serve/$DV/$nm.json"); [ "$(copy_state "$st" "$n")" != waiting ]; do   # up, and the copy's load check has an answer
    t=$((t + 1)); [ $t -gt 90 ] && { say "tests: the server for $n did not come up in 90 min ($LOGD/${DV}_serve_$nm.out): skipped"; return 1; }
    sleep 60
  done
  if [ "$(copy_state "$st" "$n")" = served ]; then
    STAGE_AS=$nm bash "$H/scripts/p2gen_runs.sh" "$DV" "$n" "$runs" "$nm" > "$LOGD/${DV}_runs_$nm.out" 2>&1 || say "tests: the runs of $n ended with an error ($LOGD/${DV}_runs_$n.out)"
  else
    say "tests: $n failed its load check on the server, so nothing was run on it"
  fi
  : > "$LOGD/${DV}_stop"; m volume put "$VOL" "$LOGD/${DV}_stop" "serve/$DV/$nm.stop" > /dev/null 2>&1 \
    && say "tests: asked the server of $n to stop" || say "tests: COULD NOT ASK THE SERVER OF $n TO STOP: it ends by itself after $SERVE_HOURS h and bills until then"
  echo "$nm" >> "$LOGD/${DV}_copies_done.txt"
}
do_tests () {
  paid; local runs=${1:-40} n; touch "$LOGD/${DV}_copies_done.txt" "$LOGD/${DV}_copies_started.txt"
  if [ "$DRY" = 1 ]; then one_copy "c<step>" "$runs"; do_score; return; fi
  say "tests: waiting for saved copies of $DV; each gets its own server and $runs runs of each of the 8 tasks; at most $PAR copies at once"
  while true; do
    for n in $(vcat "out/$DV/copies_ready.txt" | grep -oE '^c[0-9]+$'); do
      grep -qx "$n" "$LOGD/${DV}_copies_started.txt" && continue
      [ "$(jobs -rp | wc -l)" -ge "$PAR" ] && break
      echo "$n" >> "$LOGD/${DV}_copies_started.txt"; say "tests: copy $n is saved: starting its server"
      one_copy "$n" "$runs" &
    done
    stages=$(vcat "logs/$DV/stages.txt")
    if echo "$stages" | grep -q "TRAINING DONE\|TRAINING FAILED\|SMOKE FAILED"; then
      left=$(vcat "out/$DV/copies_ready.txt" | grep -oE '^c[0-9]+$' | grep -vxFf "$LOGD/${DV}_copies_started.txt" | wc -l)
      [ "$left" -eq 0 ] && break
    fi
    say "tests: $(grep -c . "$LOGD/${DV}_copies_done.txt") copies finished, $(jobs -rp | wc -l) running; training: $(echo "$stages" | tail -n 1 | cut -c1-120)"
    sleep 300
  done
  wait; do_score
  say "P2GEN TESTS DONE $DV: copies $(tr '\n' ' ' < "$LOGD/${DV}_copies_done.txt")"
}
do_score () {
  if [ "$DRY" = 1 ]; then   # prints only: a dry run writes nothing under results/
    run upy ../scripts/p2gen_configs.py base-check; run upy ../scripts/distill_eval.py score; run python3 "$H/scripts/p2gen_table.py" "$DV"; return
  fi
  mkdir -p "$H/results/logs/$DV"
  [ "$DRY" = 1 ] || vcat "out/$DV/copies.json" > "$H/results/logs/$DV/copies.json"
  run upy ../scripts/p2gen_configs.py base-check || say "score: THE UNTRAINED MODEL'S SETTINGS DIFFER FROM p2ds (see p2gen_configs.py base-check): its runs cannot be reused as they are"
  run upy ../scripts/distill_eval.py score > "$H/results/logs/$DV/score_latest.txt" 2>&1 \
    && run python3 "$H/scripts/p2gen_table.py" "$DV" > "$H/results/logs/$DV/table_latest.md" 2>&1 \
    && say "score: table written (results/distill_${DV}_table.csv, results/logs/$DV/table_latest.md)" \
    || say "score: SCORING FAILED (results/logs/$DV/score_latest.txt)"
}
do_upload () {   # copies (no optimizer state), loss files, logs and raw runs to the private dataset. The training DATA is
                 # never uploaded: its source has no licence, so it stays on the Modal volume and on GG's disk.
  local d=/workspace/p2gen_out/$DV R=model-hermeneutics/gigi-forensics-transcripts
  run mkdir -p "$d"
  run mkdir -p "$d/out" "$d/logs"; run m volume get "$VOL" "out/$DV" "$d/out"; run m volume get "$VOL" "logs/$DV" "$d/logs"
  if [ "$DRY" != 1 ] && grep -rlE --include='*.log' --include='*.txt' --include='*.json' --include='*.jsonl' 'sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}' "$d" "$H/results/logs/$DV" > /dev/null 2>&1; then
    say "upload: A KEY SHAPE WAS FOUND in the logs: nothing uploaded"; return 1; fi
  run hf upload "$R" "$d" "$HFD/runs/v1" --repo-type dataset --exclude "*optimizer.pt" --exclude "*scheduler.pt" --exclude "*rng_state*.pth" --exclude "*broken-*" \
    && say "upload: copies and logs of $DV on HF ($HFD/runs/v1)" || say "upload: UPLOAD FAILED for the copies of $DV"
  for st in "$H"/results/raw/distill_"${DV}"_*; do [ -d "$st" ] && run upload_stage "$(basename "$st")" "$HFD/eval/raw/$(basename "$st")"; done
}
do_retest () {   # do_retest <runs> <copy[:name]>...: test saved copies again, each on a new server named <name>, whose
  # runs go to stage distill_<dv>_<name> (27 Nov 2026: the Modal spend limit cut c69's tests and c90's server); then the
  # scores and the upload. A cut stage must be moved out of results/raw/ first, or it is scored as a partial column.
  paid; local runs=$1 spec n nm; shift
  for spec in "$@"; do n=${spec%%:*}; nm=${spec#*:}
    say "retest: copy $n on server $nm (stage distill_${DV}_$nm)"; one_copy "$n" "$runs" "$nm" &
    sleep 30
  done
  wait; do_score; do_upload; say "P2GEN RETEST DONE $DV: $*"
}
do_status () {
  echo "--- logs/$DV/stages.txt (last 15 lines)"; vcat "logs/$DV/stages.txt" | tail -n 15
  echo "--- copies ready: $(vcat "out/$DV/copies_ready.txt" | tr '\n' ' ')"
  echo "--- servers"; m volume ls "$VOL" "serve/$DV" 2>/dev/null
}

case "$CMD" in
  data)   do_data "${3:-}";;
  check)  paid; run m run "$APP::check" --dv "$DV";;
  smoke)  do_train --smoke-only;;
  train)  do_train;;
  tests)  do_tests "${3:-40}";;
  retest) do_retest "${RUNS:-40}" "${@:3}";;
  score)  do_score;;
  upload) do_upload;;
  status) do_status;;
  stop-server) : > "$LOGD/${DV}_stop"; run m volume put "$VOL" "$LOGD/${DV}_stop" "serve/$DV/${3:?server name}.stop";;
  all)    paid; do_data "${3:-}"; run m run "$APP::check" --dv "$DV" || { say "all: the check function did not run: stopping before any GPU is asked for"; exit 1; }
          do_train; do_tests "${4:-40}"; do_upload; say "P2GEN ALL DONE $DV";;
  *) echo "unknown stage $CMD"; exit 1;;
esac
