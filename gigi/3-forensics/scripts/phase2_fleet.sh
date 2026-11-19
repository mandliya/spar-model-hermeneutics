#!/usr/bin/env bash
# phase2_fleet.sh: phase 2, stage 3 for one teacher: the full collection, 28 tasks x 40 complete runs (19 Nov 2026;
# phase2/harness_runbook.md section 4, step 5; the classic round's classic_fleet3.sh design). Needs GG's yes for the money
# after the pilot's projection. Runs ON THE LAUNCH BOX, detached, sleep-safe:
#   TAG=deepseek-v4-pro PIN=siliconflow ESTIMATE=1900 SPEND_CAP=7600 \
#     setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/phase2_fleet.sh > /workspace/logs/phase2_fleet.out 2>&1 < /dev/null &
# - Launch counts as the classic round: 46 for the 13 long tasks, 40 for the other 15 (1,198 runs); top-ups to 40
#   complete per task (at most 2 rounds, through the same queue).
# - Paced queue: each task in chunks of at most 6 runs, CHUNK_MAXC (3) at once per chunk, stage <SHORT>_f_<k> = chunk k,
#   longest tasks first. A chunk starts only while the runs in flight stay under CAP and the provider is healthy: over the
#   last 5 minutes fewer than H429 (50) 429-retry lines and no retry past attempt 8 (memory swap-launch-pacing). Not
#   healthy: launches held and CAP -5 (down to CAP_MIN); healthy and quiet (under 10): CAP +2 (up to CAP_MAX).
# - Money: before launching, the key and the account must hold 1.5 x ESTIMATE (phase2_money.py; the workspace budget is
#   John's to check). SPEND_CAP (default 4 x ESTIMATE, memory spend-caps-generous) is checked every 5 minutes against the
#   larger of the key's usage delta and the launch logs' own [provider-usage] ledger; past it, every app of the fleet
#   stops and nothing more launches.
# - Set-aside rule (fixed before any run, as the classic round's): a (stage, task) chunk with a run cut by our or the
#   provider's failure (the wall clock, a request timeout, 429 retries given up) is set aside (results/logs/<SHORT>/
#   set_aside.tsv) and re-run once from scratch; cut runs bias a task short. Empty-reply crashes are the model's own and
#   count as lost runs.
# - Then: per-run table (HERO_ALL_ENVS=1), cost per run (run_usage.py), upload of every stage and the logs to the private
#   HF dataset, "P2 FLEET DONE: stages ..." in the stage log. A heartbeat line every 5 minutes. Never stops a box.
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; LOGD=/workspace/logs; S=$LOGD/phase2.txt
. "$H/scripts/phase2_lib.sh"
set -a; [ -f /workspace/.env ] && . /workspace/.env; set +a   # HF_TOKEN, maybe OPENROUTER_API_KEY; never printed
TAG=${TAG:-deepseek-v4-pro}; PIN=${PIN:-siliconflow}; ESTIMATE=${ESTIMATE:?set ESTIMATE (dollars, from the pilot projection)}
SPEND_CAP=${SPEND_CAP:-$(python3 -c "print(int(4 * float('$ESTIMATE')))")}
SHORT=${SHORT:-p2_$(echo "$TAG" | tr -cd 'a-z0-9' | cut -c1-10)}
CAP=${CAP:-30}; CAP_MIN=${CAP_MIN:-10}; CAP_MAX=${CAP_MAX:-60}; CHUNK_MAXC=${CHUNK_MAXC:-3}; H429=${H429:-50}
TIMEOUT_SHORT=${TIMEOUT_SHORT:-10800}; TIMEOUT_LONG=${TIMEOUT_LONG:-43200}   # set from the pilot's longest runs
Q=$LOGD/${SHORT}_queue.txt; SA=$H/results/logs/$SHORT/set_aside.tsv; mkdir -p "$H/results/logs/$SHORT"
LONG46="ai_rd async_conversion dashboard_costly_failure grade_your_own optimize_rendering_engine precommit_hook revert_or_refactor puppeteer reference_extraction leaked_tests optimize_lru_cache leaked_scorer optimize_rendering_engine_graded"
stop () { say "FLEET STOPPED ($TAG): $*"; exit 1; }
steps_of () { upy -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['agent']['max_steps'])" "$H/configs/${1}__${TAG}.yaml"; }

chunks () {   # chunks <stage prefix> <task> <runs>: queue lines "<prefix>_<k> <task> <n>", at most 6 runs per chunk
  local p=$1 e=$2 n=$3 k=1 c
  while [ "$n" -gt 0 ]; do c=$(( (n + (n + 5) / 6 - 1) / ((n + 5) / 6) )); echo "${p}_$k $e $c"; n=$((n - c)); k=$((k + 1)); done
}
stages_all () { cut -d' ' -f1 "$Q.done" 2>/dev/null | awk '!s[$0]++' | paste -sd, -; }

launch_chunk () {   # launch_chunk STAGE TASK N
  local st=$1 e=$2 n=$3 cap to sec
  cap=$(steps_of "$e"); to=$TIMEOUT_LONG; [ "$cap" -le 60 ] && to=$TIMEOUT_SHORT
  sec=openrouter-secret; [ "$e" = ai_rd ] && sec=openrouter-secret,openrouter-agent
  MAXC=$CHUNK_MAXC GAP=12 TIMEOUT_S=$to MODAL_SECRETS=$sec bash "$H/scripts/modal_full.sh" "$st" "$TAG" "$e:$n"
}

PREV429=0; HEALTHY=1
health () {   # 429s over the last poll; adjusts CAP; logs one line
  local r top d st; st=$(stages_all); [ -n "$st" ] || { HEALTHY=1; return; }
  read r top < <(rate_limits "$st"); d=$((r - PREV429)); PREV429=$r
  if [ "$d" -ge "$H429" ] || [ "$top" -ge 8 ]; then HEALTHY=0; CAP=$((CAP - 5 > CAP_MIN ? CAP - 5 : CAP_MIN))
  else HEALTHY=1; [ "$d" -lt 10 ] && CAP=$((CAP + 2 < CAP_MAX ? CAP + 2 : CAP_MAX)); fi
  say "provider, last 5 min: 429 retries +$d (highest attempt $top); in flight $(live "$st") runs in $(apps "$st") apps; spent \$$(spent "$st") of cap \$$SPEND_CAP; cap now $CAP runs$([ $HEALTHY = 0 ] && echo ', NOT HEALTHY: launches held')"
}

run_queue () {   # launch the queue as health, CAP and money allow; return once it is empty and nothing is in flight
  local next st e n l all
  while true; do
    health; all=$(stages_all)
    if [ -n "$all" ]; then guard "$all" "$SPEND_CAP" || stop "spend cap reached"; fi
    l=$([ -n "$all" ] && live "$all" || echo 0)
    while [ -s "$Q" ] && [ $HEALTHY = 1 ] && [ $((l + CHUNK_MAXC)) -le $CAP ]; do
      next=$(head -n 1 "$Q"); read st e n <<< "$next"
      launch_chunk "$st" "$e" "$n"
      echo "$next" >> "$Q.done"; sed -i 1d "$Q"; l=$((l + (n < CHUNK_MAXC ? n : CHUNK_MAXC)))
      say "launched $st $e x$n; $l runs in flight, $(grep -c . "$Q" || true) chunks queued"
    done
    all=$(stages_all)
    [ -s "$Q" ] || { [ -n "$all" ] && [ "$(apps "$all")" -gt 0 ]; } || return 0
    sleep 300
  done
}

set_aside_cuts () {   # append chunks with cut runs to set_aside.tsv; print queue lines to re-run them once
  upy - "$H" "$(stages_all)" "$SA" "${SHORT}_s" <<'PY'
import collections, pathlib, sys
H, stages, sa, prefix = pathlib.Path(sys.argv[1]), sys.argv[2].split(","), pathlib.Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(H / "scripts"))
from distill_data import run_end
done = {tuple(l.split("\t")[:2]) for l in sa.read_text().splitlines() if l and not l.startswith("#")} if sa.exists() else set()
k = collections.Counter(); out = []
for st in stages:
    for envd in sorted(p for p in (H / "results/raw" / st).glob("*") if p.is_dir()):
        if (st, envd.name) in done:
            continue
        runs = [r for r in envd.rglob("run-*") if r.is_dir()]
        cut = []
        for r in runs:
            e = run_end(r); t = (r / "rollout.log").read_text(errors="replace")[-4000:] if (r / "rollout.log").exists() else ""
            if e in ("cut off by the time limit", "our server timed out") or (e == "crash" and "RateLimitError" in t):
                cut.append(f"{r.name}: {e}{' (429s given up)' if e == 'crash' else ''}")
        if cut:
            with open(sa, "a") as f:
                if f.tell() == 0:
                    f.write("# stage\ttask\treason (phase2_fleet.sh set-aside rule: a run cut by our or the provider's failure)\n")
                f.write(f"{st}\t{envd.name}\t{'; '.join(cut)}\n")
            k[envd.name] += 1
            out.append(f"{prefix}_{k[envd.name]} {envd.name} {len(runs)}")
print("\n".join(out))
PY
}

# preflight
say "START phase 2 fleet: $TAG pinned to $PIN, estimate \$$ESTIMATE, spend cap \$$SPEND_CAP, cap $CAP runs at once"
n=$(ls "$H"/configs/*__"$TAG".yaml 2>/dev/null | wc -l); [ "$n" = 28 ] || stop "$n configs for $TAG, need 28"
grep -q "USAGE_KEY" "$H/agent-interp-envs/src/agent_interp_envs/providers/openrouter_provider.py" || stop "no usage ledger in the harness"
[ "${SKIP_PILOT_CHECK:-0}" = 1 ] || grep -q "PILOT DONE ($TAG)" "$S" || stop "no 'PILOT DONE ($TAG)' line: run phase2_pilot.sh first"
python3 "$H/scripts/phase2_money.py" --need "$(python3 -c "print(1.5 * float('$ESTIMATE'))")" --save-start "$MONEY_START" >> "$S" 2>&1; rc=$?
[ $rc = 2 ] && stop "not enough OpenRouter money for 1.5 x the estimate: ask GG / John to raise the key and workspace limits"
[ $rc = 3 ] && say "no OpenRouter key on this box: spend is tracked from the launch logs only"

# the queue: longest tasks first
: > "$Q.done"
{ for cfg in "$H"/configs/*__"$TAG".yaml; do e=$(basename "$cfg"); e=${e%%__*}; echo "$(steps_of "$e") $e"; done | sort -rn | while read cap e; do
    n=40; case " $LONG46 " in *" $e "*) n=46;; esac
    chunks "${SHORT}_f" "$e" "$n"
  done; } > "$Q"
say "queue: $(grep -c . "$Q") chunks, $(awk '{s+=$3} END {print s}' "$Q") runs"
run_queue

# fetch, set-asides (one re-run round), top-ups (at most 2 rounds)
for st in $(stages_all | tr ',' ' '); do fetch "$st"; done
re=$(set_aside_cuts)
if [ -n "$re" ]; then
  say "set aside (cut runs): $(echo "$re" | wc -l) chunks; re-running them once ($SA)"
  echo "$re" > "$Q"; before=$(stages_all); run_queue
  for st in $(stages_all | tr ',' ' '); do echo ",$before," | grep -q ",$st," || fetch "$st"; done
fi
CNT () { COUNTS_TAG=$TAG SET_ASIDE_FILE=$SA upy ../scripts/classic_counts.py "$(stages_all)" "$@"; }
CNT > "$H/results/logs/$SHORT/counts_after_fleet.txt" 2>> "$LOGD/phase2_fetch.log"
for r in 1 2; do
  need=$(CNT --shortfall 2>> "$LOGD/phase2_fetch.log")
  [ -z "$need" ] && { say "every task has 40 complete runs"; break; }
  say "top-up $r: $need"
  : > "$Q"; for x in $need; do chunks "${SHORT}_t$r" "${x%%:*}" "${x##*:}" >> "$Q"; done
  before=$(stages_all); run_queue
  for st in $(stages_all | tr ',' ' '); do echo ",$before," | grep -q ",$st," || fetch "$st"; done
done
CNT > "$H/results/logs/$SHORT/counts_final.txt" 2>> "$LOGD/phase2_fetch.log"
say "final counts: $(tail -n 1 "$H/results/logs/$SHORT/counts_final.txt")"

# tables, cost, uploads
STAGES=$(stages_all)
HERO=1 HERO_ALL_ENVS=1 HERO_MODEL=$TAG HERO_STAGES="$STAGES" HERO_OUT_PREFIX=${SHORT}_full_runs__$TAG \
  upy ../scripts/full_runs_table.py --hero >> "$LOGD/phase2_table.log" 2>&1 && say "per-run table written" || say "PER-RUN TABLE FAILED (phase2_table.log)"
upy ../scripts/run_usage.py "$STAGES" --csv "results/${SHORT}_usage.csv" > "$H/results/logs/$SHORT/usage.txt" 2>&1
say "fleet cost: $(grep -E '^total spent' "$H/results/logs/$SHORT/usage.txt")"
for st in ${STAGES//,/ }; do upload_stage "$st" "phase2/$TAG/raw/$st"; done
hf upload model-hermeneutics/gigi-forensics-transcripts "$H/results/logs/$SHORT" "phase2/$TAG/logs" --repo-type dataset >> "$LOGD/phase2_upload.log" 2>&1 \
  && say "uploaded results/logs/$SHORT to HF" || say "UPLOAD FAILED for results/logs/$SHORT"
echo "$STAGES" > "$H/results/logs/$SHORT/stages.txt"
say "P2 FLEET DONE ($TAG): stages $STAGES"
