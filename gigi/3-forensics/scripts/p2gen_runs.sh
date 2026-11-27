#!/usr/bin/env bash
# p2gen_runs.sh (24 Nov 2026), ON THE LAUNCH BOX: the test runs of one saved copy of Qwen3.6 27B trained on a general
# set, against its vLLM server on Modal (phase2/general_set/PIPELINE.md). It is scripts/p2q_runs.sh with two changes:
# the server's address comes from the Modal volume (serve/<dv>/<name>.json, written by p2gen_modal.py::serve) instead of
# a fixed Vast address, and the number of runs at once is sized to the server's KV cache instead of fixed.
#   bash scripts/p2gen_runs.sh <dv> <copy> [runs per task, default 40] [server name, default = the copy]
#   bash scripts/p2gen_runs.sh p2gen_toolscale c8 40
#   GATE=1 bash scripts/p2gen_runs.sh p2gen_toolscale c8      # one eval_tampering run first, stage distill_<dv>_gate
#   DRY_RUN=1 ... prints the sizing and the launch command; contacts nothing (P2GEN_STATE_JSON='{...}' stands in for the
#   server's file)
# Same 8 tasks as p2ds, 40 runs each, the same sampling and step caps (scripts/p2gen_configs.py writes the configs with
# distill_eval.py's own code). The untrained model is not run again: p2ds's runs are reused (p2gen_configs.py base-check).
# Sizing rule (memory vllm-agent-fleet-kv-sizing): live runs x their history <= 70% of the server's KV cache. The history
# per task is the 90th percentile over p2ds's 640 trained-copy runs (measured 24 Nov on the launch box, tokens):
set -uo pipefail
DV=${1:?training set name, e.g. p2gen_toolscale}; M=${2:?copy: c<step> (or base)}; N=${3:-40}; NAME=${4:-$M}
H=${H:-/workspace/forensics/gigi/3-forensics}; LOGD=${LOGD:-/workspace/logs}; S=$LOGD/$DV.txt
. "$H/scripts/phase2_lib.sh"                      # say, upy, PATH, Modal profile and names
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }   # VLLM_API_KEY (the server's password): never printed
P90="secret_number:45400 coding:26100 tower_of_hanoi:22500 coding_norules:18200 eval_tampering:11800 lazy_investigation:7100 currency_conversion:4600 norvane:3000"
TASKS=tower_of_hanoi,lazy_investigation,currency_conversion,eval_tampering,norvane,secret_number,coding,coding_norules
export DISTILL_V=$DV DISTILL_ENVS=$TASKS DISTILL_HELD_OUT=$TASKS
VOL=${MODAL_VOLUME_PREFIX}p2gen; DRY=${DRY_RUN:-0}
L=$H/results/logs/p2gen_runs/$DV; [ "$DRY" = 1 ] || mkdir -p "$L"
m () { (cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal "$@"); }
state () { m volume get "$VOL" "serve/$DV/$NAME.json" - 2>/dev/null | python3 -c 'import sys; t = sys.stdin.read(); print(t[t.find("{"):t.rfind("}") + 1])'; }
field () { python3 -c 'import json, sys
try: d = json.loads(sys.argv[1])
except ValueError: d = {}
v = d.get(sys.argv[2]); print(" ".join(v) if isinstance(v, list) else "" if v is None else v)' "$1" "$2"; }

ST8=$([ "$DRY" = 1 ] && echo "${P2GEN_STATE_JSON:-}" || state); URL=$(field "$ST8" url); KV=$(field "$ST8" kv_cache_tokens); R0=$(field "$ST8" restarts)
served=$([ "$M" = base ] && echo qwen36-base || echo "$M")
if [ "$DRY" != 1 ]; then
  [ "$(field "$ST8" status)" = up ] && [ -n "$URL" ] || { say "p2gen_runs $DV $M: no running server named $NAME (serve/$DV/$NAME.json): stopping"; exit 1; }
  echo " $(field "$ST8" served) " | grep -q " $served " || { say "p2gen_runs $DV $M: the server does not serve $served (its load check: $(field "$ST8" gate | cut -c1-200)): stopping"; exit 1; }
  curl -sf -m 30 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/models" | grep -q "\"$served\"" \
    || { say "p2gen_runs $DV $M: the server does not answer at its address: stopping"; exit 1; }
  upy ../scripts/p2gen_configs.py write --url "$URL" --models "$M" > "$L/configs_$M.txt" 2>&1 \
    || { say "p2gen_runs $DV $M: CONFIGS FAILED ($L/configs_$M.txt)"; exit 1; }
fi
# runs at once per task: the largest number (at most 6, p2ds's highest) that keeps the 8 tasks' histories within 70% of
# the cache; 3 (p2ds's first setting) when the server did not report its cache size. MAXC=<n> overrides.
SIZED=$(python3 -c 'import sys
kv = int(sys.argv[1] or 0); need = sum(int(x.split(":")[1]) for x in sys.argv[2].split())
print(max(1, min(6, int(0.7 * kv // need))) if kv else 3)' "${KV:-}" "$P90")
MAXC=${MAXC:-$SIZED}
alive () { pgrep -f "results/raw/$1/$2\$" > /dev/null; }
live_all () { local n=0 f e; for f in "$H/results/logs/$1"/*__*.log; do [ -f "$f" ] || continue; e=$(basename "$f"); alive "$1" "${e%%__*}" && n=$((n + 1)); done; echo $n; }
if [ "${GATE:-0}" = 1 ]; then ST=distill_${DV}_gate; COUNTS="eval_tampering:1"; else
  # STAGE_AS (27 Nov): a re-test of a copy whose first stage was cut runs under a new stage name (modal_full.sh skips
  # tasks whose launch log exists); the served copy and its configs keep the copy's own name
  ST=distill_${DV}_${STAGE_AS:-$M}; COUNTS=$(for e in ${TASKS//,/ }; do printf "%s:%s " "$e" "$N"; done); fi
if [ "$DRY" = 1 ]; then
  echo "DRY RUN: server file: ${ST8:-none yet}"
  echo "DRY RUN: KV cache ${KV:-unknown} tokens -> $MAXC runs at once per task ($((MAXC * 8)) in all)"
  echo "DRY RUN: MAXC=$MAXC GAP=12 MODAL_SECRETS=vllm-secret bash $H/scripts/modal_full.sh $ST distill_${DV}_$M $COUNTS"
  exit 0
fi
say "p2gen_runs $DV $M: launching $COUNTS (stage $ST, $MAXC at a time per task, sized to a KV cache of ${KV:-unknown} tokens)"
MAXC=$MAXC GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" "$ST" "distill_${DV}_$M" $COUNTS >> "$L/launch_${STAGE_AS:-$M}.txt" 2>&1
sleep 60
while [ "$(live_all "$ST")" -gt 0 ]; do
  sleep 300; ST8=$(state)
  if [ "$(field "$ST8" status)" != up ] || [ "$(field "$ST8" restarts)" != "$R0" ]; then   # preempted or stopped: its
    say "p2gen_runs $DV $M: THE SERVER $NAME WENT AWAY OR RESTARTED (status $(field "$ST8" status), start $(field "$ST8" restarts) against $R0). Runs in flight are cut; stopping this stage's apps. Run the copy again under a new stage after checking $L."
    stop_apps "$ST"; exit 2   # address changed, so every run in flight lost its server; a cut run is never scored
  fi
  say "p2gen_runs $DV $M: $(live_all "$ST") apps still running; server: $(python3 -c 'import json, sys
h = (json.loads(sys.argv[1]) if sys.argv[1] else {}).get("health") or {}
print("cache hits %s%%, cache %s%% in use, %s running, %s waiting%s" % (h.get("cache_hits_pct"), h.get("cache_in_use_pct"), h.get("running"), h.get("waiting"), "" if h.get("healthy", True) else ", NOT HEALTHY")) if h else print("no health line yet")' "$ST8")"
done
upy ../scripts/distill_fetch_lean.py "$ST" >> "$L/fetch.log" 2>&1 || say "p2gen_runs $DV $M: FETCH FAILED ($L/fetch.log)"
say "P2GEN RUNS DONE $DV $M (stage $ST fetched)"
