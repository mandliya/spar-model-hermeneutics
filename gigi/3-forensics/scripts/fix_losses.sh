#!/usr/bin/env bash
# Rerun every swap try lost to a fixable cause (27 Oct 2026, after the full loss audit): for each restart point of every
# plan (swap_all, swap_extra, swap_aird, swap_clean) and each model, launch target - scored - unfixable new tries, where
# unfixable = gemma's context window, "no measure", "no value for the measure", step cap without a report. Target = the
# plan's reps (10), or 60 for dashboard_costly_failure's Qwen cells (chart-4 rows). Detached; Modal paces the starts
# (STAGGER_*): stream 1 = dashboard Qwen, one start every 240 s (very long histories, the ones DeepInfra throttles);
# stream 2 = everything else, one start every 35 s. New fleets land next to the old ones. A watchdog (topup_watchdog.sh
# with LOGDIR=fix_losses) stops exactly these apps if the key's spend since 18 Oct reaches its CEIL.
#   DRY=1 bash scripts/fix_losses.sh      # only write results/logs/fix_losses/commands.txt and print the counts
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/fix_losses"; mkdir -p "$LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
( cd "$ENVS" && env -u VIRTUAL_ENV uv run python - "$HERE" <<'PY' ) > "$LOG/commands.txt"
import csv, collections, pathlib, sys
here = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(here / "scripts"))
import swap_all_score as S
R = "results/raw/swap_all"
UNFIXABLE = {"context window exceeded", "no measure", "no value for the measure", "step cap, no report"}
scored, unfix = collections.Counter(), collections.Counter()
for r in csv.DictReader(open(here / "results/gap_replays.csv")):
    k = (r["env"], r["point"], r["who"])
    if r["hacked"] != "": scored[k] += 1
    elif r["status"] in UNFIXABLE: unfix[k] += 1
jobs = []   # stream|name|out|step|count|who
for p in S.PLAN:
    name = S.point_name(p); step = here / p["run"] / f"step-{p['K']}"
    for who in ("qwen", "gemma"):
        k = (p["env"], name, who)
        target = 60 if (p["env"] == "dashboard_costly_failure" and p["incident"] and who == "qwen") else p.get("reps", 10)
        need = target - scored[k] - unfix[k]
        if need <= 0: continue
        stream = "1" if (p["env"] == "dashboard_costly_failure" and who == "qwen" and p["incident"]) else "2"
        short = name.split("_", 2)[-1]
        jobs.append((stream, f"{p['env'][:12]}_{short}_{who}", f"{R}/{p['env']}/{name}/{who}", step, need, who))
for j in jobs: print("|".join(map(str, j)))
print(f"{sum(j[4] for j in jobs)} tries in {len(jobs)} fleets", file=sys.stderr)
PY
if [ "${DRY:-}" = 1 ]; then cat "$LOG/commands.txt" | awk -F'|' '{printf "%-34s x%-3s stream %s\n", $2, $5, $1}'; exit 0; fi
off1=0; off2=0
while IFS='|' read -r stream name out step n who; do
  if [ "$stream" = 1 ]; then start=$off1; stepS=240; off1=$((off1 + n * 240)); else start=$off2; stepS=35; off2=$((off2 + n * 35)); fi
  ov=""; [ "$who" = gemma ] && ov="$GEMMA"
  ( cd "$ENVS" && STAGGER_START=$start STAGGER_STEP=$stepS MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run \
      --detach scripts/resume_modal.py --step-dir "$step" --count "$n" --use-current --max-concurrent 0 --no-judge \
      --timeout 14400 --results-dir "$HERE/$out" ${ov:+--overrides "$ov"} > "$LOG/$name.log" 2>&1 & )
  sleep 40
  echo "$(date '+%F %T') launched $name x$n (stream $stream, starts at +$((start / 60)) min)" >> "$LOG/launch.txt"
done < "$LOG/commands.txt"
echo "$(date '+%F %T') all launched" >> "$LOG/launch.txt"
