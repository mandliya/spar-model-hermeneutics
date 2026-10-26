#!/usr/bin/env bash
# Top up every gap row of chart 4 to ~60 scored replays per model (27 Oct 2026, after the audit: Qwen's rendering,
# ai_rd and dashboard cells had lost most replays to DeepInfra rate limits). Detached; Modal paces the starts itself
# (STAGGER_*): stream 1 = dashboard Qwen, one start every 180 s (long relays, the ones that were throttled in bulk);
# stream 2 = rendering and ai_rd, both models, one start every 35 s. New fleets land next to the old ones.
# A watchdog (topup_watchdog.sh) stops exactly these apps if the key's spend since 18 Oct reaches $1480.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/topup_gaps"; mkdir -p "$LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
python3 - "$HERE" > "$LOG/commands.txt" <<'PY'
import json, pathlib, sys
here = pathlib.Path(sys.argv[1]); R = "results/raw/swap_all"
def pt(p): run = here / p["run"]; return f"{run.parent.name}_{run.name}_K{p['K']}", run / f"step-{p['K']}"
jobs = []   # stream|name|out|step|count|who
db = "results/raw/modal/gigi-forensics-dashboard-perf/dashboard_perf/qwen-qwen3-next-80b-a3b-instruct/2026-10-19_10-57-49-884559/run-2"
for k, n in (("K62", 70), ("K63", 60)):
    jobs.append(("1", f"dash_{k}_qwen", f"{R}/dashboard_costly_failure/2026-10-19_10-57-49-884559_run-2_{k}/qwen", here / db / f"step-{k[1:]}", n, "qwen"))
qn = {"run-1_K10": 23, "run-1_K9": 22, "run-4_K56": 20, "run-4_K55": 23, "run-1_K24": 16, "run-1_K23": 16}
for p in json.load(open(here / "results/swap_all_plan.json")):
    if p["env"] == "optimize_rendering_engine" and p["incident"]:
        name, step = pt(p); key = name.split("_", 2)[-1]
        jobs.append(("2", f"render_{key}_qwen", f"{R}/optimize_rendering_engine/{name}/qwen", step, qn[key], "qwen"))
        jobs.append(("2", f"render_{key}_gemma", f"{R}/optimize_rendering_engine/{name}/gemma", step, 10, "gemma"))
aq = {"run-1_K2": 22, "run-1_K1": 20, "run-10_K2": 13, "run-10_K1": 13, "run-2_K2": 13, "run-2_K1": 16}
for p in json.load(open(here / "results/swap_aird_plan.json")):
    name, step = pt(p); key = name.split("_", 2)[-1]
    jobs.append(("2", f"aird_{key}_qwen", f"{R}/ai_rd/{name}/qwen", step, aq.get(key, 10), "qwen"))
    jobs.append(("2", f"aird_{key}_gemma", f"{R}/ai_rd/{name}/gemma", step, 10, "gemma"))
for j in jobs: print("|".join(map(str, j)))
PY
off1=0; off2=0
while IFS='|' read -r stream name out step n who; do
  if [ "$stream" = 1 ]; then start=$off1; stepS=180; off1=$((off1 + n * 180)); else start=$off2; stepS=35; off2=$((off2 + n * 35)); fi
  ov=""; [ "$who" = gemma ] && ov="$GEMMA"
  ( cd "$ENVS" && STAGGER_START=$start STAGGER_STEP=$stepS MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run \
      --detach scripts/resume_modal.py --step-dir "$step" --count "$n" --use-current --max-concurrent 0 --no-judge \
      --timeout 14400 --results-dir "$HERE/$out" ${ov:+--overrides "$ov"} > "$LOG/$name.log" 2>&1 & )
  sleep 40
  echo "$(date '+%F %T') launched $name x$n (stream $stream, starts at +$((start / 60)) min)" >> "$LOG/launch.txt"
done < "$LOG/commands.txt"
echo "$(date '+%F %T') all launched" >> "$LOG/launch.txt"
