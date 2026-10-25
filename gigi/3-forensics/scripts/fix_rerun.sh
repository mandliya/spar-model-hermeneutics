#!/usr/bin/env bash
# Rerun the cells our text-mode parser had cut (validation pass, 23 Oct; fix d2f7f8b, 25 Oct): a call gemma wrote
# that didn't parse ended the run instead of getting an error back. All on Modal (SPAR workspace), detached,
# built from the CURRENT tree (--use-current), which has the fix. The superseded runs and their logs are moved
# to archive/parser_cut_25oct/ (never deleted), so the tables count only the new ones.
#   A  ai_rd: gemma swapped in at Qwen's 6 incident points (swap_aird_plan.json, incident=true), 10 each
#   B  pre-commit hook: gemma swapped in at Qwen's 6 incident points (q1 at N, N-1, N-2; q1m at N-5; q2 at N, N-1), 20 each
#   C  gemma's own full runs where the parser had cut runs: ai_rd x10, puppeteer x5, revert_or_refactor x5 (whole batches)
# Logs: results/logs/swap_fix/ (A, B) and results/logs/full_fix/ (C); spend before/after in results/logs/swap_fix/cost.txt.
#   bash gigi/3-forensics/scripts/fix_rerun.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; ENVS="$HERE/agent-interp-envs"
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
LOG="$HERE/results/logs/swap_fix"; ARCH_RAW="$HERE/results/raw/archive/parser_cut_25oct"; ARCH_LOG="$HERE/results/logs/archive/parser_cut_25oct"
mkdir -p "$LOG" "$ARCH_RAW" "$ARCH_LOG"
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true"
usage () { ( set -a; . "$HERE/../../.env"; set +a
  curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["usage"])' ); }
echo "$(date '+%F %T') fix_rerun start | key spend before: \$$(usage)" >> "$LOG/cost.txt"

# ---- the launch list: "<log> <results dir> <step dir> <count>"
python3 - "$HERE" > "$LOG/commands.txt" <<'PY'
import json, pathlib, sys
here = pathlib.Path(sys.argv[1])
for p in json.load(open(here / "results/swap_aird_plan.json")):
    if not p["incident"]:
        continue
    run = here / p["run"]; point = f"{run.parent.name}_{run.name}_K{p['K']}"
    print(here / f"results/logs/swap_fix/ai_rd__{point}__gemma.log", here / f"results/raw/swap_all/ai_rd/{point}/gemma",
          run / f"step-{p['K']}", p["reps"])
Q = "results/raw/screen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-20-35-423567"
B = "results/raw/base_qwen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-24-48-348561"
for tag, run, ks in (("precommit_q1", f"{Q}/run-5", (8, 7, 6)), ("precommit_q1m", f"{Q}/run-5", (3,)), ("precommit_q2", f"{B}/run-18", (3, 2))):
    for K in ks:
        print(here / f"results/logs/swap_fix/{tag}_from_step{K}__gemma.log", here / f"results/raw/swap_text/{tag}/from_step{K}/gemma",
              here / run / f"step-{K}", 20)
PY

# ---- archive what the new runs replace (A, B: the gemma fleets at these points and their logs; C: the stage logs)
while read -r log out step n; do
  rel=${out#"$HERE/results/raw/"}
  if [ -d "$out" ]; then mkdir -p "$(dirname "$ARCH_RAW/$rel")"; mv "$out" "$ARCH_RAW/$rel"; fi
  old=""; case "$rel" in
    swap_all/*) p=${rel#swap_all/ai_rd/}; old="$HERE/results/logs/swap_all/ai_rd__${p%/gemma}__gemma.log";;
    swap_text/*) t=${rel#swap_text/}; t=${t%/gemma}; old="$HERE/results/logs/swap_text/${t%%/*}_${t##*/}.log";;
  esac
  [ -n "$old" ] && [ -f "$old" ] && mv "$old" "$ARCH_LOG/"
done < "$LOG/commands.txt"
for f in full_key/ai_rd__gemma-3-27b.log full_text/puppeteer__gemma-3-27b.log full_text/revert_or_refactor__gemma-3-27b.log; do
  [ -f "$HERE/results/logs/$f" ] && mkdir -p "$ARCH_LOG/$(dirname "$f")" && mv "$HERE/results/logs/$f" "$ARCH_LOG/$f"
done
echo "$(date '+%F %T') archived the superseded runs and logs to archive/parser_cut_25oct/" >> "$LOG/cost.txt"

# ---- A and B: one detached resume per point, all replays at once
while read -r log out step n; do
  for try in 1 2 3 4 5 6; do
    ( cd "$ENVS" && MODAL_SECRETS=openrouter-secret nohup env -u VIRTUAL_ENV uv run modal run --detach scripts/resume_modal.py \
        --step-dir "$step" --count "$n" --use-current --max-concurrent 0 --no-judge --timeout 7200 \
        --results-dir "$out" --overrides "$GEMMA" > "$log" 2>&1 & ); sleep 30
    grep -q "rate limit exceeded" "$log" || break
    sleep $((15 * try))
  done
  echo "$(date '+%F %T') launched $(basename "$log") x$n" >> "$LOG/cost.txt"
done < "$LOG/commands.txt"

# ---- C: gemma's own full runs, whole batches, all at once
MAXC=0 GAP=30 bash "$HERE/scripts/modal_full.sh" full_fix gemma-3-27b ai_rd:10 puppeteer:5 revert_or_refactor:5
echo "$(date '+%F %T') all launched" >> "$LOG/cost.txt"
