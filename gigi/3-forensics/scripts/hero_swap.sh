#!/usr/bin/env bash
# The hero swap (4 Nov 2026; plan: results/hero_swap_plan.json from scripts/hero_swap_plan.py). Modelled on
# swap_all.sh: each restart point on Llama 3.3 70B's own runs is resumed on Modal once per arm, from the current tree
# (--use-current), every continuation started at once, 4 launches in parallel, each retried while Modal's
# app-creation limit turns it away. Arms (the `who` folder name):
#   llama   no override: the checkpoint fleet's own config (configs/<env>__llama-3.3-70b.yaml: CoreWeave fp16), the control
#   qwen36  Qwen3.6 27B, native tools, thinking OFF (reasoning_effort=disabled: patches/hero_reasoning_off.patch)
#   gemma   gemma-3-27b-it, text mode (no tools sent), as in the Qwen swap
#   gemma4, gptoss   optional backups, only with --backups (or named in --arms)
# Results: results/raw/hero_swap/<env>/<point>/<who>/; logs: results/logs/hero_swap/ (arms.txt records each arm's
# override string and precision at every launch).
#   bash hero_swap.sh --dry-run                 write the command file, check every arm's merged config, launch nothing
#   bash hero_swap.sh [--backups] [--arms llama,qwen36,gemma] [--plan results/<plan>.json] [commands_file]
#   bash hero_swap.sh --topup [--dry-run]       per point and arm, launch `target` minus the replays already scored
#                                               (results/hero_gap_replays.csv; run swap_all_score.py --hero fetch,
#                                               swap_all_score.py --hero and gap_replays.py --hero first) minus the
#                                               ones lost to the context window (a rerun would lose them again)
# Every launch spends money (OpenRouter, and Opus grading later): ask first, every time (gigi/CLAUDE.md).
# MODAL_PROFILE / MODAL_SECRET_SUFFIX / MODAL_VOLUME_PREFIX default to the SPAR workspace values of topup60.sh.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$HERE/results/logs/hero_swap"
export MODAL_PROFILE=${MODAL_PROFILE:-spar} MODAL_SECRET_SUFFIX=${MODAL_SECRET_SUFFIX:--gigi} MODAL_VOLUME_PREFIX=${MODAL_VOLUME_PREFIX:-gigi-forensics-}

# ---- override strings (format of swap_all.sh: one `=` per item, no commas inside a value, single-element lists) ----
# Merged onto the checkpoint fleet's config.yaml (Llama's: only [coreweave], quantizations [fp16], allow_fallbacks
# false, reasoning_effort none, temperature 0.6, top_p 0.9), so every arm names its own provider block.
# Endpoint facts: OpenRouter GET /api/v1/models/<id>/endpoints, read without a key on 4 Nov 2026.
# Qwen3.6 27B: fp8, no full-precision endpoint (all 7 endpoints are fp8 or unlabelled; none bf16/fp16). DeepInfra fp8
#   (ctx 262,144, max out 81,920, $0.32/$3.20; the only one besides Phala with min_p). Non-thinking card set
#   (temperature 0.7, top_p 0.8, top_k 20, min_p 0); presence_penalty 1.5 cannot be sent (no harness knob).
#   No require_parameters: the harness sends parallel_tool_calls, which no Qwen3.6 endpoint lists (the reason
#   make_configs_hero.py leaves it off Llama's CoreWeave pin); only + quantizations already pick one endpoint, and it
#   lists every parameter we send (temperature, top_p, top_k, min_p, reasoning, tools).
QWEN36="agent.model=qwen/qwen3.6-27b,agent.temperature=0.7,agent.top_p=0.8,agent.top_k=20,agent.min_p=0,agent.text_tools=false,agent.reasoning_effort=disabled,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[fp8],agent.provider_preferences.allow_fallbacks=false"
# gemma-3-27b-it: fp8. The Novita bf16 endpoint (98,304 context) of 3 Nov is no longer listed: the four endpoints
#   are DeepInfra, NextBit, Parasail and Nebius, all fp8. DeepInfra fp8 (ctx 131,072, max out 16,384, $0.08/$0.16,
#   temperature, top_p, top_k and min_p supported) as in every gemma-3 result. Text mode sends no tools and no
#   parallel_tool_calls, so require_parameters=true is safe and guarantees top_k is honoured. Context risk: Llama's
#   prefixes reach ~128k tokens (CoreWeave's window), and gemma-3's is 131,072 including the reply, with the text-mode
#   tool preamble on top: the longest dashboard prefixes can be lost to the context window (hero_swap_plan.py flags
#   points above ~115k as near_gemma_ctx).
GEMMA="agent.model=google/gemma-3-27b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=true,agent.reasoning_effort=none,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[fp8],agent.provider_preferences.allow_fallbacks=false,agent.provider_preferences.require_parameters=true"
# gemma 4 31B (backup): bf16 on Crusoe (ctx 262,144, max out 262,141, $0.14/$0.40, top_k and min_p), gemma4_prep.md's
#   string with require_parameters dropped for the parallel_tool_calls reason above (Crusoe does not list it).
GEMMA4="agent.model=google/gemma-4-31b-it,agent.temperature=1.0,agent.top_p=0.95,agent.top_k=64,agent.min_p=null,agent.text_tools=false,agent.reasoning_effort=none,agent.provider_preferences.only=[crusoe],agent.provider_preferences.quantizations=[bf16],agent.provider_preferences.allow_fallbacks=false"
# GPT-OSS-20B (backup): DeepInfra bf16 (an upcast of the native MXFP4 weights; ctx 131,072, $0.03/$0.14, top_k and
#   min_p); reasoning is mandatory, so the effort is named (medium = OpenRouter's default); README sampling 1.0 / 1.0.
GPTOSS="agent.model=openai/gpt-oss-20b,agent.temperature=1.0,agent.top_p=1.0,agent.top_k=null,agent.min_p=null,agent.text_tools=false,agent.reasoning_effort=medium,agent.provider_preferences.only=[deepinfra],agent.provider_preferences.quantizations=[bf16],agent.provider_preferences.allow_fallbacks=false"

ARMS="llama,qwen36,gemma"; PLAN="$HERE/results/hero_swap_plan.json"; DRY=0; TOPUP=0; CMDS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --backups) ARMS="$ARMS,gemma4,gptoss" ;;
    --arms) ARMS="$2"; shift ;;
    --plan) PLAN="$2"; case "$PLAN" in /*) ;; *) PLAN="$HERE/$PLAN" ;; esac; shift ;;
    --dry-run) DRY=1 ;;
    --topup) TOPUP=1 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) CMDS="$1" ;;
  esac; shift
done
[ -f "$PLAN" ] || [ -n "$CMDS" ] || { echo "no plan at $PLAN: run scripts/hero_swap_plan.py first" >&2; exit 1; }
[ $DRY = 1 ] || mkdir -p "$LOG"
STAMP=$(date '+%Y%m%d-%H%M%S')
if [ -z "$CMDS" ]; then   # a new, stamped command file per launch (nothing under results/ is overwritten); a dry run writes to a temp file
  if [ $DRY = 1 ]; then CMDS=$(mktemp "${TMPDIR:-/tmp}/hero_swap_commands.XXXXXX"); else
    mkdir -p "$LOG"; CMDS="$LOG/commands$([ $TOPUP = 1 ] && echo "_topup")_$STAMP.txt"; fi
  python3 - "$HERE" "$PLAN" "$ARMS" "$TOPUP" "$STAMP" "$QWEN36" "$GEMMA" "$GEMMA4" "$GPTOSS" > "$CMDS" <<'PY' || exit 1
import collections, csv, json, os, pathlib, shlex, sys
here, plan, arms, topup, stamp = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(","), sys.argv[4] == "1", sys.argv[5]
OV = {"llama": "", "qwen36": sys.argv[6], "gemma": sys.argv[7], "gemma4": sys.argv[8], "gptoss": sys.argv[9]}
bad = [a for a in arms if a not in OV]
if bad:
    sys.exit(f"unknown arm(s) {bad}; known: {list(OV)}")
scored, ctx = collections.Counter(), collections.Counter()
if topup:
    f = pathlib.Path(os.environ.get("HERO_GAP_CSV", here / "results" / "hero_gap_replays.csv"))   # override: tests only
    if not f.exists():
        sys.exit("--topup needs results/hero_gap_replays.csv (swap_all_score.py --hero fetch; gap_replays.py --hero)")
    for r in csv.DictReader(open(f)):
        k = (r["env"], r["point"], r["who"])
        scored[k] += r["hacked"] in ("0", "1"); ctx[k] += r["status"] == "context window exceeded"
for p in json.load(open(plan)):
    run = here / p["run"]; point = f"{run.parent.name}_{run.name}_K{p['K']}"
    for who in arms:
        n = p["reps"]
        if topup:
            n = p.get("target", p["reps"]) - scored[(p["env"], point, who)] - ctx[(p["env"], point, who)]
            if n <= 0:
                continue
        out = here / "results/raw/hero_swap" / p["env"] / point / who
        log = here / "results/logs/hero_swap" / (f"{p['env']}__{point}__{who}" + (f"__topup_{stamp}" if topup else "") + ".log")
        args = ["--step-dir", str(run / f"step-{p['K']}"), "--count", str(n), "--use-current", "--max-concurrent", "0",
                "--no-judge", "--timeout", "7200", "--results-dir", str(out)] + (["--overrides", OV[who]] if OV[who] else [])
        print(shlex.quote(str(log)) + " " + " ".join(shlex.quote(a) for a in args))
PY
fi
N_RES=$(wc -l < "$CMDS" | tr -d ' ')
N_TRIES=$(grep -o -- "--count [0-9]*" "$CMDS" | awk '{s += $2} END {print s + 0}')
echo "$N_RES resumes, $N_TRIES replays (arms: $ARMS) -> $CMDS"
for a in $(echo "$ARMS" | tr ',' ' '); do
  printf "  %-7s %4s replays\n" "$a" "$(grep -E -- "__${a}(__topup_[0-9-]+)?\.log " "$CMDS" | grep -o -- "--count [0-9]*" | awk '{s += $2} END {print s + 0}')"
done

if [ $DRY = 1 ]; then
  # free check: each arm's override merged onto a Llama config exactly as resume_modal.py does (run_common.load_config)
  ( cd "$HERE/agent-interp-envs" && env -u VIRTUAL_ENV uv run python - "$HERE/configs/coding__llama-3.3-70b.yaml" \
      "$ARMS" "$QWEN36" "$GEMMA" "$GEMMA4" "$GPTOSS" <<'PY'
import sys
from omegaconf import OmegaConf
sys.path.insert(0, "scripts")
from run_common import load_config, parse_overrides
cfg, arms = sys.argv[1], sys.argv[2].split(",")
OV = {"llama": "", "qwen36": sys.argv[3], "gemma": sys.argv[4], "gemma4": sys.argv[5], "gptoss": sys.argv[6]}
KEYS = ("model", "temperature", "top_p", "top_k", "min_p", "text_tools", "reasoning_effort", "provider_preferences")
for a in arms:
    ag = OmegaConf.to_container(load_config(cfg, parse_overrides(OV[a]) or None).agent)
    print(f"  {a:7s}", {k: ag.get(k) for k in KEYS})
    assert ag["provider_preferences"].get("allow_fallbacks") is False, a
    assert isinstance(ag.get("reasoning_effort"), str), (a, ag.get("reasoning_effort"))
PY
  ) || { echo "merged-config check FAILED" >&2; exit 1; }
  echo "dry run: nothing launched"; exit 0
fi

{ echo "$(date '+%F %T') launching $N_RES resumes from $(basename "$CMDS"), arms $ARMS"
  echo "  llama  (no override: the fleet's own config, CoreWeave fp16)"
  echo "  qwen36 fp8, no full-precision endpoint: $QWEN36"
  echo "  gemma  fp8 (no bf16 endpoint listed 4 Nov): $GEMMA"
  echo "  gemma4 bf16: $GEMMA4"
  echo "  gptoss bf16 (upcast of MXFP4): $GPTOSS"; } >> "$LOG/arms.txt"
echo "$(date '+%F %T') launching $N_RES resumes from $(basename "$CMDS")" >> "$LOG/launch.txt"
launch_one () {
  log=$1; shift
  [ -f "$log" ] && grep -q "Launching\|Resuming\|Checkpoint:" "$log" && ! grep -q "rate limit exceeded" "$log" && return 0
  for try in 1 2 3 4 5 6 7 8; do
    ( cd "$HERE/agent-interp-envs" && MODAL_SECRETS=${MODAL_SECRETS:-openrouter-secret} nohup env -u VIRTUAL_ENV uv run modal run --detach \
        scripts/resume_modal.py "$@" > "$log" 2>&1 & ); sleep 25
    grep -q "rate limit exceeded" "$log" || break
    sleep $((10 * try))
  done
  echo "$(date '+%F %T') launched $(basename "$log")" >> "$(dirname "$log")/launch.txt"
}
export -f launch_one; export HERE
xargs -P 4 -L 1 bash -c 'eval "launch_one $*"' _ < "$CMDS"
echo "$(date '+%F %T') all launched ($(basename "$CMDS"))" >> "$LOG/launch.txt"
