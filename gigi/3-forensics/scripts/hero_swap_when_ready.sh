#!/usr/bin/env bash
# Launch the hero swap (scripts/hero_swap.sh: arms llama = control, qwen36, gemma) by itself once Llama 3.3 70B's
# gap-environment solo runs are graded (4 Nov 2026, approved by the user). Detached, sleep-safe (caffeinate -i, logs to
# files, timeouts, no questions), idempotent (rerun it any time: a chain already running or already launched is left
# alone). Every step is logged to results/logs/hero_swap/chain.txt.
#   1. wait  until results/hero_loss_audit.md exists (the fold-in scripts/hero_solo_foldin.sh is done) AND
#            results/hero_full_runs.csv has >= MIN_SCORED scored runs (column n) for each of the 8 gap environments
#            (REQUIRE_ENVS). An environment still at n=0 there (e.g. ai_rd, whose stage hero_solo_key2 is folded in by
#            a second fold-in: HERO_MERGE=1 HERO_STAGES=hero_solo_key2 ...) is also accepted when this script's own
#            scoring over PLAN_STAGES (full_runs_table.py --hero into results/logs/hero_swap/ready/, local, free) has
#            n >= MIN_SCORED for it. Polls every POLL_SECONDS (300) for up to WAIT_MINUTES (2880); gives up loudly.
#   2. plan  scripts/hero_swap_plan.py --model llama-3.3-70b --stages $PLAN_STAGES (hero_smoke hero_solo
#            hero_solo_key2): dry run first (logged, with the number of incident and clean points), then the real one
#            (results/hero_swap_plan.json, exclusive create; an existing plan is used as is). Zero incident points =
#            no cheating moments: logged LOUDLY, the clean points are still launched. Zero points at all: stop.
#   3. guard scripts/hero_usage.py (two free GETs: the key's remaining limit and the org's credit balance), then the
#            swap's cost from the plan: per restart point and arm, reps x (TURNS x prefix_tokens input + OUT_TOKENS
#            output), i.e. a 10-turn continuation that re-sends the prefix every turn and writes 3k tokens in all, at
#            llama $0.71/$0.71 per M (CoreWeave fp16), qwen36 $0.32/$3.20 (DeepInfra fp8), gemma $0.08/$0.16
#            (DeepInfra fp8); backups gemma4 $0.14/$0.40, gptoss $0.03/$0.14. Refuses to launch, and says why, when
#            GUARD_FACTOR (1.5) x the estimate exceeds the smaller of the key's remaining limit and the credit balance
#            (SPEND_GUARD_USD=<number> replaces that budget; an unlimited key counts as no bound; unknown numbers refuse).
#   4. launch bash scripts/hero_swap.sh --arms $ARMS (every continuation of a point at once: --max-concurrent 0 is
#            built in; 4 launches in parallel; retried on Modal's app-creation limit), after its free --dry-run check of
#            the merged configs passed. Then stays up (caffeinate) while our resume_modal.py launchers run, up to
#            LAUNCHER_WAIT_MINUTES (720), so they can download each fleet when its app ends. Writes
#            results/logs/hero_swap/chain_launched.txt: the marker that makes a rerun a no-op.
# Nothing under results/ is deleted or overwritten: new files only (the plan by exclusive create, logs by append or
# stamped names). Does NOT commit or push. Every launch spends money: the user approved this swap on 4 Nov 2026.
#   bash gigi/3-forensics/scripts/hero_swap_when_ready.sh            detaches itself (nohup caffeinate -i), returns at once
#   DRY_RUN=1 bash gigi/3-forensics/scripts/hero_swap_when_ready.sh  foreground, no network, no Modal: the readiness check
#            (once), the plan builder (dry run, then written to a scratch folder), the estimate against USAGE_FAKE="<key
#            left> <credits left>" (hero_usage.py is not called), hero_swap.sh --dry-run; launches nothing.
# Env: PLAN_STAGES (space list), ARMS, REQUIRE_ENVS (comma list), MIN_SCORED, WAIT_MINUTES, POLL_SECONDS,
# LAUNCHER_WAIT_MINUTES, SPEND_GUARD_USD, GUARD_FACTOR, TURNS, OUT_TOKENS, PRICE_<ARM>_IN / _OUT ($ per M tokens),
# PLAN_FILE (use this plan instead of building one; passed to hero_swap.sh --plan), HERO_SWAP_LOG_DIR, USAGE_FAKE.
# HERO2=1 (4 Nov 2026): the same chain on Llama's second pass with the end-of-run confirmation (stages hero2_llama +
# hero2_llama_key, folded in per model by hero_solo_foldin.sh / hero2_driver.sh). Readiness is read from the per-model
# files results/hero2_loss_audit__llama-3.3-70b.md and results/hero2_full_runs__llama-3.3-70b.csv (READY_AUDIT /
# READY_CSV override either), the plan is results/hero2_swap_plan.json (exclusive create), the logs go to
# results/logs/hero2_swap/ and hero_swap.sh runs in its HERO2 mode (agent.end_confirm=true in every swapped-in arm,
# replays in results/raw/hero2_swap/). The first pass's files and folders are not touched. Without HERO2 nothing changes.
#   HERO2=1 PLAN_STAGES="hero2_llama hero2_llama_key" bash gigi/3-forensics/scripts/hero_swap_when_ready.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=${HERO_MODEL:-llama-3.3-70b}; DRY=${DRY_RUN:-0}; HERO2=${HERO2:-0}; export HERO2
if [ "$HERO2" = 1 ]; then
  PLAN_STAGES=${PLAN_STAGES:-hero2_llama hero2_llama_key}; SWAPNAME=hero2_swap
  READY_AUDIT=${READY_AUDIT:-$HERE/results/hero2_loss_audit__$MODEL.md}; READY_CSV=${READY_CSV:-$HERE/results/hero2_full_runs__$MODEL.csv}
  DEFAULT_PLAN="$HERE/results/hero2_swap_plan.json"
else
  PLAN_STAGES=${PLAN_STAGES:-hero_smoke hero_solo hero_solo_key2}; SWAPNAME=hero_swap
  READY_AUDIT=${READY_AUDIT:-$HERE/results/hero_loss_audit.md}; READY_CSV=${READY_CSV:-$HERE/results/hero_full_runs.csv}
  DEFAULT_PLAN="$HERE/results/hero_swap_plan.json"
fi
ARMS=${ARMS:-llama,qwen36,gemma}
REQUIRE_ENVS=${REQUIRE_ENVS:-ai_rd,coding,coding_norules,dashboard_costly_failure,grade_your_own,optimize_rendering_engine,optimize_rendering_engine_graded,precommit_hook}
MIN_SCORED=${MIN_SCORED:-1}; WAIT=${WAIT_MINUTES:-2880}; POLL=${POLL_SECONDS:-300}; LWAIT=${LAUNCHER_WAIT_MINUTES:-720}
GUARD_FACTOR=${GUARD_FACTOR:-1.5}; TURNS=${TURNS:-10}; OUT_TOKENS=${OUT_TOKENS:-3000}
export PRICE_LLAMA_IN=${PRICE_LLAMA_IN:-0.71} PRICE_LLAMA_OUT=${PRICE_LLAMA_OUT:-0.71}
export PRICE_QWEN36_IN=${PRICE_QWEN36_IN:-0.32} PRICE_QWEN36_OUT=${PRICE_QWEN36_OUT:-3.20}
export PRICE_GEMMA_IN=${PRICE_GEMMA_IN:-0.08} PRICE_GEMMA_OUT=${PRICE_GEMMA_OUT:-0.16}
export PRICE_GEMMA4_IN=${PRICE_GEMMA4_IN:-0.14} PRICE_GEMMA4_OUT=${PRICE_GEMMA4_OUT:-0.40}
export PRICE_GPTOSS_IN=${PRICE_GPTOSS_IN:-0.03} PRICE_GPTOSS_OUT=${PRICE_GPTOSS_OUT:-0.14}
if [ "$DRY" = 1 ]; then
  L=${HERO_SWAP_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/hero_swap_chain_dry.XXXXXX")}
else
  L=${HERO_SWAP_LOG_DIR:-$HERE/results/logs/$SWAPNAME}
fi
mkdir -p "$L"; CHAIN="$L/chain.txt"; STAMP=$(date '+%Y%m%d-%H%M%S')
say () { echo "$(date '+%F %T') $*" >> "$CHAIN"; [ "$DRY" = 1 ] && echo "$*"; return 0; }
loud () { say "!!!!!!!!!! $*"; say "!!!!!!!!!! $*"; say "!!!!!!!!!! $*"; }

# ---- detach (real runs only): the child does the work under caffeinate, this shell returns at once ----
if [ "$DRY" != 1 ] && [ -z "${HERO_CHAIN_CHILD:-}" ]; then
  if [ -f "$L/chain.pid" ] && kill -0 "$(cat "$L/chain.pid")" 2>/dev/null; then
    echo "hero_swap_when_ready: a chain is already running (pid $(cat "$L/chain.pid")); log $CHAIN"; exit 0
  fi
  HERO_CHAIN_CHILD=1 nohup caffeinate -i bash "$0" "$@" >> "$L/chain_stdout.txt" 2>&1 &
  disown 2>/dev/null; echo "hero_swap_when_ready: detached (pid $!); log $CHAIN"; exit 0
fi
if [ "$DRY" != 1 ]; then
  if [ -f "$L/chain.pid" ] && [ "$(cat "$L/chain.pid")" != $$ ] && kill -0 "$(cat "$L/chain.pid")" 2>/dev/null; then
    say "another chain is running (pid $(cat "$L/chain.pid")): exiting"; exit 0
  fi
  echo $$ > "$L/chain.pid"; trap 'rm -f "$L/chain.pid"' EXIT
fi
say "start (model $MODEL, plan stages: $PLAN_STAGES, arms $ARMS, require $REQUIRE_ENVS >= $MIN_SCORED scored$([ "$HERO2" = 1 ] && echo ", HERO2: confirmation on, readiness from $(basename "$READY_CSV")")$([ "$DRY" = 1 ] && echo ", DRY RUN"))"

# ---- idempotency: launched already? ----
if [ "$DRY" != 1 ] && { [ -f "$L/chain_launched.txt" ] || grep -q "all launched" "$L/launch.txt" 2>/dev/null; }; then
  say "the swap was launched already ($(head -1 "$L/chain_launched.txt" 2>/dev/null || grep "all launched" "$L/launch.txt" | tail -1)): nothing to do"; exit 0
fi

# ---- 1. wait until the gap environments are scored ----
ready () {   # exit 0 when every required environment has >= MIN_SCORED scored runs; prints what is missing
  [ -f "$READY_AUDIT" ] || { echo "$(basename "$READY_AUDIT") not written yet"; return 1; }
  python3 - "$HERE" "$L" "$REQUIRE_ENVS" "$MIN_SCORED" "${PLAN_STAGES// /,}" "$READY_CSV" "$MODEL" <<'PY'
import csv, os, pathlib, subprocess, sys
H, L, envs, need, stages = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4]), sys.argv[5]
ready_csv, model = pathlib.Path(sys.argv[6]), sys.argv[7]

def scored(f):
    if not f.exists():
        return {}
    return {r["environment"]: int(r["n"] or 0) for r in csv.DictReader(open(f))}

main = scored(ready_csv)
missing = [e for e in envs if main.get(e, 0) < need]
if not main:
    print(f"{ready_csv.name} not written yet"); sys.exit(1)
if missing:   # own scoring over the plan stages (local, free), for environments the fold-in's stage set does not cover
    (L / "ready").mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(H / "scripts" / "full_runs_table.py"), "--hero"], cwd=H, capture_output=True, text=True,
                       env={**os.environ, "HERO_STAGES": stages, "HERO_OUT_DIR": str(L / "ready"), "HERO_MODEL": model,
                            "HERO_OUT_PREFIX": "hero_full_runs"}, timeout=1800)
    own = scored(L / "ready" / "hero_full_runs.csv") if r.returncode == 0 else {}
    accepted = [e for e in missing if own.get(e, 0) >= need]
    if accepted:
        print(f"accepted from own scoring over {stages}: " + ", ".join(f"{e} n={own[e]}" for e in accepted))
    missing = [e for e in missing if e not in accepted]
if missing:
    print("waiting for scored runs in: " + ", ".join(f"{e} (n={main.get(e, 0)})" for e in missing)); sys.exit(1)
print("ready: " + ", ".join(f"{e} n={main.get(e, 0)}" for e in envs))
PY
}
i=0; READY=0
while :; do
  msg=$(ready 2>&1) && { READY=1; say "$msg"; break; }
  [ "$DRY" = 1 ] && { say "dry run: not ready ($msg); continuing anyway"; break; }
  [ $((i % 6)) = 0 ] && say "waiting: $msg"
  i=$((i + 1)); [ $((i * POLL / 60)) -ge "$WAIT" ] && break
  sleep "$POLL"
done
if [ "$DRY" != 1 ] && [ $READY = 0 ]; then loud "NOT READY after $WAIT min ($msg): giving up, nothing launched; rerun this script"; exit 1; fi

# ---- 2. the plan ----
PLAN=${PLAN_FILE:-$DEFAULT_PLAN}
PLAN_ARGS=(--model "$MODEL" --stages $PLAN_STAGES)
# a plan under results/ with another name than hero_swap_plan.json (HERO2: hero2_swap_plan.json) is written by
# --force-new-name (exclusive create, like the default); a plan elsewhere by --out
if [ "$DRY" != 1 ] && [ "$(basename "$PLAN")" != hero_swap_plan.json ]; then
  if [ "$(cd "$(dirname "$PLAN")" && pwd)" = "$HERE/results" ]; then PLAN_ARGS+=(--force-new-name "$(basename "$PLAN")"); else PLAN_ARGS+=(--out "$PLAN"); fi
fi
if [ -f "$PLAN" ]; then
  say "plan exists, used as is: $PLAN"
else
  if python3 "$HERE/scripts/hero_swap_plan.py" --model "$MODEL" --stages $PLAN_STAGES --dry-run > "$L/plan_dry_$STAMP.txt" 2>&1; then
    say "plan dry run: $(head -1 "$L/plan_dry_$STAMP.txt")"
    say "plan dry run: $(python3 -c '
import re, sys; t = open(sys.argv[1]).read()
pts = sum(int(m) for m in re.findall(r"points=\s*(\d+)\s+incident points=", t)); inc = sum(int(m) for m in re.findall(r"incident points=\s*(\d+)", t))
print(f"{inc} incident points, {pts - inc} clean points")' "$L/plan_dry_$STAMP.txt") (full output: $L/plan_dry_$STAMP.txt)"
  else
    loud "plan dry run FAILED (exit $?): $(tail -3 "$L/plan_dry_$STAMP.txt" | tr '\n' ' '); nothing launched"; exit 1
  fi
  [ "$DRY" = 1 ] && PLAN="$L/$(basename "$DEFAULT_PLAN")" && PLAN_ARGS+=(--out "$PLAN")
  if python3 "$HERE/scripts/hero_swap_plan.py" "${PLAN_ARGS[@]}" > "$L/plan_build_$STAMP.txt" 2>&1 && [ -f "$PLAN" ]; then
    say "plan written: $PLAN"
  else
    loud "plan NOT written (exit $?): $(tail -2 "$L/plan_build_$STAMP.txt" | tr '\n' ' '); nothing launched"; exit 1
  fi
fi
counts=$(python3 -c '
import json, sys; p = json.load(open(sys.argv[1]))
print(len(p), sum(x["incident"] for x in p), sum(not x["incident"] for x in p), len({x["env"] for x in p}), sum(x["reps"] for x in p))' "$PLAN")
read -r N_POINTS N_INC N_CLEAN N_ENVS N_REPS <<< "$counts"
say "plan: $N_POINTS restart points ($N_INC incident, $N_CLEAN clean) in $N_ENVS environments, $N_REPS replays per arm at launch"
[ "$N_POINTS" -gt 0 ] || { loud "the plan has NO restart points: nothing launched"; exit 1; }
[ "$N_INC" -gt 0 ] || loud "NO CHEATING MOMENTS: the plan has 0 incident points (Llama never cheated in a scored run); launching the $N_CLEAN clean points only"

# ---- 3. budget: real numbers, then the estimate ----
KEY_LEFT=unknown; CREDITS_LEFT=unknown
if [ "$DRY" = 1 ]; then
  if [ -n "${USAGE_FAKE:-}" ]; then read -r KEY_LEFT CREDITS_LEFT <<< "$USAGE_FAKE"; say "dry run: hero_usage.py not called; USAGE_FAKE key left \$$KEY_LEFT, credits left \$$CREDITS_LEFT"
  else say "dry run: hero_usage.py not called (set USAGE_FAKE='<key left> <credits left>' to test the guard)"; fi
else
  for try in 1 2 3; do
    python3 "$HERE/scripts/hero_usage.py" > "$L/usage_$STAMP.txt" 2>&1
    grep -q "key limit left" "$L/usage_$STAMP.txt" && grep -q "org credits:.*remaining" "$L/usage_$STAMP.txt" && break
    say "hero_usage.py try $try: $(tr '\n' ' ' < "$L/usage_$STAMP.txt" | cut -c1-200)"; sleep 60
  done
  KEY_LEFT=$(grep "key limit left" "$L/usage_$STAMP.txt" | sed -E 's/.*: *\$?//; s/,//g' | awk '{print $1}')
  CREDITS_LEFT=$(grep "org credits:" "$L/usage_$STAMP.txt" | grep -oE '\$[0-9,.]+ remaining' | tr -d '$,' | awk '{print $1}')
  say "usage: $(grep -E 'key limit left|since 18 Oct|org credits' "$L/usage_$STAMP.txt" | sed -E 's/ +/ /g' | tr '\n' ';')"
fi
python3 - "$PLAN" "$ARMS" "$TURNS" "$OUT_TOKENS" "$GUARD_FACTOR" "${KEY_LEFT:-unknown}" "${CREDITS_LEFT:-unknown}" "${SPEND_GUARD_USD:-}" "$L/estimate_$STAMP.json" <<'PY' > "$L/estimate_$STAMP.txt" 2>&1
import json, os, sys
plan, arms, turns, out_tok, factor = json.load(open(sys.argv[1])), sys.argv[2].split(","), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
key_left, credits_left, guard, out = sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]
prices = {}
for a in arms:
    try:
        prices[a] = (float(os.environ[f"PRICE_{a.upper()}_IN"]), float(os.environ[f"PRICE_{a.upper()}_OUT"]))
    except KeyError:
        print(f"REFUSE: no price for arm {a} (set PRICE_{a.upper()}_IN and _OUT, $ per M tokens)"); sys.exit(2)
known = [p["prefix_tokens"] for p in plan if p.get("prefix_tokens")]
fallback = max(known) if known else 100_000
none = sum(1 for p in plan if not p.get("prefix_tokens"))
per_arm, per_env = {a: 0.0 for a in arms}, {}
for p in plan:
    T = p.get("prefix_tokens") or fallback
    for a in arms:
        c = p["reps"] * (turns * T * prices[a][0] + out_tok * prices[a][1]) / 1e6
        per_arm[a] += c; per_env[p["env"]] = per_env.get(p["env"], 0.0) + c
total = sum(per_arm.values()); guarded = factor * total
tries = sum(p["reps"] for p in plan)
print(f"estimate: {len(plan)} points x reps = {tries} tries per arm; prefix tokens {min(known) if known else '?'}-{max(known) if known else '?'}"
      f" (mean {round(sum(known) / len(known)) if known else '?'}), {none} points without a prefix size (counted as {fallback})")
print(f"estimate: input per try = {turns:g} turns x prefix, output {out_tok:g} tokens; " + ", ".join(f"{a} ${per_arm[a]:,.2f} (${prices[a][0]}/${prices[a][1]} per M)" for a in arms))
print("estimate by environment: " + ", ".join(f"{e} ${c:,.2f}" for e, c in sorted(per_env.items(), key=lambda x: -x[1])))
print(f"estimate total ${total:,.2f}; x{factor:g} = ${guarded:,.2f}")

def num(s):
    try:
        return float(s)
    except ValueError:
        return None
if guard:
    budget, why = num(guard), f"SPEND_GUARD_USD={guard}"
    if budget is None:
        print(f"REFUSE: SPEND_GUARD_USD is not a number: {guard!r}"); sys.exit(2)
else:
    k, c = num(key_left), num(credits_left)
    if key_left.lower() in ("none", "unlimited"):   # no limit on the key: only the credits bound it
        k = None if c is None else c
    if k is None and c is None:
        print(f"REFUSE: the key's remaining limit ({key_left}) and the credit balance ({credits_left}) are unknown; "
              "fix hero_usage.py's read or pass SPEND_GUARD_USD=<number> to launch"); sys.exit(2)
    cands = [x for x in (k, c) if x is not None]
    budget = min(cands); why = f"min(key limit left {key_left}, credits left {credits_left})" if len(cands) == 2 else f"the one known of (key {key_left}, credits {credits_left})"
json.dump({"points": len(plan), "tries_per_arm": tries, "turns": turns, "out_tokens": out_tok, "prices_per_M": prices, "per_arm_usd": per_arm,
           "per_env_usd": per_env, "total_usd": total, "guard_factor": factor, "guarded_usd": guarded, "budget_usd": budget, "budget_from": why,
           "key_left": key_left, "credits_left": credits_left}, open(out, "w"), indent=1)
if guarded > budget:
    print(f"REFUSE: {factor:g} x estimate ${guarded:,.2f} exceeds the budget ${budget:,.2f} ({why}); nothing launched. "
          f"Lower the plan (fewer reps or points), raise the cap/credits, or rerun with SPEND_GUARD_USD=<number> after the user's explicit yes"); sys.exit(2)
print(f"OK to launch: {factor:g} x estimate ${guarded:,.2f} <= budget ${budget:,.2f} ({why})")
PY
GUARD_RC=$?
while read -r line; do say "$line"; done < "$L/estimate_$STAMP.txt"
[ $GUARD_RC = 0 ] || { loud "SPEND GUARD: not launching (see above)"; exit 2; }

# ---- 4. launch ----
SWAP_ARGS=(--arms "$ARMS"); [ -n "${PLAN_FILE:-}" ] && SWAP_ARGS+=(--plan "$PLAN_FILE"); [ "$DRY" = 1 ] && [ -z "${PLAN_FILE:-}" ] && SWAP_ARGS+=(--plan "$PLAN")
# hero_swap.sh reads HERO2 from the environment (exported above): its plan default, raw and log folders and the
# end_confirm override follow; a dry run's temp plan is passed explicitly
if bash "$HERE/scripts/hero_swap.sh" "${SWAP_ARGS[@]}" --dry-run > "$L/swap_dry_$STAMP.txt" 2>&1; then
  say "hero_swap.sh dry run OK: $(head -1 "$L/swap_dry_$STAMP.txt"); merged configs: $(grep -E '^  (llama|qwen36|gemma|gemma4|gptoss) ' "$L/swap_dry_$STAMP.txt" | tr -s ' ' | tr '\n' ';' | cut -c1-600)"
else
  loud "hero_swap.sh --dry-run FAILED (exit $?): $(tail -3 "$L/swap_dry_$STAMP.txt" | tr '\n' ' '); nothing launched"; exit 1
fi
if [ "$DRY" = 1 ]; then
  say "dry run: would launch now: bash scripts/hero_swap.sh ${SWAP_ARGS[*]} (all continuations of a point at once: --max-concurrent 0)"
  echo "dry-run outputs in $L"; exit 0
fi
say "launching: bash scripts/hero_swap.sh ${SWAP_ARGS[*]} (every continuation at once, --max-concurrent 0; 4 launches in parallel)"
bash "$HERE/scripts/hero_swap.sh" "${SWAP_ARGS[@]}" > "$L/swap_launch_$STAMP.txt" 2>&1; rc=$?
say "hero_swap.sh exit $rc: $(head -1 "$L/swap_launch_$STAMP.txt"); launch log $L/launch.txt, arms $L/arms.txt"
if [ $rc = 0 ]; then   # the marker that makes a rerun a no-op; not written after a failed launch, so a rerun retries it
  { echo "$(date '+%F %T') launched by hero_swap_when_ready.sh: plan $PLAN, arms $ARMS"; cat "$L/estimate_$STAMP.txt"; } > "$L/chain_launched.txt"
else
  loud "hero_swap.sh exited $rc: check $L/swap_launch_$STAMP.txt; rerun this script (hero_swap.sh skips the points already launched)"
fi
# stay awake while our launchers run (they download each fleet when its app ends); the fold-in re-fetches anyway
for i in $(seq 1 "$LWAIT"); do
  n=$(pgrep -f "resume_modal.py .*results-dir .*/$SWAPNAME/" | wc -l | tr -d ' ')
  [ "$n" = 0 ] && break
  [ $((i % 30)) = 1 ] && say "$n launcher process(es) still running (downloading when their apps end); staying awake"
  sleep 60
done
say "done: the swap is launched; next: swap_all_score.py --hero fetch, grading (grade_modal.py SWAP_HERO), scoring, the loss audit (the parent session)"
exit 0
