"""Grade the Modal pass with the environments' own graders (Claude Opus 5, via OpenRouter pinned to Anthropic).

Collects every fleet of the STAGES launches (default full_modal, full_modal_rest, full_modal_retry), groups them by the
environment that has a grader (agent_interp_envs ENV_JUDGES), runs that grader on them, and checks the
OpenRouter key's spend after each environment: past GRADE_CAP_USD (default 60) nothing more is started.
Graders skip runs already graded, so a rerun only does what is missing. JUDGE_MODEL (e.g. claude-opus-4-8,
Anthropic's fallback for Opus 5 refusals) grades only what is still ungraded; grading.json records the judge.
Log: results/logs/grading/grade.txt.
  FORENSICS_ENV_FILE=... python3 gigi/3-forensics/scripts/grade_modal.py
HERO_SOLO=1 (4 Nov 2026): instead, grade only Llama 3.3 70B's solo runs (HERO_MODEL, default llama-3.3-70b) of the
hero stages (HERO_STAGES, default hero_smoke,hero_solo,hero_solo_key) in the four judged gap environments (every judged
environment, full_runs_table.JUDGED, when HERO_STAGES includes hero_solo_all or HERO_ALL_JUDGED=1), read from
results/raw/<stage>/<env>/<Output path> (where modal_full.sh and hero_solo_foldin.sh put them); log in
results/logs/hero/grading/ (GRADE_LOG_DIR overrides). Same graders, same GRADE_CAP_USD and TOTAL_CEILING_USD, plus a
cost estimate per environment before it starts (ungraded runs' transcript + diff at CHARS_PER_TOKEN, Opus at
$JUDGE_IN_PER_M / $JUDGE_OUT_PER_M, JUDGE_OUT_TOKENS per run, times EST_MARGIN): an environment whose estimate would
take the spend past the cap or the ceiling is not started, and nothing after it. GRADE_DRY_RUN=1 prints the fleets
and the estimate and stops: no key read, no network.
  HERO_SOLO=1 GRADE_DRY_RUN=1 python3 gigi/3-forensics/scripts/grade_modal.py
HERO2_SWAP=1 (4 Nov 2026): the hero2 swap's replays (scripts/hero_swap.sh HERO2=1): every downloaded resume fleet under
results/raw/hero2_swap/<env>/<point>/<arm>/resume/<model>/<stamp>/ whose config names a judged environment (the four
judged gap environments: dashboard_costly_failure and optimize_rendering_engine -> dashboard_perf, grade_your_own,
optimize_rendering_engine_graded -> stripped_engine), as SWAP_ALL=1 does for results/raw/swap_all; log in
results/logs/hero2_swap/grading/ (GRADE_LOG_DIR overrides). Same GRADE_CAP_USD / TOTAL_CEILING_USD and the same
estimate-before-grading rule as HERO_SOLO. Before any replay is fetched (no results/raw/hero2_swap/) it says so and exits 0.
  HERO2_SWAP=1 GRADE_DRY_RUN=1 python3 gigi/3-forensics/scripts/grade_modal.py
"""
import collections, json, os, pathlib, re, subprocess, time, urllib.request
HERE = pathlib.Path(__file__).resolve().parents[1]; ENVS = HERE / "agent-interp-envs"
HERO = os.environ.get("HERO_SOLO") == "1"; DRY = os.environ.get("GRADE_DRY_RUN") == "1"
HERO2_SWAP = os.environ.get("HERO2_SWAP") == "1"
SWAP_NAME = os.environ.get("SWAP_HERO_NAME", "hero2_swap")   # 10 Nov 2026: another swap stage with the same layout, e.g. hero9_swap (all judged tasks)
EST = HERO or HERO2_SWAP   # estimate every environment first; stop before one whose estimate would cross the cap or the ceiling
LOG = pathlib.Path(os.environ["GRADE_LOG_DIR"]) if os.environ.get("GRADE_LOG_DIR") else \
    HERE / "results" / "logs" / (f"{SWAP_NAME}/grading" if HERO2_SWAP else "hero/grading" if HERO else "grading")
LOG.mkdir(parents=True, exist_ok=True)
KEY = None if DRY else \
    re.search(r"^OPENROUTER_API_KEY=(.+)$", pathlib.Path(os.environ["FORENSICS_ENV_FILE"]).read_text(), re.M).group(1).strip()
CAP = float(os.environ.get("GRADE_CAP_USD", 60))
# and a hard ceiling on the key's total since the project baseline (18 Oct 2026), across every pass and run
CEILING = float(os.environ.get("TOTAL_CEILING_USD", 1500)); BASELINE = float(os.environ.get("CEILING_BASELINE_USD", 131.477198523))
JUDGED = {"aj_data_anon", "dashboard_perf", "grade_your_own", "leaked_fix", "lucky_seed",
          "lucky_seed_readable_grader", "optimize_cache_c", "stripped_engine", "tables_coverage_tests"}

def say(msg):
    with open(LOG / "grade.txt", "a") as f: f.write(f"{time.strftime('%F %T')} {msg}\n")

def usage():
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"]

# the hero target's four judged gap environments (config name -> grader environment)
HERO_JUDGED = {"dashboard_costly_failure": "dashboard_perf", "grade_your_own": "grade_your_own",
               "optimize_rendering_engine": "dashboard_perf", "optimize_rendering_engine_graded": "stripped_engine"}
# HERO_MODEL (4 Nov 2026, second pass "hero2"): any model tag with configs/<env>__<tag>.yaml and hero launch logs
# (llama-3.3-70b, qwen3-next-80b, gemma-3-27b); the default stages of a non-Llama model are its hero2 stages
# (hero2_qwen + hero2_qwen_key, hero2_gemma + hero2_gemma_key; Llama's hero2 pass: HERO_STAGES=hero2_llama,hero2_llama_key).
HERO_MODEL = os.environ.get("HERO_MODEL") or "llama-3.3-70b"
_HERO_DEFAULT_STAGES = {"llama-3.3-70b": "hero_smoke,hero_solo,hero_solo_key", "qwen3-next-80b": "hero2_qwen,hero2_qwen_key",
                        "gemma-3-27b": "hero2_gemma,hero2_gemma_key", "qwen3.6-27b": "hero2_qwen36,hero2_qwen36_key",
                        "qwen3.6-27b-think": "hero2_qwen36_think"}
HERO_STAGES = os.environ.get("HERO_STAGES", _HERO_DEFAULT_STAGES.get(HERO_MODEL, "")).split(",")
if HERO and not any(HERO_STAGES):
    raise SystemExit(f"grade_modal.py HERO_SOLO=1: no default stages for HERO_MODEL={HERO_MODEL}; set HERO_STAGES")
if HERO and (any(s in ("hero_solo_all", "hero2_llama_all", "hero2_qwen_all", "hero2_gemma_all", "hero2_qwen36_all", "hero4_llama_all", "hero4_llama_fix", "hero6_example_all", "hero7_control_all", "hero8_gemma_all", "hero8_qwen36_all", "hero9_gemma_all", "hero9_qwen36_all") for s in HERO_STAGES) or os.environ.get("HERO_ALL_JUDGED") == "1"):
    # stage hero_solo_all (4 Nov 2026): every judged environment, the same config -> grader mapping as the Qwen rows of
    # results/full_runs.csv (full_runs_table.JUDGED: 11 environments, 9 graders). Same cap, ceiling and estimate.
    import sys
    sys.path.insert(0, str(HERE / "scripts"))
    import full_runs_table as _T
    HERO_JUDGED = dict(_T.JUDGED)


def estimate(dirs):
    """(runs, $) to grade the ungraded finished runs under these fleets: transcript (rollout.log) + diff (graders cap it
    at 60k chars) + 15k chars of rubric, at CHARS_PER_TOKEN; JUDGE_OUT_TOKENS out per run; times EST_MARGIN. Runs the
    grader will skip as unjudgeable are counted too, so it errs high."""
    cpt = float(os.environ.get("CHARS_PER_TOKEN", 3.0)); margin = float(os.environ.get("EST_MARGIN", 1.5))
    pin, pout = float(os.environ.get("JUDGE_IN_PER_M", 5)), float(os.environ.get("JUDGE_OUT_PER_M", 25))
    out_tok = float(os.environ.get("JUDGE_OUT_TOKENS", 4000))
    n, usd = 0, 0.0
    for d in dirs:
        for run in pathlib.Path(d).glob("run-*"):
            log = run / "rollout.log"
            if not (run / "final").is_dir() or (run / "final" / "grading.json").exists() or not log.exists():
                continue
            diff = run / "final" / "diff.patch"
            chars = log.stat().st_size + (min(diff.stat().st_size, 60_000) if diff.exists() else 0) + 15_000
            n += 1; usd += chars / cpt * pin / 1e6 + out_tok * pout / 1e6
    return n, usd * margin


fleets = collections.defaultdict(list)
if HERO:   # Llama's solo runs: every fleet named in the hero launch logs, grouped by its grader environment
    for stage in HERO_STAGES:
        if not (HERE / "results" / "logs" / stage).is_dir():
            say(f"stage {stage}: no launch logs yet (results/logs/{stage}/), nothing to grade there")
            continue
        for log in (HERE / "results" / "logs" / stage).glob(f"*__{HERO_MODEL}.log"):
            env = log.stem.split("__")[0]
            out = re.search(r"^Output: results/(\S+?)/?$", log.read_text(errors="replace"), re.M)
            d = HERE / "results" / "raw" / stage / env / out.group(1) if out else None
            if env in HERO_JUDGED and d and d.is_dir() and out.group(1).split("/")[0] == HERO_JUDGED[env]:
                fleets[HERO_JUDGED[env]].append(str(d))
elif os.environ.get("SWAP_ALL") or HERO2_SWAP:   # the swap replays: every downloaded resume fleet, grouped by its config's environment
    _swap_raw = HERE / "results" / "raw" / (SWAP_NAME if HERO2_SWAP else "swap_all")
    if HERO2_SWAP and not _swap_raw.is_dir():
        say(f"not yet: results/raw/{SWAP_NAME}/ does not exist (no swap replay fetched yet): nothing to grade")
        print(f"not yet: results/raw/{SWAP_NAME}/ does not exist (no swap replay fetched yet): nothing to grade"); raise SystemExit(0)
    if HERO2_SWAP and SWAP_NAME != "hero2_swap":   # another swap stage: every judged task, full_runs_table.JUDGED's config -> grader map
        import sys as _sys; _sys.path.insert(0, str(HERE / "scripts")); import full_runs_table as _T2; HERO_JUDGED = dict(_T2.JUDGED)
    for cfg in _swap_raw.glob("*/*/*/resume/*/*/config.yaml"):
        env = re.search(r"^environment: (\S+)", cfg.read_text(), re.M)
        # HERO2_SWAP: only the four judged gap environments of the plan (the <env> folder is the config name)
        if HERO2_SWAP and HERO_JUDGED.get(cfg.parents[5].name) != (env and env.group(1)):
            continue
        if env and env.group(1) in JUDGED and any(cfg.parent.glob("run-*/final")):
            fleets[env.group(1)].append(str(cfg.parent))
for stage in ([] if os.environ.get("SWAP_ALL") or HERO or HERO2_SWAP else os.environ.get("STAGES", "full_modal,full_modal_rest,full_modal_retry").split(",")):
    for log in (HERE / "results" / "logs" / stage).glob("*__*.log"):
        t = log.read_text(errors="replace")
        vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
        if vol and out:
            inner = out.group(1).split("/")[0]
            d = HERE / "results" / "raw" / "modal" / vol.group(1) / out.group(1)
            if inner in JUDGED and d.is_dir(): fleets[inner].append(str(d))
if DRY or EST:
    est = {env: estimate(dirs) for env, dirs in sorted(fleets.items())}
    lines = [f"estimate {env}: {n} ungraded runs in {len(fleets[env])} fleets, ~${usd:.2f}" for env, (n, usd) in est.items()]
    lines.append(f"estimate total ({f'hero solo, {HERO_MODEL}, stages ' + ','.join(HERO_STAGES) if HERO else f'swap {SWAP_NAME}, results/raw/{SWAP_NAME}' if HERO2_SWAP else 'default stages'}): {sum(n for n, _ in est.values())} ungraded runs "
                 f"in {sum(map(len, fleets.values()))} fleets, ~${sum(u for _, u in est.values()):.2f}")
    for line in lines:
        say(line)
    if DRY:
        print("\n".join(lines)); raise SystemExit(0)
    if HERO2_SWAP and not sum(n for n, _ in est.values()):   # nothing ungraded: no key read, no network
        print("\n".join(lines)); say(f"nothing ungraded in the {SWAP_NAME} swap: done"); raise SystemExit(0)
start = usage(); say(f"start: {sum(map(len, fleets.values()))} fleets in {len(fleets)} environments, cap ${CAP}, "
    f"judge {os.environ.get('JUDGE_MODEL') or 'default'}")
for env, dirs in sorted(fleets.items()):
    spent = usage() - start
    if spent >= CAP: say(f"CAP REACHED (${spent:.2f}): not grading {env} and later"); break
    if usage() - BASELINE >= CEILING: say(f"CEILING REACHED (${usage() - BASELINE:.2f} since baseline): not grading {env} and later"); break
    if EST:   # stop before an environment whose estimate would cross the cap or the ceiling
        if spent + est[env][1] > CAP: say(f"CAP WOULD BE CROSSED (spent ${spent:.2f} + estimate ${est[env][1]:.2f} > ${CAP}): not grading {env} and later"); break
        if usage() - BASELINE + est[env][1] > CEILING:
            say(f"CEILING WOULD BE CROSSED (${usage() - BASELINE:.2f} since baseline + estimate ${est[env][1]:.2f} > ${CEILING}): not grading {env} and later"); break
    try:   # 2 h per environment: a laptop that slept mid-call leaves a hung grader; graded runs are skipped on the next pass
        rc = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "python", f"environments/{env}/grade_rollouts.py", *dirs,
                             "--concurrency", "20", *(["--model", os.environ["JUDGE_MODEL"]] if os.environ.get("JUDGE_MODEL") else [])], cwd=ENVS, env={**os.environ, "JUDGE_VIA_OPENROUTER": "1", "OPENROUTER_API_KEY": KEY},
                            stdout=open(LOG / f"{env}.log", "a"), stderr=subprocess.STDOUT, timeout=7200).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout (2 h)"
    say(f"{env}: {len(dirs)} fleets, exit {rc}, spent so far ${usage() - start:.2f}")
say(f"done, spent ${usage() - start:.2f}")
