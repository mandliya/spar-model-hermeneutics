"""Score the swap in every environment (scripts/swap_all.sh): for each restart point, how often Qwen (replayed)
and gemma (swapped in) show the environment's bad behaviour, with the same measures as results/full_runs.csv:
the environment's own grader verdict (final/grading.json) or its state.json measure (full_runs_table.MEASURES).
  fetch:  python3 swap_all_score.py fetch   copy each state-scored fleet (with steps) off its Modal volume
  score:  python3 swap_all_score.py         -> results/swap_all.csv (one row per point and model)
Lost replays (provider errors, crashes, gemma's context window exceeded) are counted separately, never as behaviour.
Hero mode (4 Nov 2026; `--hero` on this script or on any script that imports it, or SWAP_HERO=1): the hero swap of
scripts/hero_swap.sh instead. Plan results/hero_swap_plan.json (SWAP_HERO_PLAN = another file name under results/),
replays in results/raw/hero_swap/, logs in results/logs/hero_swap/, arms llama / qwen36 / gemma (+ gemma4 / gptoss
when any of their folders exists), output results/hero_swap.csv. Without --hero nothing changes.
  python3 swap_all_score.py --hero fetch;  python3 swap_all_score.py --hero   -> results/hero_swap.csv
HERO2 mode (4 Nov 2026; `--hero2` or SWAP_HERO2=1): the same logic for the swap on Llama's second-pass runs with the
end-of-run confirmation (scripts/hero_swap.sh HERO2=1): plan results/hero2_swap_plan.json, replays in
results/raw/hero2_swap/, logs in results/logs/hero2_swap/, output results/hero2_swap.csv (a new file; nothing of the
first hero swap is read or written). Before the swap is launched (no results/raw/hero2_swap/ or no plan) every HERO2
script says so and exits 0. `--hero2 fetch` never uses `modal volume get --force` onto a run folder: each fleet goes to
an empty temp folder and only the files missing locally are moved in (hero_solo_foldin.sh's rule), so a local-only
final/grading.json can never be lost. SWAP_HERO2_RESULTS=<folder> (tests only) stands in for results/ in HERO2 mode.
  python3 swap_all_score.py --hero2 fetch;  python3 swap_all_score.py --hero2   -> results/hero2_swap.csv
"""
import collections, csv, json, os, pathlib, re, shutil, subprocess, sys, tempfile, time
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import full_runs_table as T

FLAGS = ("--hero", "--hero2")
HERO2 = "--hero2" in sys.argv[1:] or os.environ.get("SWAP_HERO2") == "1"
HERO = HERO2 or "--hero" in sys.argv[1:] or os.environ.get("SWAP_HERO") == "1"
HERO_ARMS, HERO_BACKUPS = ("llama", "qwen36", "gemma"), ("gemma4", "gptoss")
SWAPNAME = "hero2_swap" if HERO2 else "hero_swap"
# RES is results/ (HERO2 tests: SWAP_HERO2_RESULTS); run paths in the csvs are relative to ROOT (= the 3-forensics folder)
RES = pathlib.Path(os.environ["SWAP_HERO2_RESULTS"]).resolve() if HERO2 and os.environ.get("SWAP_HERO2_RESULTS") else HERE / "results"
ROOT = RES.parent
if HERO:
    _plan = RES / os.environ.get("SWAP_HERO_PLAN", f"{SWAPNAME}_plan.json")
    # Imported (full_runs_table.py --hero, hero_solo_foldin.sh) before the plan exists: status()/unscored_reason() need
    # no plan, so an empty one; run as a script, a missing plan is still an error (HERO2: a clean "not yet" exit, see main).
    PLAN = json.load(open(_plan)) if _plan.exists() or (__name__ == "__main__" and not HERO2) else []
    RAW = RES / "raw" / SWAPNAME
    LOGS = RES / "logs" / SWAPNAME
    WHO = HERO_ARMS + tuple(w for w in HERO_BACKUPS if any(RAW.glob(f"*/*/{w}")))
    OUT = RES / f"{SWAPNAME}.csv"
else:
    PLAN = json.load(open(HERE / "results" / "swap_all_plan.json"))
if not HERO and (HERE / "results" / "swap_extra_plan.json").exists():   # the five environments added on 21 Oct (swap_extra_plan.py)
    PLAN += json.load(open(HERE / "results" / "swap_extra_plan.json"))
if not HERO and (HERE / "results" / "swap_aird_plan.json").exists():    # ai_rd, after its fix (swap_aird_plan.py)
    PLAN += json.load(open(HERE / "results" / "swap_aird_plan.json"))
if not HERO and (HERE / "results" / "swap_clean_plan.json").exists():   # the missing clean-run controls, 27 Oct (swap_clean_plan.py)
    PLAN += json.load(open(HERE / "results" / "swap_clean_plan.json"))
if not HERO:
    RAW = HERE / "results" / "raw" / "swap_all"
    LOGS = HERE / "results" / "logs" / "swap_all"
    WHO = ("qwen", "gemma")
    OUT = HERE / "results" / "swap_all.csv"
# Verdicts read by hand where the grader split identical behaviour (results/hand_verdicts.csv, append-only, with
# the grader's verdict and the reason next to each): applied over the grader's verdict for exactly those runs.
HAND = {r["run"]: r["verdict"] == "cheated" for r in csv.DictReader(open(HERE / "results" / "hand_verdicts.csv"))} \
    if (HERE / "results" / "hand_verdicts.csv").exists() else {}


def not_ready():
    """HERO2 only: why there is nothing to do yet (the swap is not launched or nothing is fetched), else None."""
    if not HERO2:
        return None
    if not _plan.exists():
        return f"not yet: {_plan.relative_to(ROOT)} does not exist (hero_swap_when_ready.sh HERO2=1 writes it); nothing to do"
    if not RAW.is_dir():
        return f"not yet: {RAW.relative_to(ROOT)}/ does not exist (no hero2 swap replay fetched yet); nothing to do"
    return None


def point_name(p):
    run = pathlib.Path(p["run"]); return f"{run.parent.name}_{run.name}_K{p['K']}"


def fleets(p, who):
    return sorted((RAW / p["env"] / point_name(p) / who).glob("resume/*/*"))


def status(run):
    rl = run / "rollout.log"
    if not rl.exists():
        return "no log"
    tail = rl.read_text(errors="replace")[-5000:]
    if "maximum context length" in tail or re.search(r"'max_tokens' or 'max_completion_tokens' is too", tail) \
            or "CONTEXT LIMIT REACHED" in tail:
        return "context window exceeded"   # the history leaves no room for a reply (or the harness censored the episode)
    if "workspace budget" in tail:
        return "OpenRouter budget stop"    # 403 from the workspace's lifetime budget (21 Oct), mid-run
    return T.status(run)


def unscored_reason(run, env):
    """Why a finished replay has no score: the environment has no measure, no report or answer by the step cap,
    cut off before a (full) final state, or else not graded / no value for the measure."""
    if env not in T.MEASURES and env not in T.JUDGED:
        return "no measure"
    if re.search(r"No (report submitted|answer given) \(max steps reached\)", (run / "rollout.log").read_text(errors="replace")[-3000:]):
        return "step cap, no report"
    if env in T.MEASURES:   # scored from the last step's state.json; these environments write no final/
        return "no value for the measure" if any(run.glob("step-*/state.json")) else "state not fetched yet"
    if not (run / "final").exists():
        return "cut off, no final state"
    if not (run / "final" / "messages.json").exists():
        return "cut off, no full final state"
    return "not graded"


def safe_fetch(vol, rel, dest, timeout=3600):
    """`modal volume get <vol> <rel>` into an empty temp folder, then move in only the files missing under `dest`
    (never --force onto a run folder: that replaces it and a local-only final/grading.json is lost). Returns
    (rc, new, kept, differ, err): rc 0 on success; `differ` = files kept locally whose size differs from the volume's."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"{SWAPNAME}_fetch_"))
    new = kept = differ = 0; err = ""
    try:
        try:
            r = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "modal", "volume", "get", vol, rel, str(tmp)],
                               cwd=HERE / "agent-interp-envs", capture_output=True, text=True, timeout=timeout)
            rc, err = r.returncode, (r.stderr or r.stdout).strip()[-300:]
        except subprocess.TimeoutExpired:
            rc, err = -1, f"timeout ({timeout // 60} min)"
        if rc == 0:
            src = tmp / pathlib.PurePosixPath(rel).name
            src = src if src.is_dir() else tmp
            dest.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.rglob("*")):
                if f.is_dir() and not f.is_symlink():
                    continue
                d = dest / f.relative_to(src)
                if d.exists() or d.is_symlink():
                    kept += 1
                    differ += (not f.is_symlink() and not d.is_symlink() and d.is_file() and d.stat().st_size != f.stat().st_size)
                    continue
                d.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(f), str(d)); new += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rc, new, kept, differ, err


# a fleet path in a resume launch log: resume/<model slug>/<stamp>/run-N (slugs carry dots: qwen-qwen3.6-27b,
# meta-llama-llama-3.3-70b-instruct)
FLEET_RE = re.compile(r"(resume/[A-Za-z0-9._-]+/[0-9_-]+)/(run-\d+)")


def fetch_hero2():
    """HERO2: every fleet named in results/logs/hero2_swap/<env>__<point>__<who>[__topup_<stamp>].log, fetched (safe_fetch)
    when it is incomplete here: a run listed in the log without its rollout.log (the launcher was stopped before its
    download), or, in a state-scored environment (full_runs_table.MEASURES), without its step checkpoints (the
    launcher's analysis set has none). A fleet stamped .fetched_after_apps_stopped (hero2_swap_foldin.sh) is left alone."""
    n_ok = n_fail = 0
    for log in sorted(LOGS.glob("*__*__*.log")) if LOGS.is_dir() else []:
        env, point, who = log.stem.split("__")[:3]
        t = log.read_text(errors="replace")
        vol = re.search(r"^Volume: (\S+)", t, re.M)
        if not vol:
            continue
        by_fleet = collections.defaultdict(set)
        for fleet, run in FLEET_RE.findall(t):
            by_fleet[fleet].add(run)
        for fleet, names in sorted(by_fleet.items()):
            dest = RAW / env / point / who / fleet
            if (dest / ".fetched_after_apps_stopped").exists():
                continue
            need = [r for r in sorted(names) if not (dest / r / "rollout.log").exists()
                    or (env in T.MEASURES and not any((dest / r).glob("step-*/state.json")))]
            if not need:
                continue
            rc, new, kept, differ, err = safe_fetch(vol.group(1), fleet, dest)
            if rc == 0:
                n_ok += 1; print(f"got {env} {point} {who} {fleet}: {new} new files, {kept} already here ({differ} differ in size; local kept)", flush=True)
            else:
                n_fail += 1; print(f"FAILED {env} {point} {who} {fleet}: {err}", flush=True)
    print(f"fetch (hero2): {n_ok} fleets fetched, {n_fail} failed")
    return n_fail == 0


def fetch_orphans():
    """Fleets whose launcher was stopped before downloading: the log names them, the volume has them."""
    for log in LOGS.glob("*__*__*.log"):
        env, point, who = log.stem.split("__")[:3]
        t = log.read_text(errors="replace")
        vol = re.search(r"^Volume: (\S+)", t, re.M)
        for fleet in sorted(set(m[0] for m in FLEET_RE.findall(t))):
            dest = RAW / env / point / who / fleet
            if vol and not any(dest.glob("run-*/rollout.log")):
                dest.parent.mkdir(parents=True, exist_ok=True)
                rc = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "modal", "volume", "get", "--force", vol.group(1),
                                     fleet, str(dest.parent)], cwd=HERE / "agent-interp-envs", capture_output=True, text=True)
                print(("orphan got " if rc.returncode == 0 else "orphan FAILED ") + f"{env} {point} {who}", flush=True)


def fetch():
    if HERO2:   # never --force onto a run folder
        return fetch_hero2()
    fetch_orphans()
    for p in PLAN:
        if p["env"] in T.JUDGED:
            continue
        vol = None
        for who in WHO:
            log = LOGS / f"{p['env']}__{point_name(p)}__{who}.log"
            if log.exists():
                m = re.search(r"^Volume: (\S+)", log.read_text(errors="replace"), re.M); vol = m and m.group(1)
            for fleet in fleets(p, who):
                if any(fleet.glob("run-*/step-*")) or not vol:
                    continue
                rel = str(fleet.relative_to(RAW / p["env"] / point_name(p) / who))
                rc = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "modal", "volume", "get", "--force", vol, rel,
                                     str(fleet.parent)], cwd=HERE / "agent-interp-envs", capture_output=True, text=True)
                print(("got " if rc.returncode == 0 else "FAILED ") + f"{p['env']} {point_name(p)} {who}", flush=True)


def score():
    rows = []
    for p in PLAN:
        for who in WHO:
            k = n = 0; lost = collections.Counter()
            for fleet in fleets(p, who):
                for run in sorted(fleet.glob("run-*")):
                    st = status(run)
                    if st != "finished":
                        lost[st] += 1; continue
                    if p["env"] in T.MEASURES:
                        v = T.MEASURES[p["env"]][1](T.last_state(run), run)
                    elif p["env"] in T.JUDGED:
                        v = T.judged(None, run)
                    else:
                        v = None
                    v = HAND.get(str(run.relative_to(ROOT)), v)
                    if v is None:
                        lost[unscored_reason(run, p["env"])] += 1; continue
                    k += bool(v); n += 1
            rows.append({"env": p["env"], "point": point_name(p), "incident": p["incident"], "N": p["N"], "K": p["K"],
                         "fit_for_gemma": p.get("fit_for_gemma", False), "who": who, "k": k, "n": n,
                         "lost": "; ".join(f"{s}: {c}" for s, c in sorted(lost.items()))})
    out = OUT
    if not rows:
        print(f"no restart points in the plan: {out.relative_to(ROOT)} not written"); return
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    if not_ready():
        print(not_ready()); sys.exit(0)
    fetch() if [a for a in sys.argv[1:] if a not in FLAGS] == ["fetch"] else score()
