"""Score the swap in every environment (scripts/swap_all.sh): for each restart point, how often Qwen (replayed)
and gemma (swapped in) show the environment's bad behaviour, with the same measures as results/full_runs.csv:
the environment's own grader verdict (final/grading.json) or its state.json measure (full_runs_table.MEASURES).
  fetch:  python3 swap_all_score.py fetch   copy each state-scored fleet (with steps) off its Modal volume
  score:  python3 swap_all_score.py         -> results/swap_all.csv (one row per point and model)
Lost replays (provider errors, crashes, gemma's context window exceeded) are counted separately, never as behaviour.
"""
import collections, csv, json, pathlib, re, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import full_runs_table as T

PLAN = json.load(open(HERE / "results" / "swap_all_plan.json"))
if (HERE / "results" / "swap_extra_plan.json").exists():   # the five environments added on 21 Oct (swap_extra_plan.py)
    PLAN += json.load(open(HERE / "results" / "swap_extra_plan.json"))
if (HERE / "results" / "swap_aird_plan.json").exists():    # ai_rd, after its fix (swap_aird_plan.py)
    PLAN += json.load(open(HERE / "results" / "swap_aird_plan.json"))
if (HERE / "results" / "swap_clean_plan.json").exists():   # the missing clean-run controls, 27 Oct (swap_clean_plan.py)
    PLAN += json.load(open(HERE / "results" / "swap_clean_plan.json"))
RAW = HERE / "results" / "raw" / "swap_all"
# Verdicts read by hand where the grader split identical behaviour (results/hand_verdicts.csv, append-only, with
# the grader's verdict and the reason next to each): applied over the grader's verdict for exactly those runs.
HAND = {r["run"]: r["verdict"] == "cheated" for r in csv.DictReader(open(HERE / "results" / "hand_verdicts.csv"))} \
    if (HERE / "results" / "hand_verdicts.csv").exists() else {}


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


def fetch_orphans():
    """Fleets whose launcher was stopped before downloading: the log names them, the volume has them."""
    for log in (HERE / "results/logs/swap_all").glob("*__*__*.log"):
        env, point, who = log.stem.split("__")[:3]
        t = log.read_text(errors="replace")
        vol = re.search(r"^Volume: (\S+)", t, re.M)
        for fleet in sorted(set(re.findall(r"(resume/[a-z0-9-]+/[0-9_-]+)/run-", t))):
            dest = RAW / env / point / who / fleet
            if vol and not any(dest.glob("run-*/rollout.log")):
                dest.parent.mkdir(parents=True, exist_ok=True)
                rc = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "modal", "volume", "get", "--force", vol.group(1),
                                     fleet, str(dest.parent)], cwd=HERE / "agent-interp-envs", capture_output=True, text=True)
                print(("orphan got " if rc.returncode == 0 else "orphan FAILED ") + f"{env} {point} {who}", flush=True)


def fetch():
    fetch_orphans()
    for p in PLAN:
        if p["env"] in T.JUDGED:
            continue
        vol = None
        for who in ("qwen", "gemma"):
            log = HERE / "results/logs/swap_all" / f"{p['env']}__{point_name(p)}__{who}.log"
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
        for who in ("qwen", "gemma"):
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
                    v = HAND.get(str(run.relative_to(HERE)), v)
                    if v is None:
                        lost[unscored_reason(run, p["env"])] += 1; continue
                    k += bool(v); n += 1
            rows.append({"env": p["env"], "point": point_name(p), "incident": p["incident"], "N": p["N"], "K": p["K"],
                         "fit_for_gemma": p.get("fit_for_gemma", False), "who": who, "k": k, "n": n,
                         "lost": "; ".join(f"{s}: {c}" for s, c in sorted(lost.items()))})
    out = HERE / "results" / "swap_all.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {out.relative_to(HERE)}")


if __name__ == "__main__":
    fetch() if sys.argv[1:] == ["fetch"] else score()
