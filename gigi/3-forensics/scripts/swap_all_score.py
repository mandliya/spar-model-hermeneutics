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
RAW = HERE / "results" / "raw" / "swap_all"


def point_name(p):
    run = pathlib.Path(p["run"]); return f"{run.parent.name}_{run.name}_K{p['K']}"


def fleets(p, who):
    return sorted((RAW / p["env"] / point_name(p) / who).glob("resume/*/*"))


def status(run):
    rl = run / "rollout.log"
    if not rl.exists():
        return "no log"
    tail = rl.read_text(errors="replace")[-5000:]
    if "maximum context length" in tail:
        return "context window exceeded"
    return T.status(run)


def fetch():
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
                    if v is None:
                        lost["unscored"] += 1; continue
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
