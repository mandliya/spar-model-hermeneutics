"""Fetch only what scoring needs for every swap replay fleet named in the launch logs: per fleet the
config.yaml, and per run the rollout.log, final/* and the last step's state.json (not every step snapshot).
Fleets are named by the "resume/<model>/<timestamp>/run-N" lines of results/logs/swap_all/*.log.
  cd agent-interp-envs && MODAL_PROFILE=... uv run python ../scripts/swap_fetch_lean.py
"""
import pathlib, re, sys
import modal
HERE = pathlib.Path(__file__).resolve().parents[1]; RAW = HERE / "results" / "raw" / "swap_all"
JUDGED_CONFIGS = {"copying_of_answers", "dashboard_costly_failure", "grade_your_own", "leaked_tests", "lucky_seed",
                  "lucky_seed_readable_grader", "optimize_lru_cache", "optimize_rendering_engine",
                  "optimize_rendering_engine_graded", "leaked_scorer", "reference_extraction"}   # graded: no state needed


def get(vol, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as f:
        for chunk in vol.read_file(remote):
            f.write(chunk)


def fetch_fleet(job):
    env, point, who, volname, fleet = job
    vol = modal.Volume.from_name(volname, version=2)
    dest = RAW / env / point / who / fleet
    runs = [e.path for e in vol.listdir(fleet) if re.search(r"/run-\d+$", e.path)]

    def complete(r):   # what scoring needs is local: the rollout.log, and for state-scored envs a state.json
        local = dest / r.rsplit("/", 1)[1]
        return (local / "rollout.log").exists() and (env in JUDGED_CONFIGS or any(local.glob("step-*/state.json")))
    if not runs or all(complete(r) for r in runs):
        return None
    try:
        get(vol, f"{fleet}/config.yaml", dest / "config.yaml")
    except Exception:
        pass
    for r in runs:
        name = r.rsplit("/", 1)[1]; local = dest / name
        names = [e.path.rsplit("/", 1)[1] for e in vol.listdir(r)]
        if "rollout.log" in names:
            get(vol, f"{r}/rollout.log", local / "rollout.log")
        if "final" in names:
            for e in vol.listdir(f"{r}/final"):
                get(vol, e.path, local / "final" / e.path.rsplit("/", 1)[1])
        steps = sorted((int(n.split("-")[1]) for n in names if re.fullmatch(r"step-\d+", n)), reverse=True)
        for k in steps[:2]:   # the last step, or the one before if the last was cut off
            try:
                get(vol, f"{r}/step-{k}/state.json", local / f"step-{k}" / "state.json"); break
            except Exception:
                continue
    return f"got {env} {point} {who} ({len(runs)} runs)"


jobs = []
for log in sorted((HERE / "results/logs/swap_all").glob("*__*__*.log")):
    env, point, who = log.stem.split("__")[:3]
    t = log.read_text(errors="replace")
    m = re.search(r"^Volume: (\S+)", t, re.M)
    if m:
        jobs += [(env, point, who, m.group(1), f) for f in sorted(set(re.findall(r"(resume/[a-z0-9-]+/[0-9_-]+)/run-", t)))]
from concurrent.futures import ThreadPoolExecutor
done = 0
with ThreadPoolExecutor(max_workers=8) as pool:   # 8 fleets at a time
    for msg in pool.map(lambda j: (lambda: fetch_fleet(j))() if True else None, jobs):
        if msg:
            done += 1; print(msg, flush=True)
print(f"{done} fleets fetched")
