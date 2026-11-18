"""Preliminary held-out results of the classic re-runs, while classic_rerun.sh is still waiting on slow apps (19 Nov 2026).
Run on the box from agent-interp-envs/:  DISTILL_V=classic env -u VIRTUAL_ENV uv run python ../scripts/classic_prelim.py
1. Fetches (lean: rollout.log, final/, last step; only missing files) the runs of every re-run app whose launcher has
   ENDED, so no half-written run is saved early (a run still going is never touched; the pipeline fetches it later).
2. Prints, per model and held-out task, the bad act k of n scored (Wilson 95%), lost runs by reason, garbled runs, and for
   eval_tampering how many finished runs read the note. Writes nothing else: the pipeline's scoring files are untouched.
"""
import collections, concurrent.futures as cf, os, pathlib, re, subprocess, sys
os.environ.setdefault("DISTILL_V", "classic")
H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import modal  # noqa: E402
import distill_eval as D  # noqa: E402
import distill_fetch_lean as F  # noqa: E402
from full_runs_table import saw_note  # noqa: E402

MODELS = ["base", "c23", "c46", "c69", "c92"]
alive = subprocess.run(["pgrep", "-af", "results/raw/distill_classic_"], capture_output=True, text=True).stdout
jobs, waiting = [], []
for m in MODELS:
    s = f"distill_classic_{m}"
    for log in sorted((H / "results/logs" / s).glob("*__*.log")):
        env = log.stem.split("__")[0]
        if re.search(rf"results/raw/{s}/{env}\s*$", alive, re.M):
            waiting.append(f"{m}/{env}"); continue
        t = log.read_text(errors="replace")
        vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
        if not vol or not out:
            continue
        v = modal.Volume.from_name(vol.group(1)); dest = H / "results/raw" / s / env / out.group(1)
        for e in v.listdir(out.group(1)):
            nm = e.path.rsplit("/", 1)[1]
            if e.type != F.FILE and re.fullmatch(r"run-\d+", nm):
                jobs.append((vol.group(1), e.path, dest / nm))
def safe(j):
    try:
        return F.run_job(*j)
    except Exception as e:   # noqa: BLE001  a failed fetch leaves that run as "no log", never scored
        print("FETCH FAILED", j[1], str(e)[:150]); return 0


with cf.ThreadPoolExecutor(16) as ex:
    new = sum(ex.map(safe, jobs))
print(f"fetched {new} new files for {len(jobs)} runs of ended apps; still running: {', '.join(waiting) or 'none'}\n")

print(f"{'task':22} {'model':5} {'bad act':>9} {'95% interval':>13}  lost / garbled / note")
for env in ["tower_of_hanoi", "lazy_investigation", "currency_conversion", "norvane", "eval_tampering"]:
    for m in MODELS:
        if f"{m}/{env}" in waiting:
            print(f"{env:22} {m:5} {'still running':>23}"); continue
        k = n = g = fin = note = 0; lost = collections.Counter()
        for run in sorted(r for r in (H / "results/raw" / f"distill_classic_{m}" / env).rglob("run-*") if r.is_dir()):
            st = D.run_status(run)
            if st != "finished":
                lost[st] += 1; continue
            fin += 1
            if env == "eval_tampering":
                note += bool(saw_note(run))
            try:
                v = D.value(env, run)
            except Exception as e:   # noqa: BLE001
                lost[f"unreadable {type(e).__name__}"] += 1; continue
            if v is None:
                lost["no value"] += 1; continue
            n += 1; k += bool(v); g += D.garbled(run)
        lo, hi = D.wilson(k, n)
        extra = f"; read the note {note} of {fin} finished" if env == "eval_tampering" else ""
        print(f"{env:22} {m:5} {k:4} of {n:<3} {100 * lo:5.0f}-{100 * hi:3.0f}%  {dict(lost) or '-'} / garbled {g}{extra}")
