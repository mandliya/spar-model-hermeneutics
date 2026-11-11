"""Lean fetch of the hero9 symmetric swap (11 Nov 2026): only what scoring, grading and the transcripts need.

Why: the shared fetch (hero9_swap_score.py fetch -> swap_all_score.safe_fetch) runs `modal volume get` on a whole fleet,
which brings every step's messages.json (each one the whole conversation so far, so a 150-step Qwen run stores ~150
growing copies, ~75 MB a run) plus the workspace blobs. 658 fleets of that do not fit on the laptop. Every verdict, grader
and transcript needs only, per run:
  rollout.log                 status, ending, grader transcript
  final/*  (files only)       grade_rollouts.py: final/messages.json, final/diff.patch; grading.json lands here
  step-<last>/* (no fs/)      full_runs_table.last_state (state.json); note_read / commands (messages.json = whole history)
plus each fleet's config.yaml and run_meta.json. (collect.coding scans every step only to place the first cheat step,
which the swap does not use: hero9_swap_score keeps verdict(...)[0].)

Run from agent-interp-envs/ (needs the Modal SDK):   env -u VIRTUAL_ENV uv run python ../scripts/hero9_swap_fetch_lean.py [state|messages]
  state     (pass 1) everything above except the last step's messages.json
  messages  (pass 2) also the last step's messages.json
Fleets whose Modal app (the ap- id in its launch log) is still running are skipped, so no rollout.log is frozen mid-run;
rerun after they stop. Copies only files missing locally, never overwrites. Stops cleanly when free disk < 1 GB.
Log: results/logs/hero9_swap/fetch_lean.txt; heartbeat file fetch_lean.beat (touched on every file) for the supervisor.
"""
import concurrent.futures as cf, json, os, pathlib, re, shutil, subprocess, sys, threading, time
import modal

H = pathlib.Path(__file__).resolve().parents[1]
RAW = H / "results/raw/hero9_swap"; L = H / "results/logs/hero9_swap"
LOG = L / "fetch_lean.txt"; BEAT = L / "fetch_lean.beat"
FLEET_RE = re.compile(r"(resume/[A-Za-z0-9._-]+/[0-9_-]+)/(run-\d+)")
MIN_FREE = 1.0e9
PASS = sys.argv[1] if len(sys.argv) > 1 else "state"
FILE = modal.volume.FileEntryType.FILE
_lock = threading.Lock(); _stop = threading.Event(); _local = threading.local()


def say(msg):
    line = f"{time.strftime('%F %T')} [{PASS}] {msg}"
    with _lock:
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")


def vol(name):
    cache = getattr(_local, "vols", None)
    if cache is None:
        cache = _local.vols = {}
    if name not in cache:
        cache[name] = modal.Volume.from_name(name)
    return cache[name]


def get(v, remote, dest):
    """Copy one file if missing locally. Returns 1 if written."""
    if dest.exists() or _stop.is_set():
        return 0
    if shutil.disk_usage(H).free < MIN_FREE:
        _stop.set(); say("DISK LOW (< 1 GB free): stopping; free space and rerun"); return 0
    data = b"".join(v.read_file(remote))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part"); tmp.write_bytes(data); tmp.rename(dest)
    BEAT.touch(); return 1


def fleet_job(volname, fleet, dest):
    v = vol(volname); n = 0
    top = v.listdir(fleet)
    for e in top:
        name = e.path.rsplit("/", 1)[1]
        if e.type == FILE and name in ("config.yaml", "run_meta.json"):
            n += get(v, e.path, dest / name)
    for r in sorted(e.path for e in top if e.type != FILE and re.search(r"/run-\d+$", e.path)):
        if _stop.is_set():
            break
        rd = dest / r.rsplit("/", 1)[1]
        ents = v.listdir(r)
        names = {e.path.rsplit("/", 1)[1]: e for e in ents}
        if "rollout.log" in names:
            n += get(v, names["rollout.log"].path, rd / "rollout.log")
        if "final" in names:
            for e in v.listdir(f"{r}/final"):
                if e.type == FILE:
                    n += get(v, e.path, rd / "final" / e.path.rsplit("/", 1)[1])
        steps = [int(k.split("-")[1]) for k in names if re.fullmatch(r"step-\d+", k)]
        if steps:
            last = f"step-{max(steps)}"
            for e in v.listdir(f"{r}/{last}"):
                nm = e.path.rsplit("/", 1)[1]
                if e.type == FILE and (nm != "messages.json" or PASS == "messages"):
                    n += get(v, e.path, rd / last / nm)
        BEAT.touch()
    return n


def running_apps():
    out = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=180).stdout
    return {a.get("app_id") or a.get("App ID") for a in json.loads(out)
            if "stopped" not in str(a.get("state", a.get("State", ""))).lower()}


def main():
    live = running_apps()
    jobs, skipped = [], 0
    for log in sorted(L.glob("*__*__*.log")):
        env, point, who = log.stem.split("__")[:3]
        t = log.read_text(errors="replace")
        m = re.search(r"^Volume: (\S+)", t, re.M)
        if not m:
            continue
        if set(re.findall(r"ap-[A-Za-z0-9]+", t)) & live:
            skipped += 1; continue
        for fleet in sorted({f for f, _ in FLEET_RE.findall(t)}):
            jobs.append((m.group(1), fleet, RAW / env / point / who / fleet))
    jobs = jobs[:int(os.environ["LIMIT"])] if os.environ.get("LIMIT") else jobs   # smoke tests
    say(f"start: {len(jobs)} fleets ({skipped} launch logs skipped: app still running)")
    done = fail = new = 0
    with cf.ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(fleet_job, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                new += f.result(); done += 1
            except Exception as e:
                fail += 1; say(f"FAILED {futs[f][1]}: {str(e)[:200]}")
            if (done + fail) % 50 == 0:
                say(f"{done + fail} of {len(jobs)} fleets, {new} new files, {fail} failed, "
                    f"{shutil.disk_usage(H).free / 1e9:.1f} GB free")
    say(f"PASS DONE: {done} fleets, {new} new files, {fail} failed, {skipped} logs skipped (running)"
        + ("; stopped early: disk" if _stop.is_set() else ""))
    sys.exit(1 if (fail or _stop.is_set()) else 0)


if __name__ == "__main__":
    main()
