"""Lean fetch of the distillation re-runs (13 Nov 2026): per run only rollout.log, final/* files and the last step (no fs/).
run_modal.py's own download at app completion brings rollout.log only, and the measures need the last step's state.json.
Same pattern as hero9_swap_fetch_lean.py; copies only files missing locally, never overwrites; stops at < 1 GB free.
Run from agent-interp-envs/:  env -u VIRTUAL_ENV uv run python ../scripts/distill_fetch_lean.py distill_v1_base,distill_v1_c24,...
"""
import concurrent.futures as cf, pathlib, re, shutil, sys
import modal

H = pathlib.Path(__file__).resolve().parents[1]
RAW = H / "results/raw"; LOGS = H / "results/logs"
FILE = modal.volume.FileEntryType.FILE


def get(v, remote, dest):
    if dest.exists():
        return 0
    if shutil.disk_usage(H).free < 1e9:
        raise SystemExit("DISK LOW (< 1 GB free): stopping")
    data = b"".join(v.read_file(remote))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part"); tmp.write_bytes(data); tmp.rename(dest)
    return 1


def run_job(volname, r, rd):
    v = modal.Volume.from_name(volname); n = 0
    names = {e.path.rsplit("/", 1)[1]: e for e in v.listdir(r)}
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
            if e.type == FILE:
                n += get(v, e.path, rd / last / e.path.rsplit("/", 1)[1])
    return n


def main():
    jobs = []
    for s in sys.argv[1].split(","):
        for log in sorted((LOGS / s).glob("*__*.log")):
            t = log.read_text(errors="replace")
            vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
            if not vol or not out:
                print(f"no Volume/Output line: {s}/{log.name}"); continue
            env, fleet = log.stem.split("__")[0], out.group(1)
            v = modal.Volume.from_name(vol.group(1)); dest = RAW / s / env / fleet
            for e in v.listdir(fleet):
                nm = e.path.rsplit("/", 1)[1]
                if e.type == FILE and nm in ("config.yaml", "run_meta.json"):
                    get(v, e.path, dest / nm)
                elif e.type != FILE and re.fullmatch(r"run-\d+", nm):
                    jobs.append((vol.group(1), e.path, dest / nm))
    print(f"{len(jobs)} runs on the volumes")
    new = fail = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for f in cf.as_completed([ex.submit(run_job, *j) for j in jobs]):
            try:
                new += f.result()
            except Exception as e:   # noqa: BLE001
                fail += 1; print("FAILED", str(e)[:200])
    print(f"done: {new} new files, {fail} failed, {shutil.disk_usage(H).free / 1e9:.1f} GB free")


if __name__ == "__main__":
    main()
