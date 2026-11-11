"""Hand the hero9 symmetric swap over from the laptop to a cloud session (11 Nov 2026: the laptop kept sleeping).

Run on the laptop AFTER the laptop launcher (hero9_swap_paced.sh) is stopped, so the list of what is left is final:
    python3 scripts/hero9_swap_handoff.py
Writes handoff/hero9_swap/ (committed to git; results/raw and *.log are gitignored, so this is how the cloud gets them):
  launch_status.tsv       one row per resume: launch log, status, Modal app, checkpoint, results dir (paths as <repo>/...)
  log_index.tsv           per launch log: the `Volume:` line and every fleet/run name in it, which is all the fold-in's
                          fetch reads from a log (hero9_swap_score.py fetch); the cloud rebuilds stub logs from it
  commands_remaining.txt  the launch lines of every resume not launched yet, paths as <repo>/...
  checkpoints.tar.gz      the step-K folders those resumes start from, plus each fleet's config.yaml and run_meta.json
  harness_overlay.tar.gz  our local changes to agent-interp-envs (modified + new files) on top of BASE_COMMIT
  BASE_COMMIT             the upstream agent-interp-envs commit the overlay applies to
Every text file is checked for OpenRouter / HF key shapes before it is written; one hit = abort, nothing written.
"""
import csv, io, pathlib, re, shlex, subprocess, sys, tarfile, collections

H = pathlib.Path(__file__).resolve().parents[1]
ROOT = str(H.parents[1])
L = H / "results/logs/hero9_swap"
OUT = H / "handoff/hero9_swap"
CMDS = sorted(L.glob("commands_*.txt"))[-1]
FLEET_RE = re.compile(r"(resume/[A-Za-z0-9._-]+/[0-9_-]+)/(run-\d+)")
KEY_RE = re.compile(rb"sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}")
HARNESS = H / "agent-interp-envs"


def rel(p):
    return str(p).replace(ROOT, "<repo>")


def check(name, data: bytes):
    if KEY_RE.search(data):
        sys.exit(f"KEY SHAPE IN {name}: aborting, nothing written")
    return data


def overlay():
    """BASE_COMMIT + harness_overlay.tar.gz only: lets the cloud install the harness before the final handoff."""
    base = subprocess.run(["git", "-C", str(HARNESS), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    changed = [l[3:] for l in subprocess.run(["git", "-C", str(HARNESS), "status", "--porcelain"], capture_output=True,
                                              text=True).stdout.splitlines() if l.strip()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for c in sorted(changed):
            p = HARNESS / c
            if not p.is_file():
                continue
            data = check(c, p.read_bytes()); ti = tarfile.TarInfo(c); ti.size = len(data); ti.mtime = int(p.stat().st_mtime)
            tf.addfile(ti, io.BytesIO(data))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "harness_overlay.tar.gz").write_bytes(buf.getvalue()); (OUT / "BASE_COMMIT").write_text(base + "\n")
    print(f"overlay: {len(changed)} files on {base[:7]} ({len(buf.getvalue()) // 1024} KB)")


def main():
    if "--overlay-only" in sys.argv:
        return overlay()
    if subprocess.run(["pgrep", "-f", "hero9_swap_paced.sh"], capture_output=True).stdout.strip():
        sys.exit("the laptop launcher (hero9_swap_paced.sh) is still running: stop it first, or the list is not final")
    status_rows, index_rows, remaining, need = [], [], [], set()
    for line in CMDS.read_text().splitlines():
        if not line.strip():
            continue
        a = shlex.split(line); log = pathlib.Path(a[0])
        step = a[a.index("--step-dir") + 1]; rdir = a[a.index("--results-dir") + 1]
        st, app, vol, fleets = "not launched", "", "", []
        if log.exists():
            t = log.read_text(errors="replace")
            if re.search(r"Launching|Resuming|Checkpoint:", t) and "rate limit exceeded" not in t:
                st = "completed" if "App completed" in t else ("client lost, app ran on" if "Connection lost" in t else "running")
            m = re.search(r"modal\.com/apps/[^/\s]+/[^/\s]+/(ap-[A-Za-z0-9]+)", t); app = m.group(1) if m else ""
            m = re.search(r"^Volume: (\S+)", t, re.M); vol = m.group(1) if m else ""
            fleets = sorted({f"{f}/{r}" for f, r in FLEET_RE.findall(t)})
        if st == "not launched":
            remaining.append(" ".join(shlex.quote(rel(x)) for x in a)); need.add(step)
        else:
            index_rows.append([log.name, vol, " ".join(fleets)])
        status_rows.append([log.name, st, app, rel(step), rel(rdir)])
    counts = collections.Counter(r[1] for r in status_rows)

    files = set()
    for s in need:
        p = pathlib.Path(s)
        files |= {f for f in p.rglob("*") if f.is_file()}
        files |= {p.parent.parent / n for n in ("config.yaml", "run_meta.json") if (p.parent.parent / n).exists()}

    base = subprocess.run(["git", "-C", str(HARNESS), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    changed = [l[3:] for l in subprocess.run(["git", "-C", str(HARNESS), "status", "--porcelain"], capture_output=True,
                                              text=True).stdout.splitlines() if l.strip()]

    def tarball(paths, root):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for p in sorted(paths):
                p = pathlib.Path(p); data = check(str(p), p.read_bytes())
                ti = tarfile.TarInfo(str(p.relative_to(root))); ti.size = len(data); ti.mtime = int(p.stat().st_mtime)
                tf.addfile(ti, io.BytesIO(data))
        return buf.getvalue()

    ck = tarball(files, H)
    ov = tarball([HARNESS / c for c in changed if (HARNESS / c).is_file()], HARNESS)

    def tsv(rows, header):
        s = io.StringIO(); w = csv.writer(s, delimiter="\t", lineterminator="\n"); w.writerow(header); w.writerows(rows)
        return check("tsv", s.getvalue().encode())

    st_b = tsv(status_rows, ["log", "status", "modal_app", "step_dir", "results_dir"])
    ix_b = tsv(index_rows, ["log", "volume", "fleet_runs"])
    rm_b = check("commands_remaining.txt", ("\n".join(remaining) + "\n").encode())
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [("launch_status.tsv", st_b), ("log_index.tsv", ix_b), ("commands_remaining.txt", rm_b),
                       ("checkpoints.tar.gz", ck), ("harness_overlay.tar.gz", ov), ("BASE_COMMIT", (base + "\n").encode())]:
        (OUT / name).write_bytes(data)
    print(f"{dict(counts)}; {len(remaining)} resumes left from {len(need)} checkpoints ({len(files)} files, "
          f"{len(ck) // 1024} KB); harness overlay {len(changed)} files on {base[:7]} ({len(ov) // 1024} KB) -> {rel(OUT)}")


if __name__ == "__main__":
    main()
