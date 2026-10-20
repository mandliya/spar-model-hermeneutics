"""Download only the fleets a stage launched (not whole volumes): for each results/logs/<stage>/*.log, its
"Volume:" and "Output:" lines name the fleet; `modal volume get` copies that folder to
results/raw/modal/<volume>/<fleet>/. Skips fleets already present. Much faster than modal_sync.py when the
volumes also hold earlier passes.   MODAL_PROFILE=... python3 gigi/3-forensics/scripts/fetch_fleets.py <stage>
"""
import pathlib, re, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
for log in sorted((HERE / "results" / "logs" / sys.argv[1]).glob("*__*.log")):
    t = log.read_text(errors="replace")
    vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
    if not (vol and out):
        print("no fleet in", log.name); continue
    dest = HERE / "results" / "raw" / "modal" / vol.group(1) / out.group(1)
    if dest.exists() and any(dest.glob("run-*")):
        print("have", out.group(1)); continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "modal", "volume", "get", vol.group(1), out.group(1),
                         str(dest.parent)], cwd=HERE / "agent-interp-envs", capture_output=True, text=True)
    print(("got " if rc.returncode == 0 else f"FAILED ({rc.stderr.strip()[-120:]}) ") + out.group(1), flush=True)
