"""Fetch fleets named in launch logs off their Modal volumes (26 Oct fold-in). Args: <log dir> <results root>; each log
ai_rd__<point>__<who>.log (or any *.log with a matching --results-dir layout) names its volume (`Volume:`) and its
fleets (resume/<model>/<ts>); they land in <results root>/<point>/<who>/<fleet>. Skips fleets that already have steps.
  cd agent-interp-envs && MODAL_PROFILE=spar uv run python ../scripts/fetch_from_logs.py ../results/logs/swap_aird_v2 ../results/raw/swap_aird_v2
"""
import pathlib, re, subprocess, sys
logs, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for log in sorted(logs.glob("*__*__*.log")):
    _, point, who = log.stem.split("__")[:3]
    t = log.read_text(errors="replace")
    vol = re.search(r"^Volume: (\S+)", t, re.M)
    fleets = sorted(set(re.findall(r"(resume/[a-z0-9-]+/[0-9_-]+)/run-", t)))
    if not vol or not fleets:
        print("NOT READY", log.name); continue
    for fleet in fleets:
        dest = root / point / who / fleet
        if any(dest.glob("run-*/step-*")):
            print("have", log.name); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(["modal", "volume", "get", "--force", vol.group(1), fleet, str(dest.parent)], capture_output=True, text=True)
        print(("got " if rc.returncode == 0 else "FAILED ") + log.name + ("" if rc.returncode == 0 else ": " + rc.stderr[-200:]), flush=True)
