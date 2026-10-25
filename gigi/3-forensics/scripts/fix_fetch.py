"""Fetch the step checkpoints of the 25 Oct reruns (scripts/fix_rerun.sh, groups A and B) off their Modal volumes.

The launcher downloads only rollout.log; scoring needs each step's state.json. For every launch in
results/logs/swap_fix/commands.txt: the volume (the log's `Volume:` line) and the fleet (resume/<model>/<timestamp>)
are read from its log, and the fleet is copied into the launch's results dir with `modal volume get`. Never
overwrites: a fleet that already has step folders locally is skipped.
  cd gigi/3-forensics/agent-interp-envs && MODAL_PROFILE=spar uv run python ../scripts/fix_fetch.py
"""
import pathlib, re, subprocess
HERE = pathlib.Path(__file__).resolve().parents[1]

for line in (HERE / "results/logs/swap_fix/commands.txt").read_text().splitlines():
    log, out, _step, _n = line.split()
    t = pathlib.Path(log).read_text(errors="replace")
    vol = re.search(r"^Volume: (\S+)", t, re.M)
    fleets = sorted(set(re.findall(r"(resume/[a-z0-9-]+/[0-9_-]+)/run-", t)))
    if not vol or not fleets:
        print("NOT READY", pathlib.Path(log).name); continue
    for fleet in fleets:
        dest = pathlib.Path(out) / fleet
        if any(dest.glob("run-*/step-*")):
            print("have", pathlib.Path(log).name); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(["modal", "volume", "get", "--force", vol.group(1), fleet, str(dest.parent)],
                            capture_output=True, text=True)
        print(("got " if rc.returncode == 0 else "FAILED ") + pathlib.Path(log).name + ("" if rc.returncode == 0 else ": " + rc.stderr[-300:]),
              flush=True)
