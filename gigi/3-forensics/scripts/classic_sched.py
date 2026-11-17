"""Runs in flight for the classic round's paced launcher (scripts/classic_fleet3.sh; 17 Nov 2026).

  python3 classic_sched.py live STAGE[,STAGE...]     -> one number: runs in flight now across those stages
  python3 classic_sched.py apps STAGE[,STAGE...]     -> how many of those stages' launchers are still running

A launch log (results/logs/<stage>/<task>__<tag>.log, from modal_full.sh) counts while its launcher is alive: a process
whose command line ends in results/raw/<stage>/<task>. Its runs in flight = min(max concurrent, launched - runs already
reported) from the log's own lines ("Launching N experiments", "Max concurrent: M", one "  ✓/✗ <run> (sandbox ...|
launch failed ...)" line per finished run). A launcher that has not printed its counts yet counts as 1 run (every
paced chunk runs one at a time). Reads /proc, so it only works on the box (Linux).
"""
import pathlib
import re
import sys

H = pathlib.Path(__file__).resolve().parents[1]


def launchers():
    out = set()
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        m = re.search(r"--results-dir \S*results/raw/([^/\s]+)/(\w+)$", cmd)
        if m and "modal" in cmd:
            out.add((m.group(1), m.group(2)))
    return out


def in_flight(stages):
    alive = launchers(); n = apps = 0
    for s in stages:
        for log in sorted((H / "results/logs" / s).glob("*__*.log")):
            env = log.name.split("__")[0]
            if (s, env) not in alive:
                continue
            apps += 1
            t = log.read_text(errors="replace")
            N = re.search(r"Launching (\d+) experiments", t); M = re.search(r"Max concurrent: (\d+)", t)
            if not N:
                n += int(M.group(1)) if M else 1; continue
            N = int(N.group(1)); M = int(M.group(1)) if M else N
            done = len(re.findall(r"^  [✓✗] \S+/run-\d+ \((?:sandbox|launch failed)", t, re.M))
            n += max(0, min(M, N - done))
    return n, apps


if __name__ == "__main__":
    n, apps = in_flight(sys.argv[2].split(",") if len(sys.argv) > 2 else [])
    print(n if sys.argv[1] == "live" else apps)
