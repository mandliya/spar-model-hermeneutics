"""classic_launched.py STAGE/TASK ...: run dirs on the Modal volume for that launch log's fleet (= runs launched so far), and per run
whether it has final/, its first/last write (UTC) and step count. Read-only."""
import pathlib, re, sys, datetime as dt
import modal
L = pathlib.Path("/workspace/forensics/gigi/3-forensics/results/logs")
for spec in sys.argv[1:]:
    st, env = spec.split("/")
    t = next((L / st).glob(f"{env}__*.log")).read_text(errors="replace")
    vol = re.search(r"^Volume: (\S+)", t, re.M).group(1); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M).group(1)
    v = modal.Volume.from_name(vol)
    runs = sorted((e.path for e in v.listdir(out) if re.search(r"run-\d+$", e.path)), key=lambda p: int(p.rsplit("-", 1)[1]))
    print(f"{spec} launched {len(runs)}")
    if "-v" in sys.argv[0:1] or env == "tower_of_hanoi":
        for r in runs:
            es = list(v.listdir(r)); ts = [e.mtime for e in es]
            f = lambda x: dt.datetime.fromtimestamp(x, dt.timezone.utc).strftime("%H:%M")
            print(f"  {r.rsplit('/',1)[1]}: {f(min(ts))}-{f(max(ts))} final={'final' in {e.path.rsplit('/',1)[1] for e in es}} steps={sum(1 for e in es if '/step-' in e.path)}")
