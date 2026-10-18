"""One row per smoke run on Modal: environment, model, steps used, and how the run ended.

Reads the runs straight off Modal's volumes (step count from a listing, rollout.log read per run;
no full download needed) and the launch logs in results/logs/modal_smoke/ (which run directory each
launch made). Run it from agent-interp-envs/ so the modal package is importable. The ending is the run's own
FINAL RESULTS block (or its last lines when it has none), cut to one line. Writes
results/smoke_modal.csv and prints it.

Run:  python gigi/3-forensics/scripts/smoke_summary.py
"""
import csv
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).resolve().parents[1]
RAW = HERE / "results" / "raw" / "modal"
LOGS = HERE / "results" / "logs" / "modal_smoke"


def ending(text):
    block = text.split("FINAL RESULTS")[-1] if "FINAL RESULTS" in text else "\n".join(text.splitlines()[-8:])
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not set(ln.strip()) <= set("=-")]
    return " | ".join(lines)[:300]


def main():
    runs = {}   # run path on the volume -> (volume, step dirs)
    for v in modal.Volume.objects.list():
        if not v.name.startswith("results-"):
            continue
        for e in v.listdir("/", recursive=True):
            m = re.match(r"(.*/run-\d+)/(step-\d+)?", e.path)
            if m:
                runs.setdefault(m.group(1), [v, set()])[1].add(m.group(2))
    rows = []
    for log in sorted(LOGS.glob("*__*.log")):
        name, model = log.stem.split("__")
        text = log.read_text(errors="replace")
        m = re.search(r"/mnt/results/(\S+/run-\d+)", text)
        run = runs.get(m.group(1)) if m else None
        steps = len(run[1] - {None}) if run else 0
        try:
            log_text = b"".join(run[0].read_file(f"{m.group(1)}/rollout.log")).decode(errors="replace") if run else ""
        except Exception:
            log_text = ""
        cap = re.search(r"max_steps (\d+)", (LOGS / "launch.txt").read_text().split(f"launched {log.stem} ")[-1][:40])
        status = ("done" if "Succeeded: 1/1" in text else "rate-limited" if "rate limit exceeded" in text
                  else "failed" if "Succeeded: 0" in text or "Error" in text else "running")
        rows.append({"env": name, "model": model, "status": status, "steps": steps, "cap": cap.group(1) if cap else "",
                     "ended": ending(log_text), "run_dir": m.group(1) if m else ""})
    out = HERE / "results" / "smoke_modal.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['env']:34s} {r['model']:15s} {r['status']:12s} {r['steps']:>3}/{r['cap']:<3} {r['ended'][:150]}")


if __name__ == "__main__":
    main()
