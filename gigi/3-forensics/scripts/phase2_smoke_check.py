"""Smoke-test gate for a phase 2 teacher (19 Nov 2026; memory harness-smoke-per-env-type). Reads fetched runs.

  python3 scripts/phase2_smoke_check.py p2_ds_smoke --pin siliconflow [--reasoning-required]

Per run, from rollout.log and the last step's messages.json:
  - how it ended (distill_data.run_end): must be "finished" (or "context window full")
  - steps; the task's FINAL RESULTS block present
  - reasoning on every assistant turn (with --reasoning-required): turns with empty reasoning are counted
  - the usage ledger: [provider-usage] lines, calls without a cost, served_by hosts (must all match --pin)
  - trouble signs: "unknown function" replies, empty-reply retries, [provider retry] lines (and the highest attempt)
Exit 1 when any run fails a check, so a launcher script stops before the pilot or fleet.
"""
import argparse
import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from distill_data import run_end   # noqa: E402

USAGE = re.compile(r"^\[provider-usage\] .*?served_by=(\S+).*?cost=(\S+)", re.M)
RETRY = re.compile(r"\[provider retry\] attempt (\d+) failed: (\w+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages")
    ap.add_argument("--pin", required=True, help="the pinned provider slug, e.g. siliconflow")
    ap.add_argument("--reasoning-required", action="store_true")
    a = ap.parse_args()
    bad = 0; n = 0
    for st in a.stages.split(","):
        for run in sorted(p for p in (HERE / "results/raw" / st).rglob("run-*") if p.is_dir()):
            n += 1
            task = run.relative_to(HERE / "results/raw" / st).parts[0]
            log = (run / "rollout.log").read_text(errors="replace") if (run / "rollout.log").exists() else ""
            end = run_end(run)
            steps = sorted(run.glob("step-*/messages.json"), key=lambda p: int(p.parent.name.split("-")[1]))
            msgs = json.loads(steps[-1].read_text()) if steps else []
            msgs = msgs if isinstance(msgs, list) else msgs.get("messages", [])
            turns = [m for m in msgs if m.get("role") == "assistant"]
            no_reason = sum(1 for m in turns if not (m.get("reasoning") or m.get("reasoning_content")))
            usage = USAGE.findall(log)
            hosts = collections.Counter(h for h, _ in usage)
            no_cost = sum(1 for _, c in usage if c in ("None", ""))
            retries = RETRY.findall(log)
            problems = []
            if end not in ("finished", "context window full"):
                problems.append(f"ended: {end}")
            if a.reasoning_required and (not turns or no_reason):
                problems.append(f"{no_reason} of {len(turns)} turns without reasoning")
            if not usage:
                problems.append("no [provider-usage] lines")
            if no_cost:
                problems.append(f"{no_cost} calls without a cost")
            wrong = [h for h in hosts if a.pin.lower() not in h.lower()]
            if wrong:
                problems.append(f"served by {wrong} (pin {a.pin})")
            bad += bool(problems)
            print(f"{st}/{task}/{run.name}: end={end}, steps={len(steps)}, turns={len(turns)}, calls={len(usage)}, "
                  f"hosts={dict(hosts)}, unknown-function={log.lower().count('unknown function')}, "
                  f"empty-reply retries={log.count('empty reply')}, provider retries={len(retries)}"
                  f"{' (max attempt ' + str(max(int(x) for x, _ in retries)) + ')' if retries else ''}"
                  f"{' | PROBLEMS: ' + '; '.join(problems) if problems else ' | ok'}")
    print(f"{n - bad} of {n} runs pass")
    sys.exit(1 if bad or not n else 0)


if __name__ == "__main__":
    main()
