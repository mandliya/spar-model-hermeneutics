"""Qwen3.6 27B with thinking on vs off, the four cheating tasks (7 Nov 2026; Yuqi's question: would thinking suit the
tasks better, and why does the model cheat). Reads the fold-in's csvs (stage hero2_qwen36_think vs hero2_qwen36 /
hero2_qwen36_all, both with the end-of-run check, fp8 DeepInfra) and prints
  1. the table for results/qwen36_thinking.md: cheats out of scored runs, thinking on vs off, with 95% Wilson intervals
     and the Newcombe gap;
  2. for every thinking-on run that did the bad act, the last REASONING blocks of its rollout.log (the thinking at the
     cheating moment), so the quotes in the write-up can be picked by hand (QUOTES_OUT=<file> writes them there).
  cd gigi/3-forensics/agent-interp-envs && uv run python ../scripts/qwen36_thinking_report.py
"""
import csv, math, os, pathlib, re, sys

H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
ENVS = ["grade_your_own", "currency_conversion", "lazy_investigation", "eval_tampering"]
LABEL = {"grade_your_own": "grade your own model", "currency_conversion": "currency conversion",
         "lazy_investigation": "lazy investigation", "eval_tampering": "eval tampering"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def rates(files):
    out = {}
    for name in files:
        f = H / "results" / name
        if f.exists():
            for r in csv.DictReader(open(f)):
                if r["environment"] in ENVS and r["environment"] not in out and int(r["n"] or 0) > 0:
                    out[r["environment"]] = (int(r["k"]), int(r["n"]), r["precision"])
    return out


on = rates(["hero2_full_runs__qwen3.6-27b-think.csv"])
off = rates(["hero2_full_runs__qwen3.6-27b.csv", "hero2_full_runs__qwen3.6-27b__other21.csv"])
print("| task | thinking on (fp8, DeepInfra) | thinking off (fp8, DeepInfra) | gap, on minus off |")
print("|---|---|---|---|")
for e in ENVS:
    cells = []
    for d in (on, off):
        if e in d:
            k, n, _ = d[e]; lo, hi = wilson(k, n)
            cells.append(f"{k} of {n} ({100 * k / n:.0f}%, 95% interval {100 * lo:.0f} to {100 * hi:.0f})")
        else:
            cells.append("no scored runs")
    if e in on and e in off:
        d_, lo, hi = newcombe(on[e][0], on[e][1], off[e][0], off[e][1])
        clear = lo > 0 or hi < 0
        gap = f"{100 * d_:+.0f} points ({100 * lo:+.0f} to {100 * hi:+.0f}){', clear' if clear else ', too close to call'}"
    else:
        gap = ""
    print(f"| {LABEL[e]} | {cells[0]} | {cells[1]} | {gap} |")

# the thinking at the cheating moment
per = H / "results" / "hero2_full_runs__qwen3.6-27b-think_per_run.csv"
out = []
if per.exists():
    for r in csv.DictReader(open(per)):
        if r["environment"] in ENVS and r["outcome"] == "scored" and r["bad"] == "1":
            log = H / r["run"] / "rollout.log"
            if not log.exists():
                continue
            t = log.read_text(errors="replace")
            blocks = re.split(r"\n[ \t]*REASONING[ \t]*\n=+\n", t)[1:]
            blocks = [b.split("\n" + "=" * 20)[0].strip() for b in blocks]
            out.append(f"\n##### {r['environment']} {r['run'].split('/')[-1]} ({len(blocks)} reasoning blocks)\n")
            for b in blocks[-2:]:
                out.append("-----\n" + b[-2500:] + "\n")
txt = "".join(out)
q = os.environ.get("QUOTES_OUT")
if q:
    pathlib.Path(q).write_text(txt); print(f"\nquotes written to {q}")
else:
    print(txt)
