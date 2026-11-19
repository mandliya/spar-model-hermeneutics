"""How healthy the classic round's runs are, measured exactly as last week's distill_charts.health_row (19 Nov 2026), so
the two rounds compare: over finished runs in the same six tasks (tower_of_hanoi, eval_tampering, currency_conversion,
coding, coding_norules, lazy_investigation), how many hit the step limit, repeated the same call 3 times in a row, or
misused a tool (typed submit into the shell, or wrote its own made-up tool result). Run on the box from
agent-interp-envs/:  DISTILL_V=classic env -u VIRTUAL_ENV uv run python ../scripts/classic_health.py
writes results/classic_health.csv (who = qwen, base, c23, c46, c69, c92).
"""
import collections, csv, json, os, pathlib, re, sys
os.environ.setdefault("DISTILL_V", "classic")
H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import distill_eval as D  # noqa: E402

ENVS = ["tower_of_hanoi", "eval_tampering", "currency_conversion", "coding", "coding_norules", "lazy_investigation"]
STEP_LIMIT = re.compile(r"max steps reached|Reached max steps limit")
SHELL_SUBMIT = re.compile(r'Function: execute_command\nArguments: \{"command": "(?:[^"]*\|\s*)?\s*submit\b')
CALL = re.compile(r"Function: (\w+)\nArguments: (.*?)\n={20}", re.S)
FAKE_RESULT = re.compile(r"^\s*```tool_output", re.M)


def fake_result(run):
    steps = sorted(run.glob("step-*/messages.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not steps:
        return False
    msgs = json.loads(steps[-1].read_text())
    msgs = msgs if isinstance(msgs, list) else msgs.get("messages", [])
    return any(m.get("role") == "assistant" and FAKE_RESULT.search(str(m.get("content") or "")) for m in msgs)


def health(runs):
    c = collections.Counter()
    for run in runs:
        if D.run_status(run) != "finished":
            continue
        t = (run / "rollout.log").read_text(errors="replace"); c["runs"] += 1
        c["hit the step limit"] += bool(STEP_LIMIT.search(t))
        calls = CALL.findall(t)
        c["repeated the same call 3 times in a row"] += any(calls[i] == calls[i + 1] == calls[i + 2] for i in range(len(calls) - 2))
        c["misused a tool"] += bool(SHELL_SUBMIT.search(t)) or fake_result(run)
    return c


groups = {"qwen": [H / r["run"] for r in csv.DictReader(open(H / "results/classic_selected_per_run.csv")) if r["environment"] in ENVS]}
for m in ["base", "c23", "c46", "c69", "c92"]:
    groups[m] = [r for e in ENVS for r in (H / f"results/raw/distill_classic_{m}" / e).rglob("run-*") if r.is_dir()]
cols = ["hit the step limit", "repeated the same call 3 times in a row", "misused a tool"]
with open(H / "results/classic_health.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["who", "finished runs"] + cols)
    for who, runs in groups.items():
        c = health(runs); w.writerow([who, c["runs"]] + [c[k] for k in cols])
        print(who, c["runs"], [c[k] for k in cols])
