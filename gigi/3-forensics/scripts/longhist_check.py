"""John's long-history test (22 Oct call; launched 26 Oct by scripts/parallel_checks.sh): does gemma still operate
after taking over a long Qwen run at turn 100 (checkpoint 99), for up to 100 more turns? Free: reads saved logs.

Pass, fixed before the runs finished: gemma reaches a normal end (a final answer, submit/end_session, or the step
cap) in >= 90% of its replays not lost to the provider, and its share of unreadable turns (call-like text our
parser could not read, or a typed-out fake tool result) is no higher than in its own full runs (2.4%, 26 Oct) by
more than 2 points. Qwen's 3 replays per run are the baseline for how far the task goes.
Writes results/longhist_check.csv (one row per replay) and prints one line per environment and model.
  cd agent-interp-envs && uv run python ../scripts/longhist_check.py
"""
import collections, csv, pathlib, re, statistics
HERE = pathlib.Path(__file__).resolve().parents[1]
RAW = HERE / "results/raw/longhist"
K = 99
STEP = re.compile(r"^\s+STEP (\d+)\s*$", re.M)
CALL_TEXT = re.compile(r"```(?:tool_code|python|tool_call)?\s*\n?\s*\[?\w+\(", re.S)


def turns(log):
    t = log.read_text(errors="replace")
    marks = [(int(m.group(1)), m.start()) for m in STEP.finditer(t)]
    out = []
    for i, (n, s) in enumerate(marks):
        if n <= K:
            continue
        block = t[s:marks[i + 1][1] if i + 1 < len(marks) else len(t)]
        r = block.find("RESPONSE"); resp = ""
        if r >= 0:
            a = block.find("\n", block.find("=====", r)) + 1; b = block.find("\n=====", a); resp = block[a:b if b > 0 else None].strip()
        if re.search(r"Function: \w+", block):
            kind = "call_error" if re.search(r"Invalid arguments|Unknown function|not a valid tool", block) else "call"
        elif not resp or resp == "(No response)":
            kind = "empty"
        elif "```tool_output" in resp:
            kind = "fake_output"
        elif CALL_TEXT.search(resp) or "tool_code" in resp:
            kind = "unreadable"
        else:
            kind = "final"
        out.append((n, kind))
    return t, out


def ending(t, tt, cap=200):
    tail = t[-5000:]
    if re.search(r"maximum context length|max_completion_tokens' is too|CONTEXT LIMIT REACHED", tail):
        return "lost: context window"
    if re.search(r"raised (RateLimitError|OpenRouterEmptyResponseError)|Error code: 5\d\d", tail):
        return "lost: provider"
    if "Traceback" in tail[-1500:]:
        return "crash"
    if not tt:
        return "no turns"
    if tt[-1][0] + 1 >= cap:
        return "step cap (100 turns done)"
    return {"final": "final answer", "unreadable": "ended by our parser", "fake_output": "ended by our parser",
            "empty": "empty reply"}.get(tt[-1][1], "ended by the environment")


rows = []
for log in sorted(RAW.glob("*/*/resume/*/*/run-*/rollout.log")):
    env, who = log.parts[-7], log.parts[-6]
    t, tt = turns(log)
    k = collections.Counter(x for _, x in tt)
    rows.append({"env": env, "who": who, "run": str(log.parent.relative_to(HERE)), "turns": len(tt), "ending": ending(t, tt),
                 **{c: k[c] for c in ("call", "call_error", "unreadable", "fake_output", "final", "empty")}})
out = HERE / "results/longhist_check.csv"
if rows:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
NORMAL = ("final answer", "ended by the environment", "step cap (100 turns done)")
for who in ("gemma", "qwen"):
    sel = [r for r in rows if r["who"] == who]
    for env in sorted({r["env"] for r in sel}) + ["ALL"]:
        s = [r for r in sel if env in ("ALL", r["env"])]
        ok = [r for r in s if not r["ending"].startswith("lost")]
        normal = sum(r["ending"] in NORMAL for r in ok); nt = sum(r["turns"] for r in s)
        bad = sum(r["unreadable"] + r["fake_output"] for r in s); errs = sum(r["call_error"] for r in s)
        med = statistics.median([r["turns"] for r in s]) if s else 0
        print(f"{who:5s} {env:28s} replays {len(s):2d} | normal end {normal}/{len(ok)} | turns: median {med:5.1f}, "
              f"unreadable {bad}/{nt} ({100 * bad / max(nt, 1):.1f}%), error-answered calls {errs} | "
              + "; ".join(f"{e} {c}" for e, c in collections.Counter(r['ending'] for r in s).most_common()))
g = [r for r in rows if r["who"] == "gemma"]; ok = [r for r in g if not r["ending"].startswith("lost")]
if ok:
    rate = sum(r["ending"] in NORMAL for r in ok) / len(ok)
    unread = sum(r["unreadable"] + r["fake_output"] for r in g) / max(sum(r["turns"] for r in g), 1)
    print(f"\nPASS RULE: normal end {rate:.0%} (>= 90%: {'yes' if rate >= 0.9 else 'NO'}); unreadable {unread:.1%} "
          f"(<= 4.4%: {'yes' if unread <= 0.044 else 'NO'})")
