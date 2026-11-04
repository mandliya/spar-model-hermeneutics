"""Does gemma still work when it takes over Qwen's run? (validation pass, 23 Oct 2026; free: reads saved logs only)

The swap hands gemma Qwen's history, reformatted to gemma's text-mode tool format (every past call as a
```tool_code``` block, every result as ```tool_output```; providers/openrouter_provider.py). The risk raised on
the 22 Oct call: Qwen's history might still be out of distribution for gemma, so gemma stumbles (unreadable calls,
early stops, crashes), and a stumble would look like "gemma doesn't cheat". This script reads every replay's
rollout.log and classifies each turn the replaying model wrote after the takeover:
  call        a tool call the harness read and ran
  unreadable  no call read, but the reply holds call-like text (```tool_code```/```python``` with a call)
  fake_output no call read, and the reply types out a ```tool_output``` block itself
  final       a plain reply with no call (the model's final answer; ends the run)
  empty       an empty reply
  qwen_format a gemma reply that writes a call in Qwen's native format (<tool_call>, {"name": ...}) [flag]
Groups compared: gemma in Qwen's runs, Qwen in Qwen's runs (the baseline), gemma in its own runs (the pre-commit
control, 20 Oct) and gemma's own full runs (text mode, 20-21 Oct; every turn counts as "after takeover").

Pass criteria, fixed before looking at the numbers:
  1. first turn after the takeover is a call or a final answer in >= 95% of gemma-in-Qwen replays
  2. per-turn unreadable + fake_output rate of gemma in Qwen's runs at most 2 points above gemma's own full runs
  3. replays ended by our parser (last turn unreadable or fake_output) <= 5%, overall and in each gap environment
     (rendering engine, ai_rd, is_prime = coding/coding_norules, pre-commit hook)
  4. no qwen_format replies (gemma never sees Qwen's native format, so any would be a red flag)
Writes results/swap_validity.csv (one row per replay) and prints the summary.
  cd agent-interp-envs && uv run python ../scripts/swap_validity.py
Hero mode (4 Nov 2026): `swap_validity.py --hero` (or SWAP_HERO=1) classifies the hero swap's replays instead (arms
of swap_all_score.py's hero mode; groups "<arm> in Llama's runs"; no pre-commit study, no own-runs group) and writes
results/hero_swap_validity.csv. Without --hero nothing changes.
HERO2 mode (4 Nov 2026): `swap_validity.py --hero2` (or SWAP_HERO2=1): the hero2 swap (swap_all_score.py's HERO2 mode,
results/raw/hero2_swap/), same groups, writes results/hero2_swap_validity.csv. Before the swap is launched it says so and exits 0.
"""
import collections, csv, math, pathlib, re, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import full_runs_table as T
import swap_all_score as S

if S.not_ready():
    print(S.not_ready()); sys.exit(0)

RAW = HERE / "results" / "raw"
GAP_ENVS = {"optimize_rendering_engine": "rendering engine", "ai_rd": "ai_rd", "coding": "is_prime",
            "coding_norules": "is_prime", "precommit_hook": "pre-commit hook"}
STEP = re.compile(r"^\s+STEP (\d+)\s*$", re.M)
CALL_TEXT = re.compile(r"```(?:tool_code|python|tool_call)?\s*\n?\s*\[?\w+\(", re.S)
QWEN_FMT = re.compile(r"<tool_call>|\{\s*\"name\"\s*:\s*\"\w+\"\s*,\s*\"arguments\"")


def turns(log, after):
    """[(step, kind, text)] for every step > after."""
    t = log.read_text(errors="replace")
    marks = [(int(m.group(1)), m.start()) for m in STEP.finditer(t)]
    out = []
    for i, (n, s) in enumerate(marks):
        if n <= after:
            continue
        block = t[s:marks[i + 1][1] if i + 1 < len(marks) else len(t)]
        r = block.find("RESPONSE")
        resp = ""
        if r >= 0:
            a = block.find("\n", block.find("=====", r)) + 1
            b = block.find("\n=====", a)
            resp = block[a:b if b > 0 else None].strip()
        has_call = re.search(r"Function: \w+", block) is not None
        if has_call:
            kind = "call"
        elif not resp or resp == "(No response)":
            kind = "empty"
        elif "```tool_output" in resp:
            kind = "fake_output"
        elif CALL_TEXT.search(resp) or "tool_code" in resp:
            kind = "unreadable"
        else:
            kind = "final"
        out.append((n, kind, resp))
    return out


def max_steps(run):
    for c in (run / "config.yaml", run.parent / "config.yaml"):
        if c.exists():
            m = re.search(r"max_steps:\s*(\d+)", c.read_text())
            if m:
                return int(m.group(1))
    return None


HERO_GROUP = {"llama": "Llama in Llama's runs", "qwen36": "Qwen3.6 in Llama's runs", "gemma": "gemma in Llama's runs",
              "gemma4": "gemma 4 in Llama's runs", "gptoss": "GPT-OSS in Llama's runs"}


def replays():
    """(group, env, point, K, who, run) for every replay we ran."""
    if S.HERO:
        for p in S.PLAN:
            for who in S.WHO:
                for fleet in S.fleets(p, who):
                    for run in sorted(fleet.glob("run-*")):
                        yield (HERO_GROUP.get(who, f"{who} in Llama's runs"), p["env"], S.point_name(p), p["K"], p["incident"], run)
        return
    for p in S.PLAN:
        for who in ("qwen", "gemma"):
            for fleet in S.fleets(p, who):
                for run in sorted(fleet.glob("run-*")):
                    yield ("gemma in Qwen's runs" if who == "gemma" else "Qwen in Qwen's runs",
                           p["env"], S.point_name(p), p["K"], p["incident"], run)
    for d in sorted((RAW / "swap_text").glob("precommit_*/from_step*")):
        K = int(d.name.replace("from_step", "")); own = d.parent.name.startswith("precommit_g")
        for run in sorted(d.glob("gemma/resume/*/*/run-*")):
            yield ("gemma in its own runs" if own else "gemma in Qwen's runs", "precommit_hook",
                   d.parent.name + "_" + d.name, K, d.parent.name.startswith("precommit_q"), run)
    for env, model, run in T.modal_runs():
        cfg = run.parent / "config.yaml"
        if model == "gemma-3-27b" and cfg.exists() and "text_tools: true" in cfg.read_text():
            yield ("gemma's own full runs", env, run.parent.name, -1, False, run)


rows = []
for group, env, point, K, incident, run in replays():
    log = run / "rollout.log"
    st = S.status(run) if log.exists() else "no log"
    tt = turns(log, K) if log.exists() else []
    kinds = collections.Counter(k for _, k, _ in tt)
    cap = max_steps(run)
    last = tt[-1][1] if tt else ""
    if st != "finished":
        ending = f"lost: {st}"
    elif last in ("unreadable", "fake_output"):
        ending = "ended by our parser"
    elif last == "empty":
        ending = "empty reply"
    elif last == "final":
        ending = "final answer"
    elif cap and tt and tt[-1][0] + 1 >= cap:
        ending = "step cap"
    else:
        ending = "ended by the environment (submit / end_session / game over)"
    rows.append({"group": group, "env": env, "point": point, "K": K, "incident": incident,
                 "run": str(run.relative_to(S.ROOT)), "status": st, "turns": len(tt),
                 "first": tt[0][1] if tt else "", "call": kinds["call"], "unreadable": kinds["unreadable"],
                 "fake_output": kinds["fake_output"], "final": kinds["final"], "empty": kinds["empty"],
                 "qwen_format": sum(1 for _, k, r in tt if "gemma" in group and QWEN_FMT.search(r)),
                 "ending": ending})

out = S.RES / ("hero2_swap_validity.csv" if S.HERO2 else "hero_swap_validity.csv" if S.HERO else "swap_validity.csv")
if not rows:
    print(f"no replays found under {S.RAW.relative_to(S.ROOT)}/ for the plan's {len(S.PLAN)} points: {out.name} not written"); sys.exit(0)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def wilson(k, n):
    if not n:
        return (0, 0)
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% ({100 * lo:.1f}-{100 * hi:.1f})" if n else "-"


def summary(sel, label):
    ran = [r for r in sel if r["turns"]]
    first_ok = sum(r["first"] in ("call", "final") for r in ran)
    nt = sum(r["turns"] for r in ran); bad = sum(r["unreadable"] + r["fake_output"] for r in ran)
    fin = [r for r in sel if not r["ending"].startswith("lost")]
    parser = sum(r["ending"] == "ended by our parser" for r in fin)
    print(f"\n== {label}: {len(sel)} replays, {len(ran)} with turns after the takeover")
    print(f"   first turn a call or final answer : {pct(first_ok, len(ran))}")
    print(f"   turns unreadable or fake output   : {pct(bad, nt)}")
    print(f"   finished replays ended by parser  : {pct(parser, len(fin))}")
    print(f"   Qwen-format replies (gemma)       : {sum(r['qwen_format'] for r in sel)}")
    ends = collections.Counter(r["ending"] for r in sel)
    print("   endings:", "; ".join(f"{k} {v}" for k, v in ends.most_common()))


print(f"{len(rows)} replays -> {out.relative_to(S.ROOT)}")
for g in ([HERO_GROUP[w] for w in S.WHO if w in HERO_GROUP] if S.HERO else
          ("gemma in Qwen's runs", "Qwen in Qwen's runs", "gemma in its own runs", "gemma's own full runs")):
    summary([r for r in rows if r["group"] == g], g)
print("\n######## the four gap environments, at Qwen's incidents")
for env_label in ("rendering engine", "ai_rd", "is_prime", "pre-commit hook"):
    for g in ([HERO_GROUP[w] for w in S.WHO if w in HERO_GROUP] if S.HERO else ("gemma in Qwen's runs", "Qwen in Qwen's runs")):
        sel = [r for r in rows if r["group"] == g and GAP_ENVS.get(r["env"]) == env_label and r["incident"] in (True, "True")]
        if sel:
            summary(sel, f"{env_label} · {g}")
