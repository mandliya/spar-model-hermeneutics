"""Why a scored swap replay ended as "stopped" (28 Oct 2026): the light segment of charts 5A-5G. These are scored,
not lost tries: the model's reply carried no tool call, and in these environments the harness treats a plain reply as the
final report (environments/*/run_step.py, treat_no_tool_call_as_done=True), so the run ended and the final state was
scored. The reasons, from the last assistant message (first match wins):
  ran out of steps            ending step_cap
  malformed or unavailable tool call   ending parser, or a tool call the harness could not run
  believed the session was over        "the session has ended", "I have exited", tries end_task / exit
  waited for the user or for output    asks a question, "let me know", "let's wait"
  pasted code or a prompt as text      a fenced block of 12+ lines instead of a write command
  announced its next action, no call   "Let me try a different approach...", "I'll create the file..." and stops
  plan or summary, no action           anything else: lays out next steps or sums up, with no call and no claim
Writes results/stopped_reasons.csv (one row per stopped try in the gap environments, with the last 300 characters of the
reply) and prints the counts per model. gap_analysis.py reads the CSV for the chart footnotes.
  cd gigi/3-forensics/agent-interp-envs && uv run python ../scripts/stopped_reasons.py
"""
import collections, csv, json, pathlib, re

HERE = pathlib.Path(__file__).resolve().parents[1]
HONEST_ENDINGS = {"says_done", "quick_answer", "gave_up", "handed_back"}
GAP_ENVS = {"coding", "coding_norules", "optimize_rendering_engine", "optimize_rendering_engine_graded",
            "dashboard_costly_failure", "ai_rd", "grade_your_own", "precommit_hook"}
REASONS = ["repeated itself until the token limit", "announced its next action, no call",
           "tried to end the task, then said the session was over", "pasted code or a prompt as text",
           "plan or summary, no action", "waited for the user or for output", "ran out of steps",
           "malformed or unavailable tool call", "empty reply"]
# "tried to end the task": the dashboard prompt (configs/dashboard_perf/costly_failure.yaml, from the paper) lists an
# `end_task` tool, but the framework deliberately offers none (environments/dashboard_perf/tools.py): a plain reply ends
# the run. gemma follows the prompt, runs `end_task`, then `exit`, then writes "the session has ended" — which is the
# plain reply that ends the run. The turn before usually reports the shortfall ("I couldn't meet the performance target").
# "repeated itself until the token limit": one line ("Let me try a different approach - I'll create the file...") repeated
# for tens of thousands of characters; the reply has no call because generation collapsed, not because the model stopped.
SESSION_RE = re.compile(r"session (has )?(ended|terminated)|have exited|I have exited|exit(ed)? the session|end the session|"
                        r"end_task|end_session|I'll end|I will end", re.I)
WAIT_RE = re.compile(r"\?\s*$|let me know|please (provide|share|confirm|run)|could you|would you like|wait for|let's wait|waiting for", re.I)
NEXT_RE = re.compile(r"\b(let me|I'll|I will|I need to|I should|next,? I|now I|let's)\b", re.I)


def last_assistant(run):
    """(text, has_tool_call) of the last assistant message: final/messages.json, else the last step's."""
    d = pathlib.Path(run); p = d / "final" / "messages.json"
    if not p.exists():
        steps = sorted(d.glob("step-*/messages.json"), key=lambda q: int(q.parent.name.split("-")[1]))
        if not steps:
            return None, False
        p = steps[-1]
    try:
        msgs = json.loads(p.read_text())
    except Exception:
        return None, False
    if isinstance(msgs, dict):
        msgs = msgs.get("messages", [])
    a = [m for m in msgs if m.get("role") == "assistant"]
    if not a:
        return "", False
    c = a[-1].get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return (c or ""), bool(a[-1].get("tool_calls"))


def reason(ending, text, has_call):
    if ending == "step_cap":
        return "ran out of steps"
    if ending == "parser" or has_call:
        return "malformed or unavailable tool call"
    t = (text or "").strip()
    if not t:
        return "empty reply"
    lines = collections.Counter(s.strip()[:60] for s in re.split(r"\n+", t) if len(s.strip()) > 20)
    if len(t) > 20000 or (lines and lines.most_common(1)[0][1] >= 5):
        return "repeated itself until the token limit"
    if SESSION_RE.search(t):
        return "tried to end the task, then said the session was over"
    if WAIT_RE.search(t):
        return "waited for the user or for output"
    if len(re.findall(r"```", t)) >= 2 and t.count("\n") > 12:
        return "pasted code or a prompt as text"
    if NEXT_RE.search(t):
        return "announced its next action, no call"
    return "plan or summary, no action"


if __name__ == "__main__":
    rows = []
    for r in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
        if r["hacked"] != "0" or r["ending"] in HONEST_ENDINGS or r["env"] not in GAP_ENVS:
            continue
        text, has_call = last_assistant(HERE / r["run"] if not r["run"].startswith("/") else r["run"])
        rows.append({"env": r["env"], "point": r["point"], "incident": r["incident"], "who": r["who"], "ending": r["ending"],
                     "turns": r["turns"], "reason": reason(r["ending"], text, has_call),
                     "last_reply": (text or "").strip().replace("\n", " ")[-300:], "run": r["run"]})
    with open(HERE / "results" / "stopped_reasons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    c = collections.Counter((x["who"], x["reason"]) for x in rows)
    for who in ("qwen", "gemma"):
        n = sum(v for (w, _), v in c.items() if w == who)
        print(f"{who}: {n} stopped tries")
        for rs in REASONS:
            if c[(who, rs)]:
                print(f"   {c[(who, rs)]:3d}  {rs}")
    print(f"{len(rows)} rows -> results/stopped_reasons.csv")
