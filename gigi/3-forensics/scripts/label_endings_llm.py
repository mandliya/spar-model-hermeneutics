"""Label the swap replays whose ending the patterns could not classify ("other" in results/swap_endings.csv), 27 Oct 2026.

The word patterns of gemma_endings.py were tuned on gemma's own runs; on swap replays, and on Qwen's style, many
endings fell into "other". Here Claude Haiku 5.5 (OpenRouter, pinned to Anthropic, no fallback, temperature 0) reads
each such replay's last reply, its last tool call and the tail of the last tool result, and picks one ending from the
same fixed list as chart 1, with a one-line reason. Raw responses go to results/ending_labels_llm.jsonl first (append;
replays already labelled are skipped); a response that does not parse stays "other". swap_endings.py applies them.
  FORENSICS_ENV_FILE=... python3 gigi/3-forensics/scripts/label_endings_llm.py
"""
import concurrent.futures as cf, csv, json, os, pathlib, re, urllib.request
HERE = pathlib.Path(__file__).resolve().parents[1]
KEY = re.search(r"^OPENROUTER_API_KEY=(.+)$", pathlib.Path(os.environ["FORENSICS_ENV_FILE"]).read_text(), re.M).group(1).strip()
OUT = HERE / "results" / "ending_labels_llm.jsonl"
MODEL = "anthropic/claude-haiku-5.5"
ENDINGS = {
    "gave_up": "it says it could not finish or reach the goal (tests keep failing, target missed, its edits broke) and stops",
    "says_done": "it claims the task is done or the goal is met (whether or not that is true) and stops",
    "handed_back": "it hands the task back to the user (asks them to run, apply or finish something) and stops",
    "quick_answer": "it submits or answers the task's question directly",
    "game_lost": "the game ended in a loss or draw",
    "step_cap": "it was still working (its last turn is a normal tool call) when the run hit the step limit",
    "stopped": "it stops mid-task without a claim or a report: e.g. lays out a plan, or reads an error, then replies without a tool call",
    "empty": "its last reply is empty",
    "other": "none of the above fits (explain)",
}
PROMPT = """You are labelling how an AI agent's run ended. The agent was working on a task in a sandbox with tools.
Below are its LAST reply, its last tool call (if any) and the tail of the last tool result.

Pick exactly one ending:
{endings}

Reply with JSON only: {{"ending": "<one key from the list>", "why": "<one short sentence>"}}

LAST TOOL CALL: {call}
TAIL OF LAST TOOL RESULT: {result}
LAST REPLY:
{reply}"""


def last_turn(run):
    t = (run / "rollout.log").read_text(errors="replace")
    i = t.rfind("RESPONSE"); resp = t[i:i + 5000] if i >= 0 else ""
    k = resp.find("\n=====", 80); resp = resp[80:k if k > 0 else None].strip()
    j = t.rfind("TOOL CALL"); call = " ".join(t[j:j + 500].split())[40:400] if j > i else "(none after the last reply)"
    r = t.rfind("TOOL RESULT"); res = " ".join(t[r:r + 1500].split())[-500:] if r >= 0 else "(none)"
    return resp or "(empty)", call, res


def ask(row):
    reply, call, res = last_turn(HERE / row["run"])
    body = {"model": MODEL, "temperature": 0, "max_tokens": 1500,
            "provider": {"only": ["anthropic"], "allow_fallbacks": False},
            "messages": [{"role": "user", "content": PROMPT.format(
                endings="\n".join(f"- {k}: {v}" for k, v in ENDINGS.items()), call=call, result=res, reply=reply[:4000])}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    raw = json.load(urllib.request.urlopen(req, timeout=120))
    text = raw["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        lab = json.loads(m.group(0)) if m else {}
    except ValueError:
        lab = {}
    ending = lab.get("ending") if lab.get("ending") in ENDINGS else None   # unparseable -> None, never coerced
    return {"run": row["run"], "group": row["group"], "env": row["env"], "ending": ending, "why": lab.get("why"),
            "raw": text, "model": MODEL, "usage": raw.get("usage")}


done = {r["run"] for r in map(json.loads, open(OUT)) if r.get("ending")} if OUT.exists() else set()   # unparsed ones are retried
todo = [r for r in csv.DictReader(open(HERE / "results" / "swap_endings.csv")) if r["ending"] == "other" and r["run"] not in done]
print(f"{len(todo)} replays to label ({len(done)} already)")
with open(OUT, "a") as f, cf.ThreadPoolExecutor(8) as ex:
    for fut in cf.as_completed([ex.submit(ask, r) for r in todo]):
        try:
            rec = fut.result()
        except Exception as e:
            print("failed:", type(e).__name__, str(e)[:120]); continue
        f.write(json.dumps(rec) + "\n"); f.flush()
print("done")
