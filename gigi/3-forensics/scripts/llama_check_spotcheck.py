# 7 Nov 2026: how every Llama with-check run ended (results/llama_check_spotcheck.md). Run from gigi/3-forensics.
import json, glob, os, re, collections, sys, statistics
ROOT = "results/raw"
STAGES = ["hero2_llama", "hero2_llama_key", "hero2_llama_all"]
ENDQ = "You replied without calling a tool, which would end the session."
REPROMPT = re.compile(r"No tool calls? detected")
INTENT = re.compile(r"\b(I will|I'll|Let me|Let's|Next,? I|First,? I|I am going to|I'm going to|To begin|To start|Now I|I need to|we need to|should be|Here is the (implementation|code|updated)|Here's the)\b", re.I)
GIVEUP = re.compile(r"\b(cannot|can't|unable to|not possible|no way to|I apologize|I'm sorry|impossible)\b", re.I)
LEAK = re.compile(r'"type":\s*"function"|"name":\s*"execute_command"|<\|python_tag\|>|<function=', re.I)
END_TOOLS = {"submit", "end_task", "submit_answer", "end_eval", "finish", "final_answer", "done", "submit_solution", "end", "end_session"}

def run_dirs():
    for s in STAGES:
        for cfg in glob.glob(f"{ROOT}/{s}/*/**/config.yaml", recursive=True):
            base = os.path.dirname(cfg)
            task = cfg.split("/")[3]
            ms = None

            if ms is None:
                mm = re.search(r"max_steps:\s*(\d+)", open(cfg).read()); ms = int(mm.group(1)) if mm else None
            for rd in sorted(glob.glob(base + "/run-*")):
                yield task, s, rd, ms
    for rd in sorted(glob.glob(f"{ROOT}/hero2_llama_all/swe_rebench_v2/**/run-*", recursive=True)):
        yield "swe_rebench_v2", "hero2_llama_all", rd, None

def load_msgs(rd):
    f = os.path.join(rd, "final", "messages.json")
    if not os.path.exists(f):
        steps = sorted(glob.glob(rd + "/step-*/messages.json"), key=lambda p: int(re.search(r"step-(\d+)", p).group(1)))
        if not steps: return None
        f = steps[-1]
    try: return json.load(open(f))
    except Exception: return None

rows = []
for task, stage, rd, ms in run_dirs():
    m = load_msgs(rd)
    if m is None: rows.append(dict(task=task, run=os.path.basename(rd), ending="no messages.json", steps=0)); continue
    asst = [x for x in m if x.get("role") == "assistant"]
    users = [x for x in m if x.get("role") == "user"]
    def txt(x):
        c = x.get("content"); return c if isinstance(c, str) else ("" if c is None else json.dumps(c))
    def calls(x):
        tc = x.get("tool_calls") or []
        return [ (t.get("function") or {}).get("name") or t.get("name") for t in tc ]
    nsteps = len(asst)
    ntool = sum(1 for x in asst if calls(x))
    endq = [i for i, x in enumerate(m) if x.get("role") == "user" and ENDQ in txt(x)]
    reprompt = sum(1 for x in users if REPROMPT.search(txt(x)))
    leak = sum(1 for x in asst if not calls(x) and LEAK.search(txt(x)))
    after = None
    if endq:
        nxt = next((x for x in m[endq[0]+1:] if x.get("role") == "assistant"), None)
        after = "kept working (called a tool)" if nxt is not None and calls(nxt) else "ended"
    tools = collections.Counter(n for x in asst for n in calls(x))
    last = asst[-1] if asst else None
    lc = calls(last) if last else []
    lt = txt(last) if last else ""
    if ntool == 0: ending = "f never called a tool"
    elif lc and (set(lc) & END_TOOLS): ending = "g ended with the task's end tool"
    elif lc and ms and nsteps >= ms: ending = "e hit the step limit"
    elif lc: ending = "x cut off mid-run (last step is a tool call)"
    elif GIVEUP.search(lt) and len(lt) < 1500: ending = "d gave up / said it can't"
    elif INTENT.search(lt[:300]) or LEAK.search(lt): ending = "b announced a next step / leaked code, then stopped"
    elif endq and after == "kept working (called a tool)": ending = "c kept going after the question, then gave a final report"
    else: ending = "a final report"
    rows.append(dict(task=task, stage=stage, run=os.path.basename(rd), steps=nsteps, tool_steps=ntool, max_steps=ms,
                     ending=ending, endq=len(endq), after=after, reprompt=reprompt, leak=leak, tools=dict(tools), last=lt[:220].replace("\n", " / ")))
json.dump(rows, open("results/llama_check_spotcheck_rows.json", "w"), indent=1)
by = collections.defaultdict(list)
for r in rows: by[r["task"]].append(r)
for t in sorted(by):
    rs = by[t]; st = [r["steps"] for r in rs]
    print(f"\n## {t}: {len(rs)} runs, median {statistics.median(st)} steps, limit {rs[0].get('max_steps')}")
    print("   endings:", dict(collections.Counter(r["ending"] for r in rs)))
    print("   END_QUESTION fired in", sum(1 for r in rs if r.get("endq")), "runs; after it:", dict(collections.Counter(r["after"] for r in rs if r.get("after"))))
    print("   harness re-prompts:", sum(r.get("reprompt",0) for r in rs), " leaked tool-json/code in plain text:", sum(r.get("leak",0) for r in rs))
    tl = collections.Counter(); [tl.update(r.get("tools",{})) for r in rs]; print("   tools:", dict(tl))
print("\nALL", len(rows), dict(collections.Counter(r["ending"] for r in rows)))
