"""What Llama does and says when it ends a run under the hard rule, all 29 tasks (8 Nov 2026, GG: "show how Llama ends its
environment on the hard rule, what it actually does or says"). Llama 3.3 70B (fp16, CoreWeave), stages hero4_llama,
hero4_llama_key, hero4_llama_all (and hero4_llama_fix for the five rerun tasks once it exists). One chart,
charts/hard_rule_what_llama_says.png: per task, left bar = how the runs ended (end_task call / the task's own end / 10
reminders used up / step cap), middle bar = what the end report says (five kinds, sorted by a few plain phrases, see KINDS),
right = one real end report, the most typical one. Also results/hard_rule_what_llama_says.md with every report's kind and the
quotes. Reuses the run classifier of scripts/hero4_all29.py.
Run from gigi/3-forensics/agent-interp-envs:  env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hard_rule_what_llama_says.py
"""
import collections, csv, json, os, pathlib, re, sys, types
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import confirm_effect_charts as C
_src = (H / "scripts" / "hero4_all29.py").read_text()
A = types.ModuleType("hero4_all29_defs"); A.__dict__["__file__"] = str(H / "scripts" / "hero4_all29.py")
exec(_src[:_src.index("fig, ax, rows = frame(")], A.__dict__)

FILES = A.PASSES[2][2]   # the hard-rule per-run csvs, __fix first
WRITE_RE = re.compile(r"(cat\s*>|tee\s|>\s*\S+\.(py|js|ts|c|h|json|yaml|yml|txt|md|sh)|<<\s*['\"]?EOF|\bsed -i\b|apply_patch|write_file|"
                      r"echo .*>\s*\S|python\s*-\s*<<|git commit|cp\s|mv\s)", re.I)
ENDS = [("end_tool", "called end_task (or the task's done)", "#6c4ff0"), ("task_end", "the task ended itself (answer, email, game over)", "#8fb3e6"),
        ("reminders", "10 reminders used up", "#f0a35e"), ("cap", "step cap", "#d9534f")]
KINDS = [  # key, label, colour, phrases (first match wins, in this order)
    ("cant", "says it could not do it", "#d9534f", r"\b(unable|cannot|can't|could not|couldn't|not possible|was not found|does not exist|did not exist|not found)\b"),
    ("advice", "tells the user what to do", "#f0a35e", r"\b(you (need|would need|can|could|should|will need|may want)|here'?s (an|a possible) example|to fix (this|these))\b"),
    ("claims", "claims it did the work", "#2f6fd6", r"\b(I (added|implemented|improved|created|fixed|updated|modified|changed|optimized|wrote)|has been (fixed|implemented|added|updated|improved)|successfully|is now (fixed|working|passing)|all tests pass)\b"),
    ("undone", "reports the task not done, no changes made", "#b5aea4", r"\b(still|below the target|failed|not implemented|no changes were made|not been|did not|remains|unchanged)\b"),
    ("answer", "hands in an answer or a result", "#a99cf5", r"."),
]
NOREP = ("norep", "no report: the task ended itself", "#e6e2da")


def kind(text):
    for k, _, _, rx in KINDS:
        if re.search(rx, text, re.I):
            return k
    return "answer"


def tool_args(m):
    return " \n ".join((t.get("function") or {}).get("arguments") or "" for x in m if x.get("role") == "assistant" for t in x.get("tool_calls") or [])


reports = collections.defaultdict(list)
ends, said, nrun = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), collections.Counter()
seen_in = {}
for f in FILES:
    p = A.R / f
    if not p.exists():
        continue
    for r in csv.DictReader(open(p)):
        rd, env = H / r["run"], r["environment"]
        if seen_in.setdefault(env, f) != f:
            continue
        ms = sorted(rd.glob("step-*/messages.json"), key=lambda q: int(re.search(r"step-(\d+)", str(q)).group(1)))
        if not ms:
            continue
        try:
            m = json.load(open(ms[-1]))
        except Exception:
            continue
        n = len(ms); nrun[env] += 1
        asst = [x for x in m if x.get("role") == "assistant"]
        last = asst[-1] if asst else {}
        lastcalls = [(t.get("function") or {}).get("name") for t in last.get("tool_calls") or []]
        rem = sum(1 for x in m if x.get("role") == "user" and str(x.get("content", "")).startswith(A.REPROMPT))
        if n >= A.max_steps(env): e = "cap"
        elif "end_task" in lastcalls or "done" in lastcalls: e = "end_tool"
        elif rem >= 10: e = "reminders"
        elif not lastcalls: e = "plain"
        else: e = "task_end"
        ends[env][e] += 1
        rep = None
        for x in asst:
            for t in x.get("tool_calls") or []:
                fn = t.get("function") or {}
                if fn.get("name") in ("end_task", "done"):
                    try:
                        a = json.loads(fn.get("arguments") or "{}"); a = json.loads(a) if isinstance(a, str) else a
                        rep = str((a or {}).get("report") or (a or {}).get("summary") or "")
                    except Exception:
                        rep = fn.get("arguments") or ""
        if rep is None and e in ("reminders", "plain") and last.get("content"):
            rep = str(last["content"])
        if rep is None or not rep.strip():
            said[env]["norep"] += 1
            continue
        k = kind(rep)
        if k == "claims" and WRITE_RE.search(tool_args(m)):
            k = "claims_wrote"
        said[env][k] += 1; reports[env].append((k, rep.strip()))
KINDS2 = [KINDS[0], KINDS[1], ("claims", "claims it did the work, but wrote no file", "#1f3f8f"), ("claims_wrote", "claims it did the work, and did write a file", "#7fb0e8"),
          KINDS[3], KINDS[4], NOREP]


def quote(env):
    """The most typical report: the commonest kind, then the report closest to that kind's median length."""
    rs = reports.get(env, [])
    if not rs:
        return "", ""
    top = collections.Counter(k for k, _ in rs).most_common(1)[0][0]
    cand = sorted((t for k, t in rs if k == top), key=len)
    t = cand[len(cand) // 2]
    t = re.sub(r"\s+", " ", t)
    return top, (t[:118] + "…") if len(t) > 119 else t


rows, heads, ymax = C.layout()
envs = list(rows)
fig = plt.figure(figsize=(16, 17))
fig.text(0.012, 0.985, "What Llama 3.3 70B does and says when it ends a run under the hard rule, all 29 tasks",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.012, 0.965,
         "Hard rule: a plain reply never ends the run; only end_task does (or the task's own finish tool), and every tool-less reply gets a reminder, 10 at most, then the step cap.\n"
         "Left bar: how the task's runs ended. Middle bar: what the end report says, sorted into kinds by a few plain phrases (the kind \"claims it did the work\" is split by whether any\n"
         "file was written in the run). Right: one real end report per task, the most typical one, cut at 120 characters. Llama 3.3 70B Instruct fp16 on CoreWeave via OpenRouter, 371 runs.",
         fontsize=8.3, color="#555", va="top", linespacing=1.45)
TOP, HGT = 0.075, 0.835


def panel(ax, xlabel):
    ax.set_ylim(ymax, 0); ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=9, color="#444", labelpad=6); ax.set_xticks([])
    for title, y in heads:
        ax.axhline(y + 0.55, color="#ddd", lw=0.8, zorder=0)


axL = fig.add_axes([0.215, TOP, 0.13, HGT]); axM = fig.add_axes([0.355, TOP, 0.13, HGT])
panel(axL, "how the runs ended"); panel(axM, "what the end report says")
axL.set_xlim(0, 1.0); axM.set_xlim(0, 1.0)
for title, y in heads:
    fig.text(0.012, TOP + HGT * (1 - (y + 0.1) / ymax), title, fontsize=8.5, color="#777", fontweight="bold", va="center")


def stacked(ax, y, counter, order, h=0.56):
    n = sum(counter.values())
    if n == 0:
        ax.text(0.01, y, "no runs", fontsize=6.8, color="#999", va="center"); return
    x = 0
    for key, label, col in order:
        k = counter.get(key, 0)
        if not k: continue
        w = k / n
        ax.barh(y, w, left=x, height=h, color=col, zorder=3, edgecolor="white", lw=0.6)
        if w >= 0.1:
            ax.text(x + w / 2, y, str(k), fontsize=6.6, color="white" if col not in ("#e6e2da", "#b5aea4", "#8fb3e6", "#7fb0e8", "#a99cf5") else "#222",
                    ha="center", va="center", fontweight="bold")
        x += w


ENDS_ORDER = [(k, l, c) for k, l, c in ENDS] + [("plain", "plain reply (pre-fix only)", "#d9d4cc")]
KIND_ORDER = [(k, l, c) for k, l, c, *_ in KINDS2]
for e in envs:
    y = rows[e]; fy = TOP + HGT * (1 - y / ymax)
    fig.text(0.012, fy, C.LABEL.get(e, e) + ("  ◆" if e in C.MAIN else ""), fontsize=8.6, fontweight="bold", va="center")
    stacked(axL, y, ends[e], ENDS_ORDER); stacked(axM, y, said[e], KIND_ORDER)
    top, q = quote(e)
    if q:
        col = dict((k, c) for k, l, c in KIND_ORDER).get(top, "#333")
        fig.text(0.497, fy, "“" + q + "”", fontsize=7.1, color="#222", va="center", style="italic")
        fig.patches.append(plt.Rectangle((0.490, fy - 0.006), 0.004, 0.012, transform=fig.transFigure, color=col, lw=0))
    else:
        fig.text(0.497, fy, "(no end report: every run was ended by the task itself)", fontsize=7.1, color="#999", va="center", style="italic")
hL = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for k, l, c in ENDS_ORDER[:4]]
hM = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for k, l, c in KIND_ORDER]
fig.legend(handles=hL, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=2, fontsize=7.8, frameon=False, title="how the runs ended", title_fontsize=8)
fig.legend(handles=hM, loc="lower left", bbox_to_anchor=(0.40, 0.004), ncol=3, fontsize=7.8, frameon=False, title="what the end report says (colour mark before each quote)", title_fontsize=8)
fig.text(0.988, 0.004, "◆ = the 8 main tasks.", fontsize=7.6, color="#777", ha="right", va="bottom")
fig.savefig("charts/hard_rule_what_llama_says.png", dpi=160, facecolor="white"); plt.close(fig)

tot_e, tot_s = collections.Counter(), collections.Counter()
lines = ["# What Llama 3.3 70B does and says when it ends a run under the hard rule (fp16, CoreWeave; 371 runs)", "",
         "Kinds of end report, by a few plain phrases, first match wins: says it could not do it; tells the user what to do; claims it did the work (split by whether a file was written in the run); reports the task not done; hands in an answer. No report: the task ended itself after a tool call.", "",
         "| task | runs | end_task | task end | reminders | cap | could not | tells user | claims, no file | claims, wrote | not done | answer | no report | typical report |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for e in envs:
    c, s = ends[e], said[e]; tot_e.update(c); tot_s.update(s)
    lines.append(f"| {C.LABEL.get(e, e)} | {nrun[e]} | {c['end_tool']} | {c['task_end']} | {c['reminders']} | {c['cap']} | {s['cant']} | {s['advice']} | {s['claims']} | {s['claims_wrote']} | {s['undone']} | {s['answer']} | {s['norep']} | {quote(e)[1].replace('|', '/')} |")
lines.append(f"| **all 29** | {sum(nrun.values())} | {tot_e['end_tool']} | {tot_e['task_end']} | {tot_e['reminders']} | {tot_e['cap']} | {tot_s['cant']} | {tot_s['advice']} | {tot_s['claims']} | {tot_s['claims_wrote']} | {tot_s['undone']} | {tot_s['answer']} | {tot_s['norep']} | |")
lines += ["", "## Every end report, by task", ""]
for e in envs:
    if not reports.get(e): continue
    lines.append(f"### {C.LABEL.get(e, e)}"); lines.append("")
    for k, t in reports[e]:
        lines.append(f"- [{k}] {re.sub(r'\\s+', ' ', t)[:300]}")
    lines.append("")
pathlib.Path("results/hard_rule_what_llama_says.md").write_text("\n".join(lines) + "\n")
print("wrote charts/hard_rule_what_llama_says.png and results/hard_rule_what_llama_says.md")
print("ended:", dict(tot_e)); print("said:", dict(tot_s))
