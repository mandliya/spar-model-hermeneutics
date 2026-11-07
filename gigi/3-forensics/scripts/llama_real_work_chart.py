"""Did the model open the code it was asked to change? (7 Nov 2026, for the Slack post that drops Llama.)
One chart, charts/llama_real_work.png: for each of the 8 main tasks, the share of runs in which the model read at least one
source file, for Llama under the three end-of-run rules (no check: hero_solo; soft check: hero2_llama; hard rule: hero4_llama)
next to gemma-3 27B and Qwen3-Next 80B with the soft check (hero2_gemma, hero2_qwen), so the room can see what a working
model looks like. The "opened a source file" rule is the one in scripts/hero4_llama_spotcheck.py and
scripts/llama_check_spotcheck.py. Run from gigi/3-forensics/agent-interp-envs:  uv run --with matplotlib python ../scripts/llama_real_work_chart.py
"""
import json, os, pathlib, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import full_runs_table as T
import confirm_effect_charts as C

READ_RE = re.compile(r"\b(cat|less|head|tail|sed -n|nl|bat|view|open)\b[^|;&\n]*\.(py|js|mjs|cjs|ts|tsx|jsx|c|cc|cpp|h|rs|go|java|json|yaml|yml|toml|md|txt|sh)\b|"
                     r"\b(grep|rg|ag)\b|\bread_file\b|\bview_file\b", re.I)
WRITE_RE = re.compile(r"(cat\s*>|tee\s|>\s*\S+\.(py|js|ts|c|h|json|yaml|yml|txt|md|sh)|<<\s*['\"]?EOF|\bsed -i\b|apply_patch|write_file|"
                      r"echo .*>\s*\S|python\s*-\s*<<|git commit|cp\s|mv\s)", re.I)
MAIN8 = ["coding", "coding_norules", "precommit_hook", "dashboard_costly_failure", "grade_your_own",
         "optimize_rendering_engine", "optimize_rendering_engine_graded", "ai_rd"]
PASSES = [  # label, model, stages, colour
    ("Llama, no check (4 Nov)", "llama-3.3-70b", ("hero_solo", "hero_solo_key2"), "#b5aea4"),
    ("Llama, soft check: asked once (4 Nov)", "llama-3.3-70b", ("hero2_llama", "hero2_llama_key"), "#a99cf5"),
    ("Llama, hard rule: only end_task ends the run (7 Nov)", "llama-3.3-70b", ("hero4_llama", "hero4_llama_key"), "#6c4ff0"),
    ("gemma-3 27B, soft check", "gemma-3-27b", ("hero2_gemma", "hero2_gemma_key"), "#2f6fd6"),
]  # Qwen3-Next 80B left out on purpose (7 Nov): it was never part of the Llama question, so it only muddied the picture


def txt(x):
    c = x.get("content"); return c if isinstance(c, str) else ("" if c is None else json.dumps(c))


def calls(x):
    return [((t.get("function") or {}).get("name") or t.get("name"), (t.get("function") or {}).get("arguments") or "")
            for t in x.get("tool_calls") or []]


def load_msgs(rd):
    f = rd / "final" / "messages.json"
    if not f.exists():
        steps = sorted(rd.glob("step-*/messages.json"), key=lambda p: int(re.search(r"step-(\d+)", str(p)).group(1)))
        if not steps:
            return None
        f = steps[-1]
    try:
        return json.load(open(f))
    except Exception:
        return None


def text_calls(x):
    """gemma runs in text-tool mode: the call is in the message body, so search the whole text too."""
    return txt(x)


counts = {}   # (pass index, task) -> (opened, wrote, n)
for i, (label, model, stages, _) in enumerate(PASSES):
    for env, _, rd, stage in T.hero_runs(stages, model):
        if env not in MAIN8:
            continue
        m = load_msgs(pathlib.Path(rd))
        if m is None:
            continue
        asst = [x for x in m if x.get("role") == "assistant"]
        args = " \n ".join(a for x in asst for _, a in calls(x)) + " \n " + " \n ".join(text_calls(x) for x in asst if not calls(x))
        o, w, n = counts.get((i, env), (0, 0, 0))
        counts[(i, env)] = (o + bool(READ_RE.search(args)), w + bool(WRITE_RE.search(args)), n + 1)

for env in MAIN8:
    print(env, "  ".join(f"{PASSES[i][0].split(',')[0]}: {counts.get((i, env), (0, 0, 0))[0]}/{counts.get((i, env), (0, 0, 0))[2]}" for i in range(len(PASSES))))

# one row per task, five dots per row, the value next to each dot
fig = plt.figure(figsize=(12, 8.6))
fig.text(0.015, 0.975, "Did the model open the code it was asked to change? Llama under three end-of-run rules, next to gemma",
         fontsize=12, fontweight="bold", va="top")
fig.text(0.015, 0.935, "Each dot is the share of finished runs of that task in which the model read at least one source file (cat, head, grep, sed -n, a read tool...).\n"
         "Llama's three passes are three separate sets of runs under the three rules. gemma-3 27B ran with the soft check, the same setting as\n"
         "Llama's middle dot, so it shows what a model that actually attempts the task looks like. Counts on the right: runs that opened a file / runs.",
         fontsize=8.5, color="#555", va="top", linespacing=1.4)
ax = fig.add_axes([0.335, 0.175, 0.42, 0.69])
ys = list(range(len(MAIN8)))[::-1]
off = [0.27, 0.09, -0.09, -0.27]
for y, env in zip(ys, MAIN8):
    ax.axhline(y - 0.5, color="#eee", lw=0.8, zorder=0)
    for i, (label, model, stages, col) in enumerate(PASSES):
        o, w, n = counts.get((i, env), (0, 0, 0))
        if n == 0:
            continue
        x = 100 * o / n
        ax.plot(x, y + off[i], "o", color=col, ms=8, zorder=3)
        ax.text(x + 2, y + off[i], f"{o} of {n}", fontsize=7.5, color=col, va="center", fontweight="bold" if i == 2 else None)
ax.set_yticks(ys); ax.set_yticklabels([C.LABEL.get(e, e) + ("  ◆" if e in C.MAIN else "") for e in MAIN8], fontsize=9.5, fontweight="bold")
ax.set_xlim(-2, 118); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("runs in which the model opened a source file", fontsize=9.5, color="#444")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eee", lw=0.8); ax.set_axisbelow(True)
handles = [plt.Line2D([], [], marker="o", color=c, ls="", ms=8, label=l) for l, _, _, c in PASSES]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.008), ncol=2, fontsize=8.5, frameon=False)
fig.savefig("charts/llama_real_work.png", dpi=160, facecolor="white"); plt.close(fig)
print("wrote charts/llama_real_work.png")
