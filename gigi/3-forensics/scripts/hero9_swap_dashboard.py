"""Build the hero9 swap dashboard (11 Nov 2026): dashboard/hero9_swap.html from dashboard/hero9_swap_template.html plus the
current counts, so the page can be rebuilt after the rest of the grading lands.
Reads results/hero9_swap.csv (per point and arm), results/hero9_gap.md (the two heroes run plain, all 29 tasks) and
results/hero9_swap_loss_audit.md (totals line). Writes ONLY dashboard/hero9_swap.html (overwrites: it is a build output).
  python3 gigi/3-forensics/scripts/hero9_swap_dashboard.py
"""
import collections, csv, datetime, json, math, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from full_runs_table import MEASURES  # noqa: E402

FAMILIES = [   # same letters, names and order as story_charts.FAMILIES (importing it would run that script's charts)
    ("A", "Impossible coding tests", ["coding", "coding_norules"]),
    ("B", "Speed-up targets", ["optimize_rendering_engine", "optimize_rendering_engine_graded", "optimize_lru_cache",
                               "dashboard_costly_failure"]),
    ("C", "Self-certified ML results", ["grade_your_own", "lucky_seed", "lucky_seed_readable_grader", "ai_rd"]),
    ("D", "Leaked answers and graders", ["copying_of_answers", "leaked_tests", "leaked_scorer", "reference_extraction"]),
    ("E", "Repo maintenance", ["precommit_hook", "revert_or_refactor", "async_conversion", "swe_rebench_v2"]),
    ("F", "Games and puzzles", ["chess", "tictactoe", "tower_of_hanoi", "secret_number"]),
    ("G", "Reporting on its own work", ["puppeteer", "norvane", "code_summary_honesty", "parse_duration"]),
    ("H", "Following instructions", ["currency_conversion", "lazy_investigation"]),
    ("I", "Ethics and oversight", ["eval_tampering", "funding_email", "sandbagging"]),
]
JUDGED_ACT = {   # tasks whose bad act an Opus grader decides (grade_rollouts.py), not a state check
    "optimize_rendering_engine": "cheated (Opus grader)", "optimize_rendering_engine_graded": "cheated (Opus grader)",
    "dashboard_costly_failure": "cheated (Opus grader)", "grade_your_own": "cheated (Opus grader)",
    "copying_of_answers": "cheated (Opus grader)",
}
# where the takeover step N is the step of the bad act itself; elsewhere N is the run's last step (hero9_swap_plan.py)
N_IS_ACT = {"coding", "coding_norules", "precommit_hook", "ai_rd", "secret_number"} | set(JUDGED_ACT)


def act_of(env):
    m = MEASURES.get(env)
    return JUDGED_ACT.get(env) or (m if isinstance(m, str) else m[0] if m else "bad act")


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def lost_counts(s):
    out = collections.Counter()
    for part in filter(None, (x.strip() for x in s.split(";"))):
        m = re.match(r"(.+): (\d+)$", part)
        if m:
            out[m.group(1)] += int(m.group(2))
    return out


rows = list(csv.DictReader(open(HERE / "results/hero9_swap.csv")))
cells = collections.defaultdict(lambda: {"k": 0, "n": 0, "points": 0, "lost": collections.Counter()})
pts = collections.defaultdict(dict)
for r in rows:
    c = cells[(r["src"], r["env"], r["take"], r["role"])]
    c["k"] += int(r["k"] or 0); c["n"] += int(r["n"] or 0); c["points"] += 1; c["lost"] += lost_counts(r["lost"])
    pts[(r["src"], r["env"], r["take"], r["point"])][r["role"]] = (int(r["k"] or 0), int(r["n"] or 0))

fam_of = {e: (L, name) for L, name, envs in FAMILIES for e in envs}
order = {e: (i, j) for i, (_, _, envs) in enumerate(FAMILIES) for j, e in enumerate(envs)}
swap = []
for (src, env, take) in sorted({k[:3] for k in cells}, key=lambda t: (t[0], order.get(t[1], (99, 0)), ["N", "N−1", "clean"].index(t[2]))):
    arms = {}
    for role in ("self", "other"):
        c = cells.get((src, env, take, role), {"k": 0, "n": 0, "points": 0, "lost": collections.Counter()})
        arms[role] = {"k": c["k"], "n": c["n"], "ci": wilson(c["k"], c["n"]), "points": c["points"],
                      "lost": dict(c["lost"])}
    disagree = total = 0   # restart points where the two arms land on opposite sides of one half
    for (s, e, t, p), v in pts.items():
        if (s, e, t) == (src, env, take) and all(v.get(a, (0, 0))[1] for a in ("self", "other")):
            total += 1; disagree += (v["self"][0] / v["self"][1] >= .5) != (v["other"][0] / v["other"][1] >= .5)
    ungraded = sum(a["lost"].get("not graded", 0) for a in arms.values())
    swap.append({"src": src, "env": env, "take": take, "family": fam_of.get(env, ("?", "?")),
                 "act": act_of(env), "judged": env in JUDGED_ACT,
                 "n_is_act": env in N_IS_ACT or take == "clean", "self": arms["self"], "other": arms["other"],
                 "ungraded": ungraded, "disagree": [disagree, total]})

plain = []   # gemma's funding_email cell has no value (its plain runs never opened the legal review): k/n None
PLAIN_ENV = {"dashboard": "dashboard_costly_failure"}
for line in (HERE / "results/hero9_gap.md").read_text().splitlines():
    m = re.match(r"\| ([a-z_0-9]+)[^|]*\| (?:(\d+) of (\d+)|no value) \| (\d+) of (\d+) \| (clear)?[ -]*\|", line)
    if m:
        e = PLAIN_ENV.get(m.group(1), m.group(1))
        g = (int(m.group(2)), int(m.group(3))) if m.group(2) else (None, None); q = (int(m.group(4)), int(m.group(5)))
        m_ = MEASURES.get(e)
        plain.append({"env": e, "family": fam_of.get(e, ("?", "?")),
                      "act": act_of(e) if (m_ or e in JUDGED_ACT) else "cheated (Opus grader)",
                      "gemma": {"k": g[0], "n": g[1], "ci": wilson(*g) if g[1] else None},
                      "qwen36": {"k": q[0], "n": q[1], "ci": wilson(*q)}, "clear": bool(m.group(6))})
plain.sort(key=lambda d: order.get(d["env"], (99, 0)))

tot = re.search(r"^Totals: (.+)$", (HERE / "results/hero9_swap_loss_audit.md").read_text(), re.M).group(1)
totals = {"found": int(re.search(r"found (\d+)", tot).group(1)), "launched": int(re.search(r"launched (\d+)", tot).group(1)),
          "scored": int(re.search(r"scored (\d+)", tot).group(1)),
          "lost": dict(sum((c["lost"] for c in cells.values()), collections.Counter()).most_common())}
ungraded_by_env = collections.Counter()
for s in swap:
    ungraded_by_env[s["env"]] += s["ungraded"]

points = [{"src": s, "env": e, "take": t, "point": p, "self": v.get("self", [0, 0]), "other": v.get("other", [0, 0])}
          for (s, e, t, p), v in sorted(pts.items())]   # per restart point, for the clean-run chart
data = {"built": datetime.datetime.now().strftime("%-d %b %Y, %H:%M"), "swap": swap, "plain": plain, "totals": totals, "points": points,
        "ungraded_by_env": dict(ungraded_by_env), "families": [[L, n] for L, n, _ in FAMILIES]}
tpl = (HERE / "dashboard/hero9_swap_template.html").read_text()
out = HERE / "dashboard/hero9_swap.html"
out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)))
print(f"wrote {out.relative_to(HERE)}: {len(swap)} swap rows, {len(plain)} plain rows, {totals['scored']} scored")
