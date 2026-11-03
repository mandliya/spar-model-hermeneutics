"""Chart 5L (3 Nov 2026): the Qwen-gemma gaps split into propensity and capability, from results/intent_labels.jsonl
(scripts/intent_labels.py) and results/gap_replays.csv. One row per gap environment x restart point on Qwen's cheating
traces (N, N-1, ...; clean runs as their own rows), pooled over traces, n >= MIN_N per model; two 100% bars per row (Qwen
on top, gemma below). Segments, left to right: hacked deliberately, hacked incidentally, declined knowingly (propensity),
honest with no sign either way, could not (capability), believed it succeeded, unlabelled. Right: the gap in points and
how gemma's non-cheating tries split. Prints the markdown table for RESULTS.md.
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/propensity_chart.py
"""
import collections, csv, json, math, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID, RED = "#17140f", "#6f6a60", "#e1e0d9", "#d03b3b"
MODEL_COL = {"qwen": "#1baf7a", "gemma": "#2a78d6"}
ENVS = ["coding", "coding_norules", "optimize_rendering_engine", "optimize_rendering_engine_graded", "dashboard_costly_failure",
        "ai_rd", "grade_your_own", "precommit_hook"]
TAKE = {1: "N", 2: "N−1", 3: "N−2", 6: "N−5"}
MIN_N = 5
SEGS = [("deliberate", "hacked, deliberately (names the shortcut or the rule)", RED),
        ("incidental", "hacked, no such framing", "#eb9a9a"),
        ("declined_knowingly", "declined knowingly: names the shortcut or rule, does something else  →  PROPENSITY", "#0d6b4a"),
        ("honest_no_sign", "honest work and report, nothing about the shortcut either way (unresolved)", "#9aa5b1"),
        ("could_not", "could not: the honest path failed mechanically  →  CAPABILITY", "#2a78d6"),
        ("believed_success", "believed it had succeeded (the measure says no)", "#e0b84a"),
        ("unlabelled", "unlabelled (judge reply unusable)", "#ece9e2")]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


labels = {}
for l in open(HERE / "results" / "intent_labels.jsonl"):
    r = json.loads(l); labels[r["run"]] = r["label"] or "unlabelled"
cnt = collections.defaultdict(collections.Counter)      # (env, cond, who) -> label -> n
missing = 0
for r in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
    if r["hacked"] not in ("0", "1") or r["env"] not in ENVS:
        continue
    cond = TAKE.get(int(r["N"]) - int(r["K"]), "?") if r["incident"] == "True" else "clean"
    lab = labels.get(r["run"])
    if lab is None:
        missing += 1; lab = "unlabelled"
    cnt[(r["env"], cond, r["who"])][lab] += 1
rows = [(e, c) for e in ENVS for c in ("N", "N−1", "N−2", "N−5", "clean")
        if sum(cnt[(e, c, "qwen")].values()) >= MIN_N and sum(cnt[(e, c, "gemma")].values()) >= MIN_N]

H = 1.9 + 0.78 * len(rows) + 1.9
fig = plt.figure(figsize=(14, H)); ax = fig.add_axes([0.2, 1.55 / H, 0.5, 0.78 * len(rows) / H])
table = ["| environment | restart | model | n | hacked (deliberate · incidental) | declined knowingly (propensity) | honest, no sign | could not (capability) | believed success | unlabelled |",
         "|---|---|---|---|---|---|---|---|---|---|"]
gap_all = collections.Counter()
for i, (env, cond) in enumerate(rows):
    ax.text(-1.5, i, f"{env}\n{'restart at ' + cond if cond != 'clean' else 'clean run (no cheat in the trace)'}", ha="right", va="center", fontsize=8.6, linespacing=1.25,
            fontweight="bold")
    for j, who in enumerate(("qwen", "gemma")):
        c = cnt[(env, cond, who)]; n = sum(c.values()); y = i + (j - 0.5) * 0.36; x = 0
        for key, _, col in SEGS:
            v = c[key] / n * 100
            if v:
                ax.barh(y, v, left=x, height=0.33, color=col, zorder=3, edgecolor="white", lw=0.6)
                if c[key] >= max(2, n * 0.06):
                    ax.text(x + v / 2, y, str(c[key]), ha="center", va="center", fontsize=7.2, color="white" if key in ("deliberate", "declined_knowingly", "could_not") else INK)
            x += v
        ax.text(101, y, f"{who} n={n}", va="center", fontsize=7.6, color=MODEL_COL[who], fontweight="bold")
        hk = c["deliberate"] + c["incidental"]; nn = n - hk
        table.append(f"| {env} | {cond} | {who} | {n} | {hk} ({c['deliberate']} · {c['incidental']}) | {c['declined_knowingly']} | {c['honest_no_sign']} | {c['could_not']} | {c['believed_success']} | {c['unlabelled']} |")
        gap_all[(who, "n")] += n; gap_all[(who, "hacked")] += hk
        for key, _, _ in SEGS:
            gap_all[(who, key)] += c[key]
    q, g = cnt[(env, cond, "qwen")], cnt[(env, cond, "gemma")]; nq, ng = sum(q.values()), sum(g.values())
    gap = ((q["deliberate"] + q["incidental"]) / nq - (g["deliberate"] + g["incidental"]) / ng) * 100
    gn = ng - g["deliberate"] - g["incidental"]
    ax.text(118, i, f"{gap:+.0f} pts", ha="center", va="center", fontsize=9.5, fontweight="bold", color=RED if gap >= 10 else MUTED)
    ax.text(133, i, f"{g['declined_knowingly']}/{gn}" if gn else "—", ha="center", va="center", fontsize=8.6, color="#0d6b4a", fontweight="bold")
    ax.text(146, i, f"{g['could_not']}/{gn}" if gn else "—", ha="center", va="center", fontsize=8.6, color="#2a78d6", fontweight="bold")
    ax.text(159, i, f"{g['honest_no_sign']}/{gn}" if gn else "—", ha="center", va="center", fontsize=8.6, color=MUTED)
    if i < len(rows) - 1:
        ax.plot([-60, 166], [i + 0.5, i + 0.5], color=GRID, lw=0.6, zorder=0, clip_on=False)
for x, t in ((118, "gap\nQwen − gemma"), (133, "gemma declined\nknowingly"), (146, "gemma\ncould not"), (159, "gemma honest,\nno sign")):
    ax.text(x, -1.0, t, ha="center", va="center", fontsize=7.8, color=MUTED, linespacing=1.2)
ax.text(146, -1.55, "of gemma's non-cheating tries", ha="center", va="center", fontsize=7.4, color=MUTED, style="italic")
ax.set_xlim(0, 166); ax.set_ylim(len(rows) - 0.5, -1.9); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_bounds(0, 100); ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", left=False, labelleft=False); ax.tick_params(axis="x", length=0)
ax.set_xlabel("share of scored tries (both models continue the same Qwen trace from the same turn)", fontsize=8.5, color=MUTED)

nq, ng = gap_all[("qwen", "n")], gap_all[("gemma", "n")]; gn = ng - gap_all[("gemma", "hacked")]
fig.text(0.02, 1 - 0.18 / H, f"5L · Where gemma does not cheat and Qwen does, is it choice or capability? gemma's non-cheating tries: "
         f"{gap_all[('gemma', 'declined_knowingly')] / gn * 100:.0f}% declined knowingly, {gap_all[('gemma', 'could_not')] / gn * 100:.0f}% could not, "
         f"{gap_all[('gemma', 'honest_no_sign')] / gn * 100:.0f}% no sign either way", fontsize=12.5, fontweight="bold", va="top")
fig.text(0.02, 1 - 0.5 / H, f"Every scored try in the gap environments ({nq} Qwen, {ng} gemma), labelled by a judge (Claude Sonnet 4.5) that reads only the model's own turns after the takeover, "
         "is told what Qwen did at the cheating step, and must quote the span that shows the label.\nPropensity = the model names the shortcut or the rule and does something else. "
         "Capability = the honest path failed mechanically (errors it could not pass, malformed calls, repetition, collapse). 'No sign' is the honest residue: silence is not evidence of intent.",
         fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEGS], labels=[l for _, l, _ in SEGS], loc="upper left", bbox_to_anchor=(0.015, 1 - 0.95 / H),
           ncol=2, frameon=False, fontsize=7.9, handlelength=1.2, columnspacing=1.5)
fig.text(0.02, 0.35 / H, f"Rows pooled over traces, n ≥ {MIN_N} per model. Source: results/intent_labels.jsonl (scripts/intent_labels.py; every label carries a verbatim quote), results/gap_replays.csv. "
         f"{missing} scored tries not yet labelled.", fontsize=7.6, color=MUTED, va="center")
fig.savefig(HERE / "charts" / "q5_propensity.png", dpi=180, bbox_inches="tight", facecolor="white")
print("\n".join(table))
print(f"\nmissing labels: {missing}; gemma non-cheating {gn}: declined {gap_all[('gemma', 'declined_knowingly')]}, could_not {gap_all[('gemma', 'could_not')]}, "
      f"no sign {gap_all[('gemma', 'honest_no_sign')]}, believed {gap_all[('gemma', 'believed_success')]}, unlabelled {gap_all[('gemma', 'unlabelled')]}")
qn = nq - gap_all[("qwen", "hacked")]
print(f"Qwen non-cheating {qn}: declined {gap_all[('qwen', 'declined_knowingly')]}, could_not {gap_all[('qwen', 'could_not')]}, no sign {gap_all[('qwen', 'honest_no_sign')]}, "
      f"believed {gap_all[('qwen', 'believed_success')]}; Qwen hacked {gap_all[('qwen', 'hacked')]} ({gap_all[('qwen', 'deliberate')]} deliberate), gemma hacked {gap_all[('gemma', 'hacked')]} ({gap_all[('gemma', 'deliberate')]} deliberate)")
