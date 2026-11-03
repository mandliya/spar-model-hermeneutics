"""Chart 5M (3 Nov 2026): the propensity / capability split in one picture. Reads results/intent_labels.jsonl and the
replay tables, keeps only gemma's NON-cheating tries (the tries where a gap could come from), and splits them three ways:
  PROPENSITY   declined knowingly: gemma names the shortcut or the rule and does something else
  CAPABILITY   could not: the honest path failed mechanically (errors, malformed calls, repetition, collapse)
  UNRESOLVED   honest with no sign either way, or believed it had succeeded (silence is not evidence of intent)
One row per gap environment, pooled over the restart points on Qwen's cheating traces (N, N−1, N−2, N−5); two pooled rows
at the bottom (all cheating traces; all clean runs). Right: Qwen's hack rate, gemma's, the gap in points, and Qwen's own
three-way split for comparison. Three stat tiles on top carry the headline numbers. Prints the markdown table.
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/split_chart.py
"""
import collections, csv, json, pathlib, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID, RED = "#17140f", "#6f6a60", "#e1e0d9", "#d03b3b"
PROP, CAP, UNR = "#0d6b4a", "#2a78d6", "#b8bfc7"
ENVS = ["coding", "coding_norules", "precommit_hook", "ai_rd", "optimize_rendering_engine", "dashboard_costly_failure", "grade_your_own"]
NAME = {"coding": "coding (is_prime, rules)", "coding_norules": "coding_norules (is_prime)", "precommit_hook": "precommit_hook", "ai_rd": "ai_rd",
        "optimize_rendering_engine": "optimize_rendering_engine", "dashboard_costly_failure": "dashboard_costly_failure", "grade_your_own": "grade_your_own"}
GROUP = {"declined_knowingly": "prop", "could_not": "cap", "honest_no_sign": "unr", "believed_success": "unr", "unlabelled": "unr"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})

labels = {}
for l in open(HERE / "results" / "intent_labels.jsonl"):
    r = json.loads(l); labels[r["run"]] = r["label"] or "unlabelled"
cnt = collections.defaultdict(collections.Counter)      # (env, kind, who) -> {hacked, prop, cap, unr, nosign, believed, n}
for f in ("gap_replays.csv", "gap_replays_precommit.csv"):
    for r in csv.DictReader(open(HERE / "results" / f)):
        if r["hacked"] not in ("0", "1") or r["env"] not in ENVS:
            continue
        kind = "gap" if r["incident"] == "True" else "clean"
        c = cnt[(r["env"], kind, r["who"])]; c["n"] += 1
        if r["hacked"] == "1":
            c["hacked"] += 1
        else:
            lab = labels.get(r["run"], "unlabelled"); c[GROUP[lab]] += 1; c[lab] += 1
for who in ("qwen", "gemma"):
    for kind in ("gap", "clean"):
        for e in ENVS:
            cnt[("ALL", kind, who)].update(cnt[(e, kind, who)])

rows = [(e, "gap") for e in ENVS if cnt[(e, "gap", "gemma")]["n"] >= 10] + [("ALL", "gap"), ("ALL", "clean")]
rows[:-2] = sorted(rows[:-2], key=lambda r: -(cnt[(r[0], "gap", "gemma")]["cap"] / max(1, cnt[(r[0], "gap", "gemma")]["n"] - cnt[(r[0], "gap", "gemma")]["hacked"])))
LABEL = {"gap": "", "clean": ""}

H = 2.75 + 0.72 * len(rows) + 1.1
fig = plt.figure(figsize=(13.5, H)); ax = fig.add_axes([0.21, 1.05 / H, 0.46, 0.72 * len(rows) / H])
table = ["| environment | traces | gemma non-cheating tries | propensity: declined knowingly | capability: could not | unresolved (no sign · believed) | Qwen hacked | gemma hacked | gap (pts) | Qwen's split (prop / cap / unresolved) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for i, (env, kind) in enumerate(rows):
    g, q = cnt[(env, kind, "gemma")], cnt[(env, kind, "qwen")]; gn = g["n"] - g["hacked"]; qn = q["n"] - q["hacked"]
    if env == "ALL":
        ax.plot([-75, 212], [i - 0.5, i - 0.5], color=INK if kind == "gap" else GRID, lw=1.0 if kind == "gap" else 0.6, zorder=0, clip_on=False)
        title = "ALL GAP ENVIRONMENTS, cheating traces" if kind == "gap" else "ALL GAP ENVIRONMENTS, clean runs"
        sub = "restart at N, N−1, N−2, N−5 on Qwen's cheating runs" if kind == "gap" else "restart before Qwen's last clean turn"
    else:
        title = NAME[env]; sub = "cheating traces, pooled over restart points"
        if i:
            ax.plot([-75, 212], [i - 0.5, i - 0.5], color=GRID, lw=0.6, zorder=0, clip_on=False)
    ax.text(-2, i, f"{title}\n{sub}", ha="right", va="center", fontsize=8.6 if env != "ALL" else 9, fontweight="bold", linespacing=1.3)
    x = 0
    for key, col, txtcol in (("prop", PROP, "white"), ("cap", CAP, "white"), ("unr", UNR, INK)):
        v = g[key] / gn * 100 if gn else 0
        if v:
            ax.barh(i, v, left=x, height=0.58, color=col, zorder=3, edgecolor="white", lw=0.8)
            lab = f"{g[key]}\n{v:.0f}%" if v >= 9 else (f"{g[key]}" if v >= 4 else "")
            ax.text(x + v / 2, i, lab, ha="center", va="center", fontsize=8.6 if v >= 9 else 7.5, fontweight="bold", color=txtcol, linespacing=1.1)
        x += v
    ax.text(102, i, f"n={gn}", va="center", fontsize=7.8, color=MUTED)
    qr, gr = q["hacked"] / q["n"] * 100, g["hacked"] / g["n"] * 100; gap = qr - gr
    ax.text(130, i, f"{q['hacked']}/{q['n']}\n{qr:.0f}%", ha="center", va="center", fontsize=8.2, color=INK, linespacing=1.15)
    ax.text(148, i, f"{g['hacked']}/{g['n']}\n{gr:.0f}%", ha="center", va="center", fontsize=8.2, color=INK, linespacing=1.15)
    ax.text(165, i, f"{gap:+.0f}", ha="center", va="center", fontsize=10, fontweight="bold", color=RED if gap >= 10 else MUTED)
    qs = f"{q['prop'] / qn * 100:.0f} / {q['cap'] / qn * 100:.0f} / {q['unr'] / qn * 100:.0f}%" if qn else "—"
    ax.text(192, i, qs, ha="center", va="center", fontsize=8.2, color=MUTED)
    table.append(f"| {env if env != 'ALL' else '**all gap environments**'} | {'cheating' if kind == 'gap' else 'clean'} | {gn} | {g['prop']} ({g['prop'] / gn * 100:.0f}%) | {g['cap']} ({g['cap'] / gn * 100:.0f}%) | "
                 f"{g['unr']} ({g['unr'] / gn * 100:.0f}%; {g['honest_no_sign']} · {g['believed_success']}) | {q['hacked']}/{q['n']} | {g['hacked']}/{g['n']} | {gap:+.0f} | {qs} |")
for x, t in ((50, "gemma's non-cheating tries, split by why it did not cheat"), (130, "Qwen\nhacked"), (148, "gemma\nhacked"), (165, "gap\n(pts)"), (192, "Qwen's own split\nprop / cap / unresolved")):
    ax.text(x, -0.95, t, ha="center", va="center", fontsize=7.9, color=MUTED, linespacing=1.2, fontweight="bold" if x == 50 else "normal")
ax.set_xlim(0, 212); ax.set_ylim(len(rows) - 0.5, -1.45)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_bounds(0, 100); ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", left=False, labelleft=False); ax.tick_params(axis="x", length=0)
for xv in (25, 50, 75):
    ax.plot([xv, xv], [-0.5, len(rows) - 0.5], color="white", lw=0.8, zorder=4, alpha=0.5)

# ---- stat tiles: the headline numbers (all gap environments, cheating traces) -------------------------------------------
g = cnt[("ALL", "gap", "gemma")]; gn = g["n"] - g["hacked"]
fig.text(0.02, 1 - 0.18 / H, f"5M · Why gemma does not cheat where Qwen does: of its {gn} non-cheating tries on Qwen's cheating traces, "
         f"{g['prop'] / gn * 100:.0f}% is choice, {g['cap'] / gn * 100:.0f}% is capability, {g['unr'] / gn * 100:.0f}% unresolved",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.02, 1 - 0.5 / H, "Same data as chart 5L, one question only. Each try is a continuation of one of Qwen's cheating runs from the same turn by both models; the judge "
         "(Claude Sonnet 4.5) reads the model's own turns and must quote the span behind each label.\nRows are sorted by the capability share; the gap column is Qwen's hack rate minus gemma's "
         "(chart 4). The last column gives Qwen's own split of its non-cheating tries, so the two models can be compared.", fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
tiles = [(PROP, "CHOICE  (propensity)", g["prop"], "declined knowingly: names the shortcut\nor the rule, does something else"),
         (CAP, "CAPABILITY", g["cap"], "could not: the honest path failed\n(errors, malformed calls, repetition, collapse)"),
         (UNR, "UNRESOLVED", g["unr"], f"honest, nothing about the shortcut ({g['honest_no_sign']})\nor believed it had succeeded ({g['believed_success']})")]
for k, (col, name, v, desc) in enumerate(tiles):
    x0 = 0.02 + k * 0.325
    tx = fig.add_axes([x0, 1 - 2.45 / H, 0.3, 1.4 / H]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(0, 1)
    tx.add_patch(plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.1 if col != UNR else 0.35, lw=0, transform=tx.transAxes))
    tx.add_patch(plt.Rectangle((0, 0), 0.012, 1, color=col if col != UNR else MUTED, lw=0, transform=tx.transAxes))
    tx.text(0.05, 0.82, name, fontsize=9, fontweight="bold", color=col if col != UNR else MUTED, va="center")
    tx.text(0.05, 0.5, f"{v}", fontsize=22, fontweight="bold", color=INK, va="center")
    tx.text(0.05 + 0.09 * len(str(v)), 0.47, f"of {gn}  ·  {v / gn * 100:.0f}%", fontsize=10.5, color=INK, va="center")
    tx.text(0.05, 0.17, desc, fontsize=7.6, color=MUTED, va="center", linespacing=1.25)
foot = ("gemma-3-27b's non-cheating tries only (a gap can only come from tries where gemma did not do the bad act). Cheating traces pooled over the restart points; grade_your_own pools run 3 (the false "
        "report is already in the trace: both models repeat it 10/10) and run 5 (neither cheats). Source: results/intent_labels.jsonl (scripts/intent_labels.py), results/gap_replays.csv and "
        "gap_replays_precommit.csv. The spot-check (results/intent_labels_spotcheck.md) found 'believed it had succeeded' unreliable, so it is counted as unresolved; "
        "'declined knowingly' and 'could not' agreed with their quotes in 23/25 reads.")
fig.text(0.02, 0.45 / H, "\n".join(textwrap.wrap(foot, 205)), fontsize=7.6, color=MUTED, va="center", linespacing=1.35)
fig.savefig(HERE / "charts" / "q5_split.png", dpi=180, bbox_inches="tight", facecolor="white")
print("\n".join(table))
