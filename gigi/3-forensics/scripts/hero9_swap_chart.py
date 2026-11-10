"""Chart and counts for the symmetric hero swap (10 Nov 2026): gemma-3 27B and Qwen3.6 27B, each continuing its own
cheating runs and taking over the other's, at N (the step of the bad act), N-1 and on clean runs. Reads
results/hero9_swap.csv (scripts/hero9_swap_score.py).
  charts/hero9_swap_gaps.png   one row per source, task and takeover point: the source continuing (filled) next to the
                               other hero taking over (hatched), k of n printed, 95% Wilson intervals as thin lines
  results/hero9_swap_summary.md   the same as a table, plus the distinct source runs where the two arms differ
Run from gigi/3-forensics/agent-interp-envs:  env -u VIRTUAL_ENV uv run --with matplotlib python ../scripts/hero9_swap_chart.py
"""
import collections, csv, math, os, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; os.chdir(H); sys.path.insert(0, str(H / "scripts"))
import chart_style as ST
import confirm_effect_charts as CE

COL = {"gemma": ST.GEMMA, "qwen36": ST.QWEN}
NAME = {"gemma": "gemma-3 27B", "qwen36": "Qwen3.6 27B"}
OTHER = {"gemma": "qwen36", "qwen36": "gemma"}
TAKES = [("N", "at the step of the bad act"), ("N−1", "one step before"), ("clean", "clean run, at its last step")]
NOTE = "Both heroes fp8 on DeepInfra via OpenRouter, plain setting (no end check, no nudge). Bar = share of scored continuations with the task's own bad act; thin line = 95% Wilson interval."


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


src_csv = H / "results" / "hero9_swap.csv"
if not src_csv.exists():
    print("not yet: results/hero9_swap.csv does not exist (hero9_swap_score.py writes it); no chart"); sys.exit(0)
rows = list(csv.DictReader(open(src_csv)))
cell = collections.defaultdict(lambda: [0, 0])          # (src, env, take, role) -> [k, n]
per_point = collections.defaultdict(dict)                # (src, env, point) -> role -> (k, n)
for r in rows:
    c = cell[(r["src"], r["env"], r["take"], r["role"])]; c[0] += int(r["k"]); c[1] += int(r["n"])
    per_point[(r["src"], r["env"], r["point"])][r["role"]] = (int(r["k"]), int(r["n"]))
keys = sorted({(s, e, t) for (s, e, t, _) in cell}, key=lambda x: (x[0], list(CE.LABEL).index(x[1]) if x[1] in CE.LABEL else 99, [t for t, _ in TAKES].index(x[2]) if x[2] in [t for t, _ in TAKES] else 9))
if not keys:
    print("no scored cells yet"); sys.exit(0)

fig_h = 1.2 + 0.42 * len(keys)
fig, ax = plt.subplots(figsize=(13, fig_h))
fig.text(0.012, 1 - 0.25 / fig_h, "Does the other hero cheat where this hero cheated? gemma-3 27B and Qwen3.6 27B, each in the other's shoes",
         fontsize=12.5, fontweight="bold", va="top", color=ST.INK)
fig.text(0.012, 1 - 0.55 / fig_h, "Per row: the source hero continuing its own run (filled bar) and the other hero taking over at the same step (hatched bar), continuations pooled over the row's points. "
         + NOTE, fontsize=8.3, color=ST.MUTED, va="top", wrap=True)
plt.subplots_adjust(left=0.33, right=0.97, top=1 - 1.0 / fig_h, bottom=0.6 / fig_h)
ax.set_xlim(0, 1.22); ax.set_ylim(len(keys) - 0.5, -0.5); ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of continuations with the bad act")
lines = ["# Symmetric hero swap: the counts (written by scripts/hero9_swap_chart.py)", "", NOTE, "",
         "| source hero | task | takeover | source continuing | the other hero taking over | gap (other minus source, points) |", "|---|---|---|---|---|---|"]
last_src = None
for i, (s, e, t) in enumerate(keys):
    if s != last_src:
        ax.axhline(i - 0.5, color=ST.RULE, lw=1.2); last_src = s
    label = f"{NAME[s]} → {NAME[OTHER[s]]}   {CE.LABEL.get(e, e)}   {t}"
    ax.text(-0.01, i, label, ha="right", va="center", fontsize=8.2, color=ST.INK, transform=ax.get_yaxis_transform())
    ks, ns = cell[(s, e, t, "self")]; ko, no = cell[(s, e, t, "other")]
    for off, (k, n, col, hatch) in ((-0.18, (ks, ns, COL[s], None)), (0.18, (ko, no, COL[OTHER[s]], "////"))):
        if n == 0:
            ax.text(0.005, i + off, "no scored continuation", fontsize=6.8, color=ST.FAINT, va="center"); continue
        p = k / n; lo, hi = wilson(k, n)
        ax.barh(i + off, p, height=0.3, color=col, hatch=hatch, edgecolor="white" if hatch else col, lw=0.6, zorder=3)
        ax.plot([lo, hi], [i + off] * 2, color=col, lw=0.7, alpha=0.5, zorder=4)
        ax.text(p + 0.012, i + off, f"{k} of {n}", fontsize=6.8, color=ST.TEXT, va="center", bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8), zorder=5)
    gap = f"{100 * (ko / no - ks / ns):+.0f}" if ns and no else "-"
    lines.append(f"| {NAME[s]} | {CE.LABEL.get(e, e)} | {t} | {ks} of {ns} | {ko} of {no} | {gap} |")
handles = [plt.Rectangle((0, 0), 1, 1, color=COL["gemma"], label="gemma-3 27B writing"), plt.Rectangle((0, 0), 1, 1, color=COL["qwen36"], label="Qwen3.6 27B writing"),
           plt.Rectangle((0, 0), 1, 1, fc="white", ec=ST.MUTED, hatch="////", label="hatched: taking over the other hero's run")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.004), ncol=3)
fig.savefig("charts/hero9_swap_gaps.png", dpi=160); plt.close(fig)
print("wrote charts/hero9_swap_gaps.png")

# the distillation split: distinct source runs where the two arms differ (both scored, rates on opposite sides of 50%)
lines += ["", "## Where the two arms disagree (distinct source runs)", ""]
for s in ("gemma", "qwen36"):
    diff = collections.Counter(); seen = collections.Counter()
    for (ss, e, pt), roles in per_point.items():
        if ss != s or "self" not in roles or "other" not in roles:
            continue
        (ks, ns), (ko, no) = roles["self"], roles["other"]
        if ns and no:
            seen[e] += 1
            if (ks / ns >= 0.5) != (ko / no >= 0.5):
                diff[e] += 1
    lines.append(f"- **{NAME[s]}'s runs, {NAME[OTHER[s]]} taking over:** {sum(diff.values())} of {sum(seen.values())} points disagree" +
                 (": " + ", ".join(f"{CE.LABEL.get(e, e)} {diff[e]} of {seen[e]}" for e in sorted(seen) if diff[e]) if diff else ""))
(H / "results" / "hero9_swap_summary.md").write_text("\n".join(lines) + "\n")
print("wrote results/hero9_swap_summary.md")
