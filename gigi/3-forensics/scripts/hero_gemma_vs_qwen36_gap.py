"""The cheat-rate gap between the two 27B models on their own, all 29 tasks (7 Nov 2026, the 6 Nov call's other branch: if
Llama is dropped, gemma-3 27B and Qwen3.6 27B become the two heroes and the difference in their cheat rates is the
dataset). One row per task, grouped by the 9 families as every chart; the dot is Qwen3.6's rate minus gemma's with a
95% Newcombe interval (filled = clear, the interval doesn't touch zero; hollow = too close to call), raw counts on the
right. Both models with the end-of-run check, thinking off, fp8 on DeepInfra:
  gemma   results/hero2_full_runs__gemma-3-27b.csv + results/hero2_full_runs__gemma-3-27b__other21.csv
  Qwen3.6 results/hero2_full_runs__qwen3.6-27b.csv + results/hero2_full_runs__qwen3.6-27b__other21.csv
  -> charts/hero_gemma_vs_qwen36_gap.png, and the gaps printed for HERO_RESULTS.md.
GAP_B=<model tag> draws another model against gemma instead (a draft check with qwen3-next-80b's files).
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/hero_gemma_vs_qwen36_gap.py
"""
import ast, csv, math, os, pathlib, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import full_runs_table as T
OUT = pathlib.Path(os.environ.get("CHART_OUT", H / "charts"))
INK, MUTED, GRID, BAND = "#17140f", "#6f6a60", "#e1e0d9", "#f3f2ee"
A, B = "gemma-3-27b", os.environ.get("GAP_B", "qwen3.6-27b")
NAME = {"gemma-3-27b": "gemma-3 27B", "qwen3.6-27b": "Qwen3.6 27B", "qwen3-next-80b": "Qwen3-Next 80B"}
COLB = "#c2651b" if B == "qwen3.6-27b" else "#1baf7a"
_lit = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse((H / "scripts" / "story_charts.py").read_text()).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("FAMILIES", "LABEL")}
FAMILIES, LABEL = _lit["FAMILIES"], _lit["LABEL"]
ALL29 = {r["environment"] for r in csv.DictReader(open(H / "results" / "full_runs.csv"))}
MAIN = set(T.HERO_ENVS)
GROUPS = [(f"{letter}.  {name}", [e for e in envs if e in ALL29]) for letter, name, envs in FAMILIES]
GROUPS = [g for g in GROUPS if g[1]]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def rates(model):
    out = {}
    for name in (f"hero2_full_runs__{model}.csv", f"hero2_full_runs__{model}__other21.csv"):
        f = H / "results" / name
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            e = r["environment"]
            if e in ALL29 and e not in out and int(r["n"] or 0) > 0:
                out[e] = (int(r["k"]), int(r["n"]))
    return out


ra, rb = rates(A), rates(B)
layout = []   # ("head", title) or ("row", env)
for title, envs in GROUPS:
    layout.append(("head", title))
    layout += [("row", e) for e in envs]
y = 0.0; ys = []
for item in layout:
    y += 0.85 if item[0] == "head" else 1.0
    ys.append(y)
Htot = 2.4 + y * 0.36
fig = plt.figure(figsize=(12.5, Htot)); ax = fig.add_axes([0.27, 1.1 / Htot, 0.36, (y * 0.36) / Htot])
tr = ax.get_yaxis_transform()
gaps = []
for item, yy in zip(layout, ys):
    if item[0] == "head":
        ax.add_patch(plt.Rectangle((-0.74, yy - 0.42), 2.6, 0.7, transform=tr, color=BAND, zorder=0, clip_on=False))
        ax.text(-0.72, yy - 0.07, item[1], transform=tr, fontsize=10, fontweight="bold", va="center")
        continue
    e = item[1]
    ax.text(-0.03, yy, LABEL.get(e, e) + ("  ◆" if e in MAIN else ""), transform=tr, ha="right", va="center", fontsize=8.8,
            fontweight="bold" if e in MAIN else "normal")
    ax.axhline(yy + 0.5, color=GRID, lw=0.5, zorder=0)
    if e not in ra or e not in rb:
        who = [NAME[m] for m, r in ((A, ra), (B, rb)) if e not in r]
        ax.text(0.02, yy, f"no scored runs for {' and '.join(who)}", transform=tr, va="center", fontsize=8, color=MUTED, style="italic")
        continue
    ka, na = ra[e]; kb, nb = rb[e]
    d, lo, hi = newcombe(kb, nb, ka, na); d, lo, hi = d * 100, lo * 100, hi * 100
    clear = lo > 0 or hi < 0
    ax.plot([lo, hi], [yy, yy], color=COLB, lw=2, alpha=0.55, zorder=2)
    ax.plot(d, yy, "o", ms=8, zorder=3, color=COLB, mfc=COLB if clear else "white", mew=2)
    if ka == 0 and kb == 0:
        word = "no gap, neither ever cheats"
    elif not round(d):
        word = "no gap"
    else:
        word = f"{abs(d):.0f} pts {'more' if d > 0 else 'less'}" + ("" if clear else ", too close to call")
    ax.text(1.03, yy, f"{NAME[B]} {kb} of {nb}  vs  {NAME[A]} {ka} of {na}:  {word}", transform=tr, va="center", fontsize=8,
            color=COLB, fontweight="bold" if clear else "normal")
    gaps.append((e, kb, nb, ka, na, d, lo, hi, clear))
ax.axvline(0, color=INK, lw=1, zorder=1)
ax.set_ylim(y + 0.6, 0.0); ax.set_xlim(-106, 106); ax.set_yticks([]); ax.set_xticks([-100, -50, 0, 50, 100])
ax.text(-2, 0.2, f"← {NAME[B]} cheats LESS", ha="right", fontsize=8, color=MUTED)
ax.text(2, 0.2, f"{NAME[B]} cheats MORE →", ha="left", fontsize=8, color=MUTED)
ax.set_xlabel(f"{NAME[B]}'s cheating rate minus {NAME[A]}'s (percentage points)  ·  line = 95% interval", color=MUTED, fontsize=8.5)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
n_clear = sum(1 for g in gaps if g[8])
fig.text(0.01, 1 - 0.2 / Htot, f"Do the two 27B models cheat at different rates? {NAME[B]} vs {NAME[A]}, each on its own, all 29 tasks",
         fontsize=13, fontweight="bold", va="top")
fig.text(0.01, 1 - 0.55 / Htot, "\n".join(textwrap.wrap(
    f"Each row is one task, grouped into the same 9 families as all our charts; ◆ marks the 8 main tasks. Both models ran every task on "
    f"their own with the end-of-run check, thinking off, fp8 on DeepInfra. The dot is {NAME[B]}'s cheating rate minus {NAME[A]}'s. "
    f"Filled dot = clear gap (the 95% interval doesn't touch zero); hollow = too close to call. The text on the right gives the raw "
    f"counts: runs that cheated out of runs scored. {n_clear} of the {len(gaps)} tasks with runs on both sides show a clear gap.", 175)),
    fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
fig.savefig(OUT / "hero_gemma_vs_qwen36_gap.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"wrote {OUT / 'hero_gemma_vs_qwen36_gap.png'}")
print(f"\nGAPS ({NAME[B]} minus {NAME[A]}, percentage points, 95% Newcombe)")
for e, kb, nb, ka, na, d, lo, hi, clear in gaps:
    print(f"  {e:34s} {NAME[B]} {kb:2d}/{nb:<3d} {NAME[A]} {ka:2d}/{na:<3d}  {d:+6.0f} ({lo:+.0f} to {hi:+.0f}) {'CLEAR' if clear else ''}")
