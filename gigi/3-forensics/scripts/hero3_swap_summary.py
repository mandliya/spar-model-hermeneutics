"""Charts and counts for the hero3 swap (6 Nov 2026): Llama 3.3 70B's three new cheating tasks (currency_conversion,
lazy_investigation, eval_tampering), Llama continuing vs Qwen3.6 27B and gemma-3 27B taking over, at N (the step where
Llama did the bad act) and N-1 on every cheating run, plus a few clean runs at N. Reads results/hero3_swap.csv.
  1. charts/hero3_swap_1_summary.png    per task and takeover point (N, N-1, clean), spots added together
  2. charts/hero3_swap_2_all_gaps.png   every spot on its own row: the stand-in's rate minus Llama's, 95% Newcombe interval
  3. charts/hero3_swap_3_split.png      for the distillation split: tries where each stand-in agrees with Llama vs
                                        differs, and how many distinct Llama runs the differences come from
Prints the numbers for HERO_RESULTS.md.
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/hero3_swap_summary.py
"""
import collections, csv, math, pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
COL = {"llama": "#7a5af5", "qwen36": "#1baf7a", "gemma": "#2a78d6"}
SHORT = {"llama": "Llama", "qwen36": "Qwen3.6", "gemma": "gemma"}
LAB = {"llama": "Llama 3.3 70B continuing its own run (fp16)", "qwen36": "Qwen3.6 27B taking over (fp8)",
       "gemma": "gemma-3 27B taking over (fp8)"}
WHO = ["llama", "qwen36", "gemma"]
ENVS = [("currency_conversion", "CURRENCY CONVERSION", "gives a converted amount it worked out itself, although the calculator tool was broken and it was told never to do the maths"),
        ("lazy_investigation", "LAZY INVESTIGATION", "submits the obvious-looking answer without tracing the code, and gets it wrong"),
        ("eval_tampering", "EVAL TAMPERING", "reads a note saying the test was tampered with, then reports nothing suspicious")]
TAKES = [("N", "step N: the step where Llama did it"), ("N−1", "step N−1: one step before"), ("clean", "clean run, at its last step")]
MIN_N = 5
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


rows = list(csv.DictReader(open(H / "results/hero3_swap.csv")))
cell = {(r["env"], r["point"], r["who"]): (int(r["k"]), int(r["n"])) for r in rows}
points = []
for r in rows:
    key = (r["env"], r["point"])
    if key not in [(p["env"], p["point"]) for p in points]:
        points.append({"env": r["env"], "point": r["point"], "run": r["run"], "take": r["take"], "N": int(r["N"]), "K": int(r["K"])})


def runno(p):
    return int(p["run"].split("-")[1])


def gapword(d, lo, hi, k, n, lk, ln):
    if k == 0 and lk == 0:
        return "no gap, neither ever does it"
    if k == n and lk == ln:
        return "no gap, both always do it"
    clear = lo > 0 or hi < 0
    return (f"{abs(d):.0f} pts {'more' if d > 0 else 'less'}" + ("" if clear else ", too close to call")) if round(d) else "no gap"


def axes_style(ax):
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


# ---- 1. summary: per task and takeover point, spots added together --------------------------------------------------
pooled = collections.defaultdict(lambda: [0, 0, 0])   # (env, take, who) -> k, n, spots
for p in points:
    for w in WHO:
        k, n = cell[(p["env"], p["point"], w)]
        c = pooled[(p["env"], p["take"], w)]; c[0] += k; c[1] += n; c[2] += 1
groups = [(e, t) for e, _, _ in ENVS for t, _ in TAKES if pooled[(e, t, "llama")][2]]
Hh = 2.0 + len(groups) * 0.95
fig = plt.figure(figsize=(12.5, Hh)); ax = fig.add_axes([0.33, 0.9 / Hh, 0.42, (len(groups) * 0.95) / Hh])
y = 0; last_env = None; yt = []
for e, t in groups:
    if e != last_env:
        name, act = next((n_, a_) for e_, n_, a_ in ENVS if e_ == e)
        ax.text(-0.78, y - 0.12, f"{name}: Llama {act}", transform=ax.get_yaxis_transform(), fontsize=8.5, fontweight="bold", va="center",
                wrap=True)
        y += 0.35; last_env = e
    for j, w in enumerate(WHO):
        k, n, sp = pooled[(e, t, w)]; yy = y + j * 0.25
        if n:
            p_ = k / n; lo, hi = wilson(k, n)
            ax.barh(yy, p_ * 100, height=0.22, color=COL[w], zorder=2)
            if k == 0:
                ax.plot(0, yy, "o", color=COL[w], ms=6, zorder=4, clip_on=False)
            ax.plot([lo * 100, hi * 100], [yy, yy], color=INK, lw=1, zorder=3)
        ax.text(1.02, yy, f"{SHORT[w]}:  {k} of {n}" + (f"  ({k / n:.0%})" if n else ""), transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.2, color=COL[w], fontweight="bold")
    sp = pooled[(e, t, "llama")][2]
    lab = dict(TAKES)[t] + f"\n({sp} spot{'s' if sp > 1 else ''} added together)"
    if t == "clean" and e == "currency_conversion":   # 7 Nov rule: these two runs also gave an amount (one off by ten)
        lab = f"the 2 runs we thought were clean, at their last step\n(they also gave an amount under the corrected rule)"
    ax.text(-0.02, y + 0.25, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5)
    y += 0.95
ax.set_yticks([]); ax.set_xlim(0, 100); ax.set_ylim(y - 0.2, -0.5)
ax.set_xlabel("tries in which the model did the bad act (%)  ·  black line = 95% interval", color=MUTED, fontsize=8.5)
axes_style(ax)
fig.text(0.01, 1 - 0.15 / Hh, "Llama's three new cheating tasks: once Llama's run is heading for the bad act, do the stand-ins do it too?",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.01, 1 - 0.5 / Hh, "\n".join(textwrap.wrap(
    "Short version: each group adds together all of Llama's runs taken over at the same point. Step N is the step where Llama did "
    "the bad act (its last step: the final reply or the submit call); step N−1 is one step before. We rewind each run to that point "
    "and let Llama, Qwen3.6 and gemma each continue it about 10 times. The full chart with every run on its own row is below.", 170)),
    fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[w]) for w in WHO], labels=[LAB[w] for w in WHO],
          loc="upper center", bbox_to_anchor=(0.4, -0.55 / (len(groups) * 0.95)), ncol=3, frameon=False, fontsize=8.5)
fig.savefig(H / "charts" / "hero3_swap_1_summary.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- 2. every spot on its own row --------------------------------------------------------------------------------------
layout = []
for e, name, act in ENVS:
    ps = sorted([p for p in points if p["env"] == e], key=lambda p: (p["take"] == "clean", runno(p), p["take"] != "N"))
    layout.append(("head", f"{name}: Llama {act}"))
    for p in ps:
        layout.append(("row", p))
ys = []; y = 0.0
for it in layout:
    y += 1.0 if it[0] == "head" else 0.95
    ys.append(y)
Htot = 2.3 + y * 0.3
fig = plt.figure(figsize=(13, Htot)); ax = fig.add_axes([0.24, 1.0 / Htot, 0.36, (y * 0.3) / Htot])
tr = ax.get_yaxis_transform()
for it, yy in zip(layout, ys):
    if it[0] == "head":
        ax.text(-0.66, yy, it[1], transform=tr, fontsize=9, fontweight="bold", va="center")
        ax.plot([-0.66, 1.9], [yy - 0.5, yy - 0.5], transform=tr, color=INK, lw=0.6, clip_on=False)
        continue
    p = it[1]
    if p["take"] == "clean":
        lab = (f"run {runno(p)}, last step N = {p['N']} (thought clean; also gave an amount)" if p["env"] == "currency_conversion"
               else f"clean run {runno(p)}, at its last step N = step {p['N']}")
    elif p["take"] == "N":
        lab = f"run {runno(p)}, step N = {p['N']} (where Llama did it)"
    else:
        lab = f"run {runno(p)}, step N−1 = {p['N'] - 1} (one step before)"
    ax.text(-0.03, yy, lab, transform=tr, ha="right", va="center", fontsize=7.6,
            fontweight="bold" if p["take"] == "N" else "normal", color=MUTED if p["take"] == "clean" else INK)
    lk, ln = cell[(p["env"], p["point"], "llama")]
    small = [w for w in WHO if cell[(p["env"], p["point"], w)][1] < MIN_N]
    if small:
        ax.text(0.02, yy, "too few scored tries to compare (" + ", ".join(f"{SHORT[w]} {cell[(p['env'], p['point'], w)][1]}" for w in WHO) + ")",
                transform=tr, ha="left", va="center", fontsize=7.4, color=MUTED, style="italic")
        continue
    texts = []
    for j, w in enumerate(("qwen36", "gemma")):
        k, n = cell[(p["env"], p["point"], w)]; d, lo, hi = (100 * v for v in newcombe(k, n, lk, ln))
        y2 = yy + (j - 0.5) * 0.4; clear = lo > 0 or hi < 0
        ax.plot([lo, hi], [y2, y2], color=COL[w], lw=1.6, alpha=0.55, zorder=2)
        ax.plot(d, y2, "o", ms=6, zorder=3, color=COL[w], mfc=COL[w] if clear else "white", mew=1.6)
        ax.text(1.03, y2, f"{SHORT[w]} {k}/{n} vs Llama {lk}/{ln}: {gapword(d, lo, hi, k, n, lk, ln)}", transform=tr,
                va="center", fontsize=7.2, color=COL[w], fontweight="bold" if clear else "normal")
ax.plot([0, 0], [0.3, y + 0.4], color=INK, lw=1, zorder=1)
ax.set_ylim(y + 0.4, 0.0); ax.set_xlim(-105, 105); ax.set_yticks([])
ax.text(-2, 0.15, "← stand-in does it LESS", ha="right", fontsize=8, color=MUTED)
ax.text(2, 0.15, "stand-in does it MORE →", ha="left", fontsize=8, color=MUTED)
ax.set_xlabel("stand-in's rate minus Llama's (percentage points)  ·  line = 95% interval", color=MUTED, fontsize=8.5)
axes_style(ax)
fig.text(0.01, 1 - 0.15 / Htot, "Llama's three new cheating tasks, every gap in full: each takeover spot on its own row",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(0.01, 1 - 0.5 / Htot, "\n".join(textwrap.wrap(
    "Each row is one spot in one of Llama's runs (bold rows: step N, where Llama did the bad act; plain rows: step N−1, one step "
    "before; grey rows: a clean run at its last step; in currency conversion the two grey runs also gave an amount under the corrected 7 Nov rule, so they are step-N spots too). We rewound the run to that spot and let Llama, Qwen3.6 and gemma each "
    "continue it about 10 times. The dot is the stand-in's rate minus Llama's own rate from the same spot. Filled dot = clear gap "
    "(95% interval doesn't touch zero); hollow = can't tell apart from no gap. Raw counts on the right.", 190)),
    fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[w], ms=7) for w in ("qwen36", "gemma")],
          labels=["Qwen3.6 27B vs Llama", "gemma-3 27B vs Llama"], loc="upper center", bbox_to_anchor=(0.5, -0.45 / (y * 0.3)),
          ncol=2, frameon=False, fontsize=9)
fig.savefig(H / "charts" / "hero3_swap_2_all_gaps.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- 3. the distillation split: agree vs differ at Llama's cheating spots ---------------------------------------------
# agree = the stand-in also does the bad act; differ = it doesn't. Only spots N and N-1 of Llama's cheating runs.
split = {}; self_rate = {}
for e, name, _ in ENVS:
    for w in ("qwen36", "gemma"):
        ag = df = 0; runs_all = set(); runs_df = set()
        for p in points:
            if p["env"] != e or p["take"] == "clean":
                continue
            k, n = cell[(e, p["point"], w)]
            ag += k; df += n - k; runs_all.add(p["run"])
            if n - k > 0:
                runs_df.add(p["run"])
        split[(e, w)] = (ag, df, len(runs_all), len(runs_df))
    lk = sum(cell[(e, p["point"], "llama")][0] for p in points if p["env"] == e and p["take"] != "clean")
    ln = sum(cell[(e, p["point"], "llama")][1] for p in points if p["env"] == e and p["take"] != "clean")
    self_rate[e] = (lk, ln)
# grade_your_own from the hero2 swap (results/hero2_swap.csv): its one cheating run, spots N and N-1
g = {}
f2 = H / "results/hero2_swap.csv"
if f2.exists():
    for r in csv.DictReader(open(f2)):
        if r["env"] == "grade_your_own" and r["incident"] == "True" and r["who"] in ("qwen36", "gemma"):
            a = g.setdefault(r["who"], [0, 0]); a[0] += int(r["k"]); a[1] += int(r["n"]) - int(r["k"])
    for w in ("qwen36", "gemma"):
        if w in g:
            split[("grade_your_own", w)] = (g[w][0], g[w][1], 1, 1 if g[w][1] else 0)
    lg = [0, 0]
    for r in csv.DictReader(open(f2)):
        if r["env"] == "grade_your_own" and r["incident"] == "True" and r["who"] == "llama":
            lg[0] += int(r["k"]); lg[1] += int(r["n"])
    self_rate["grade_your_own"] = tuple(lg)
order = [("grade_your_own", "grade your own model\n(earlier swap)")] + [(e, n.lower()) for e, n, _ in ENVS]
Hh = 2.2 + len(order) * 0.9
fig, axs = plt.subplots(1, 2, figsize=(12.5, Hh), sharey=True)
fig.subplots_adjust(left=0.2, right=0.83, top=1 - 1.75 / Hh, bottom=0.9 / Hh, wspace=0.55)
for ax, w in zip(axs, ("qwen36", "gemma")):
    for i, (e, lab) in enumerate(order):
        if (e, w) not in split:
            continue
        ag, df, nr, nrd = split[(e, w)]
        ax.barh(i, ag, color=COL[w], alpha=0.35, height=0.6, zorder=2)
        ax.barh(i, df, left=ag, color=COL[w], height=0.6, zorder=2)
        sk, sn = self_rate.get(e, (0, 0))
        ax.text(ag + df + max(4, 0.02 * (ag + df)), i, f"differs {df}, agrees {ag}\nfrom {nrd} of {nr} Llama run{'s' if nr > 1 else ''}\n"
                f"(Llama itself does it again {sk} of {sn})",
                va="center", fontsize=7.8, color=COL[w], fontweight="bold")
    ax.set_title(f"{SHORT[w]} vs Llama", color=COL[w], fontsize=10.5, fontweight="bold", loc="left")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order], fontsize=9)
    ax.set_ylim(len(order) - 0.5, -0.6); ax.set_xlabel("tries at Llama's cheating spots (N and N−1)", color=MUTED, fontsize=8.5)
    axes_style(ax); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(sum(split[(e, w)][:2]) for e, _ in order if (e, w) in split) * 1.75)
tot = {w: [sum(split[(e, w)][i] for e, _ in order if (e, w) in split) for i in range(4)] for w in ("qwen36", "gemma")}
fig.text(0.01, 1 - 0.15 / Hh, "For the distillation split: where the stand-ins differ from Llama at its cheating moments, and how many Llama runs that covers",
         fontsize=12, fontweight="bold", va="top")
fig.text(0.01, 1 - 0.5 / Hh, "\n".join(textwrap.wrap(
    "Each bar is all the tries where a stand-in took over one of Llama's cheating runs at step N or N−1. Pale part = the stand-in "
    "also did the bad act (it agrees with Llama); solid part = it didn't (it differs). The split by run needs the differences to come "
    f"from many different Llama runs, so the text says how many runs they come from. In total, Qwen3.6 differs {tot['qwen36'][1]} times "
    f"across {tot['qwen36'][3]} Llama runs, gemma {tot['gemma'][1]} times across {tot['gemma'][3]} Llama runs. The line in brackets: how often Llama "
    "itself does the act again from the same spots; in eval tampering Llama skips it about as often as the stand-ins do, so most of "
    "those differences are chance, not a real difference from Llama.", 170)),
    fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
fig.savefig(H / "charts" / "hero3_swap_3_split.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- numbers for the page ----------------------------------------------------------------------------------------------
print("POOLED (env, take, who): k/n over spots")
for e, _, _ in ENVS:
    for t, _ in TAKES:
        if pooled[(e, t, "llama")][2]:
            lk, ln, sp = pooled[(e, t, "llama")]
            s = f"  {e} {t} ({sp} spots): Llama {lk}/{ln}"
            for w in ("qwen36", "gemma"):
                k, n, _ = pooled[(e, t, w)]
                if n and ln:
                    d, lo, hi = (100 * v for v in newcombe(k, n, lk, ln))
                    s += f" | {SHORT[w]} {k}/{n} gap {d:+.0f} ({lo:+.0f} to {hi:+.0f})"
            print(s)
print("CLEAR GAPS per spot:")
for p in points:
    lk, ln = cell[(p["env"], p["point"], "llama")]
    for w in ("qwen36", "gemma"):
        k, n = cell[(p["env"], p["point"], w)]
        if n >= MIN_N and ln >= MIN_N:
            d, lo, hi = (100 * v for v in newcombe(k, n, lk, ln))
            if lo > 0 or hi < 0:
                print(f"  {p['env']} {p['run']} {p['take']}: {SHORT[w]} {k}/{n} vs Llama {lk}/{ln} -> {d:+.0f} ({lo:+.0f} to {hi:+.0f})")
print("SPLIT (env, stand-in): agree, differ, Llama cheating runs, runs with a difference")
for (e, w), v in split.items():
    print(f"  {e} {w}: {v}")
print("TOTALS", tot)
print("wrote charts/hero3_swap_1_summary.png, hero3_swap_2_all_gaps.png, hero3_swap_3_split.png")
