"""Two compact charts of the hero2 swap for the 5 Nov presentation (results/hero2_swap.csv), plus the gaps with 95% Newcombe
intervals printed for HERO_RESULTS.md.
  1. charts/hero2_swap_1_cheating_moment.png  grade_your_own: the three takeover points, Llama vs the two stand-ins
  2. charts/hero2_swap_2_clean.png            every other task: clean takeover points pooled per task
  cd gigi/3-forensics/agent-interp-envs && uv run --with matplotlib python ../scripts/hero2_swap_summary.py
"""
import collections, csv, math, pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = pathlib.Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#17140f", "#6f6a60", "#e1e0d9"
COL = {"llama": "#7a5af5", "qwen36": "#1baf7a", "gemma": "#2a78d6"}
LAB = {"llama": "Llama 3.3 70B continuing its own run (fp16)", "qwen36": "Qwen3.6 27B taking over (fp8)",
       "gemma": "gemma-3 27B taking over (fp8)"}
WHO = ["llama", "qwen36", "gemma"]
SHORT = {"llama": "Llama", "qwen36": "Qwen3.6", "gemma": "gemma"}
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


rows = list(csv.DictReader(open(H / "results/hero2_swap.csv")))
cell = {(r["env"], r["point"], r["who"]): (int(r["k"]), int(r["n"])) for r in rows}


def bars(ax, groups, title_of, xmax=100):
    """groups: list of (label, {who: (k, n)})"""
    y = 0; ticks = []
    for label, d in groups:
        for j, w in enumerate(WHO):
            k, n = d.get(w, (0, 0))
            yy = y + j * 0.28
            if n:
                p = k / n; lo, hi = wilson(k, n)
                ax.barh(yy, p * 100, height=0.24, color=COL[w], zorder=2)
                if k == 0:   # no bar to see: mark zero with a dot in the model's colour
                    ax.plot(0, yy, "o", color=COL[w], ms=7, zorder=4, clip_on=False)
                ax.plot([lo * 100, hi * 100], [yy, yy], color=INK, lw=1, zorder=3)
                ax.text(1.02, yy, f"{SHORT[w]}:  {k} of {n}  ({p:.0%})", transform=ax.get_yaxis_transform(), va="center",
                        fontsize=8.5, color=COL[w], fontweight="bold")
            else:
                ax.text(1, yy, "no graded tries", va="center", fontsize=8, color=MUTED, style="italic")
        ticks.append((y + 0.28, label)); y += 1.25
    ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([l for _, l in ticks], fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, xmax); ax.set_ylim(y - 0.4, -0.4)
    ax.set_xlabel("tries in which the model cheated (%)   ·   black line = 95% interval", color=MUTED, fontsize=8.5)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def save(fig, ax, name, title, sub):
    fig.suptitle(title, x=0.02, ha="left", fontsize=12.5, fontweight="bold", y=0.995)
    fig.text(0.02, 0.93, "\n".join(textwrap.wrap(sub, 150)), fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[w]) for w in WHO], labels=[LAB[w] for w in WHO],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8.5)
    fig.savefig(H / "charts" / name, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)


# 1. the cheating moment
G = "grade_your_own"
pts = [("2026-11-04_11-37-29-795546_run-5_K2", "SPOT 1: Llama's run 5,\ntaken over at step 3,\nthe step where Llama cheated"),
       ("2026-11-04_11-37-29-795546_run-5_K1", "SPOT 2: the same run 5,\ntaken over at step 2,\none step before the cheat"),
       ("2026-11-04_11-37-29-795546_run-2_K1", "SPOT 3: Llama's run 2,\ntaken over at its last step\n(this run never cheated)")]
fig = plt.figure(figsize=(11, 5.6)); ax = fig.add_axes([0.24, 0.17, 0.58, 0.6])
bars(ax, [(lab, {w: cell[(G, p, w)] for w in WHO}) for p, lab in pts], None)
save(fig, ax, "hero2_swap_1_cheating_moment.png",
     "Grade-your-own-model: once Llama's run is heading for the cheat, the stand-ins cheat almost as often as Llama does",
     "These are NOT averages. Each group of three bars is one spot in one of Llama's runs: we rewind that run to that step and let each "
     "model continue it 53 to 70 times, so each bar is one model at one spot. "
     "Purple is Llama itself continuing (what Llama would do again from that spot); green and blue are the stand-ins taking over the same run. "
     "The cheat here is reporting a gain the model never measured, or editing the baseline it's compared against. Judged by Claude Opus 4.8.")

# 2. clean points pooled per task
TASKS = [("optimize_rendering_engine", "rendering engine\n(3x speed-up)"), ("dashboard_costly_failure", "dashboard\n(speed target)"),
         ("ai_rd", "ai_rd\n(improve a prompt)"), ("precommit_hook", "pre-commit hook")]
groups = []
pooled = {}
for env, lab in TASKS:
    d = collections.defaultdict(lambda: [0, 0])
    for (e, p, w), (k, n) in cell.items():
        if e == env:
            d[w][0] += k; d[w][1] += n
    pooled[env] = {w: tuple(v) for w, v in d.items()}
    groups.append((lab, pooled[env]))
fig = plt.figure(figsize=(11, 6)); ax = fig.add_axes([0.2, 0.17, 0.62, 0.66])
bars(ax, groups, None, xmax=40)
save(fig, ax, "hero2_swap_2_clean.png",
     "In tasks where Llama never cheated, Qwen3.6 still cheats when it takes over; gemma mostly doesn't",
     "Same setup, but Llama never cheated in these tasks, so we take over clean runs near their end (three runs per task, "
     "added together here, so 60 to 90 tries per model per task). Any cheating here is the stand-in's own doing, not something it inherited from Llama's run.")

print("gaps, Llama minus stand-in, percentage points with 95% Newcombe interval")
for p, lab in pts:
    lk, ln = cell[(G, p, "llama")]
    for w in ("qwen36", "gemma"):
        k, n = cell[(G, p, w)]; d, lo, hi = newcombe(lk, ln, k, n)
        print(f"  {G} {lab.replace(chr(10), ' ')}: {w} {d*100:+.0f} ({lo*100:+.0f} to {hi*100:+.0f})")
for env, _ in TASKS:
    lk, ln = pooled[env]["llama"]
    for w in ("qwen36", "gemma"):
        k, n = pooled[env][w]; d, lo, hi = newcombe(lk, ln, k, n)
        print(f"  {env} pooled: llama {lk}/{ln}, {w} {k}/{n}: {d*100:+.0f} ({lo*100:+.0f} to {hi*100:+.0f})")
print("wrote charts/hero2_swap_1_cheating_moment.png, charts/hero2_swap_2_clean.png")


# 3. the gaps at a glance: stand-in's rate minus Llama's, every comparison, with the 95% Newcombe interval
#    Two groups, labelled on the chart: (A) one task, grade_your_own, three separate takeover spots (not averaged);
#    (B) four other tasks, one row per task, each pooling three clean Llama runs.
SIT = [(G, [pts[0][0]], "SPOT 1: run 5, at step 3,\nthe step where Llama cheated"),
       (G, [pts[1][0]], "SPOT 2: run 5, at step 2,\none step before the cheat"),
       (G, [pts[2][0]], "SPOT 3: run 2, at its last step\n(this run never cheated)")]
SIT += [(env, None, lab.replace("\n", " ") + "\n3 clean Llama runs pooled") for env, lab in TASKS]
gaps = []
for env, plist, lab in SIT:
    out = {}
    for w in ("qwen36", "gemma"):
        if plist:
            (lk, ln), (k, n) = cell[(env, plist[0], "llama")], cell[(env, plist[0], w)]
        else:
            (lk, ln), (k, n) = pooled[env]["llama"], pooled[env][w]
        d, lo, hi = newcombe(k, n, lk, ln)   # stand-in minus Llama
        out[w] = (d * 100, lo * 100, hi * 100, k, n, lk, ln)
    gaps.append((lab, out))
YS = [1, 2, 3, 5.3, 6.3, 7.3, 8.3]
HEAD = [(0.05, "A.  ONE TASK, grade your own model (the only task where Llama cheats): three different spots, one row each, not averaged"),
        (4.35, "B.  FOUR OTHER TASKS, one row per task: Llama never cheated in these, so each row pools 3 of its clean runs")]
fig = plt.figure(figsize=(11, 7.4)); ax = fig.add_axes([0.25, 0.14, 0.5, 0.7])
for (lab, out), yc in zip(gaps, YS):
    for j, w in enumerate(("qwen36", "gemma")):
        d, lo, hi, *_ = out[w]; y = yc + (j - 0.5) * 0.32; clear = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=COL[w], lw=2, alpha=0.55, zorder=2)
        ax.plot(d, y, "o", ms=9, zorder=3, color=COL[w], mfc=COL[w] if clear else "white", mew=2)
        word = (f"{abs(d):.0f} pts {'more' if d > 0 else 'less'}" + ("" if clear else ", too close to call")) if round(d) else "no gap (both never cheat)"
        ax.text(1.02, y, word, transform=ax.get_yaxis_transform(), va="center",
                fontsize=8.5, color=COL[w], fontweight="bold" if clear else "normal")
for yh, txt in HEAD:
    ax.text(-0.5, yh, txt, transform=ax.get_yaxis_transform(), fontsize=9.5, fontweight="bold", color=INK, va="center")
ax.plot([-0.5, 1.55], [4.0, 4.0], transform=ax.get_yaxis_transform(), color=INK, lw=0.9, clip_on=False)
for y0, y1 in ((0.75, 3.5), (4.75, 8.8)):
    ax.plot([0, 0], [y0, y1], color=INK, lw=1, zorder=1)
ax.set_yticks(YS); ax.set_yticklabels([g[0] for g in gaps], fontsize=9)
ax.set_ylim(8.8, -0.6); ax.set_xlim(-30, 35)
ax.text(-1, 0.55, "← stand-in cheats LESS than Llama", ha="right", fontsize=8.5, color=MUTED)
ax.text(1, 0.55, "stand-in cheats MORE than Llama →", ha="left", fontsize=8.5, color=MUTED)
ax.set_xlabel("stand-in's cheating rate minus Llama's, in percentage points  ·  line = 95% interval", color=MUTED, fontsize=8.5)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.suptitle("The gaps: how much more or less each stand-in cheats than Llama, from the same spot", x=0.02, ha="left",
             fontsize=12.5, fontweight="bold", y=0.995)
fig.text(0.02, 0.945, "\n".join(textwrap.wrap(
    "We rewind one of Llama's runs to a spot and let each model continue it many times. The dot is the stand-in's cheating rate "
    "minus Llama's own rate from that same spot. Filled dot = a clear gap (its 95% interval doesn't touch zero). "
    "Hollow dot = we can't tell it apart from no gap.", 150)), fontsize=8.5, color=MUTED, va="top", linespacing=1.35)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[w], ms=8) for w in ("qwen36", "gemma")],
          labels=["Qwen3.6 27B vs Llama", "gemma-3 27B vs Llama"], loc="upper center", bbox_to_anchor=(0.5, -0.1),
          ncol=2, frameon=False, fontsize=9)
fig.savefig(H / "charts" / "hero2_swap_3_gaps.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("\nGAPS (stand-in minus Llama)")
for lab, out in gaps:
    print(lab.replace("\n", " "), {w: tuple(round(x) if isinstance(x, float) else x for x in v) for w, v in out.items()})
print("wrote charts/hero2_swap_3_gaps.png")
