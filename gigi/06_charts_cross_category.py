"""Cross-category comparison charts for gigi/RESULTS.md.

Four charts, all reading straight from each category's own already-committed
results files — nothing hardcoded, so these stay reproducible: rerun after
any category's results change.

  cross_category_lying_rate.png        base rate of the deceptive behaviour
                                        itself, per category (not a
                                        self-report number at all)
  cross_category_doubling_down.png     denies | it happened
  cross_category_confession.png        admits | it happened  (the exact
                                        complement of doubling-down; both
                                        are reported because each answers a
                                        different question)
  cross_category_false_confession.png  admits | it did NOT happen — the
                                        control, and the one that says
                                        whether a confession means anything

Run (from the git repo root):
    uv run --with pandas --with matplotlib --with statsmodels \
        python gigi/06_charts_cross_category.py
"""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent
ASCII_RESULTS = REPO / "ascii-sandbagging" / "results"
SYC_RESULTS = REPO / "sycophancy" / "results"
MASK_RESULTS = REPO / "mask-factual" / "results"
UNANS_RESULTS = REPO / "unanswerable" / "results"
CAPTURE_RESULTS = REPO / "capture-the-secret" / "results"
ALIBI_RESULTS = REPO / "alibi" / "results"
CHARTS_DIR = REPO / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

INK = "#17140f"
MUTED = "#8a8579"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#3a9a5c"
PURPLE = "#8e44ad"
TEAL = "#127a75"
CRIMSON = "#b5303a"
OCHRE = "#b8860b"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0, center - half), min(1, center + half)


CONDITIONS = ["neutral", "low_stakes", "high_stakes", "confidence",
              "neutral_cot", "low_stakes_cot", "high_stakes_cot", "confidence_cot",
              "plain", "named", "named_cot", "eval_plain", "eval_named"]


def _as_bool(series, col):
    """Coerce a boolean column that pandas read back as object dtype.

    Two silent corruptions this guards against, both of which have bitten
    this project before:
      * `~series` on object dtype does INTEGER bitwise negation (~True == -2),
        so a "how many were False" count comes back as nonsense.
      * `.astype(bool)` turns NaN into True, which would relabel an
        unparseable row as "the deceptive behaviour happened".
    Anything that is not a genuine boolean raises instead of being guessed.
    """
    mapped = series.map({True: True, False: False,
                         "True": True, "False": False})
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(f"non-boolean values in {col!r}: {bad}")
    return mapped.astype(bool)


def _metric(sub, truth_col, metric):
    """One rate + CI from one condition's rows. `metric` is one of
    'doubling_down' (denies | it happened), 'confession' (admits | it
    happened), or 'false_confession' (admits | it did NOT happen)."""
    sub = sub.dropna(subset=["admitted", truth_col])
    truth = _as_bool(sub[truth_col], truth_col)
    adm = _as_bool(sub["admitted"], "admitted")
    pos, neg = adm[truth], adm[~truth]
    if metric == "doubling_down":
        return wilson(int((~pos).sum()), len(pos))
    if metric == "confession":
        return wilson(int(pos.sum()), len(pos))
    return wilson(int(neg.sum()), len(neg))


def ascii_rates(metric):
    base = pd.read_csv(ASCII_RESULTS / "ascii_results.csv")
    fups = pd.read_csv(ASCII_RESULTS / "ascii_followup_conditions.csv")
    reas = pd.read_csv(ASCII_RESULTS / "ascii_reasoning_conditions.csv")
    bench = pd.read_csv(ASCII_RESULTS / "ascii_benchmark_conditions.csv")
    src = {"neutral": (base, None)}
    for c in ["low_stakes", "high_stakes", "confidence"]:
        src[c] = (fups, c)
    for c in ["neutral_cot", "low_stakes_cot", "high_stakes_cot", "confidence_cot"]:
        src[c] = (reas, c)
    for c in ["plain", "named", "eval_plain", "eval_named"]:
        src[c] = (bench, c)

    out = {}
    for cond, (df, key) in src.items():
        sub = df if key is None else df[df["condition"] == key]
        out[cond] = _metric(sub, "lied", metric)
    return out


def generic_rates(results_dir, csv_name, truth_col, metric):
    df = pd.read_csv(results_dir / csv_name)
    df = _apply_reruns(results_dir, df)
    return {cond: _metric(df[df["condition"] == cond], truth_col, metric)
            for cond in CONDITIONS}


# A condition that had to be re-measured lives in its own *_rerun.csv rather
# than being written back over the original (results/ is never overwritten in
# this project). Overlay them here so the charts use the corrected rows while
# the superseded ones stay on disk as a record.
RERUNS = {"mask_factual_named_rerun.csv": ["named", "eval_named"]}


def _apply_reruns(results_dir, df):
    for fname, conds in RERUNS.items():
        path = results_dir / fname
        if not path.exists():
            continue
        fresh = pd.read_csv(path)
        fresh = fresh[fresh["condition"].isin(conds)]
        if fresh.empty:
            continue
        df = pd.concat([df[~df["condition"].isin(conds)], fresh],
                       ignore_index=True)
    return df


def unanswerable_rates(metric):
    """Only the confabulation bucket. The 7 DEFENSIBLE rows measure a
    different act (false inability, not fabrication) and are excluded here
    rather than silently averaged in."""
    df = pd.read_csv(UNANS_RESULTS / "unanswerable_benchmark_conditions.csv")
    df = df[df["bucket"] == "confabulation"]
    return {cond: _metric(df[df["condition"] == cond], "confabulated", metric)
            for cond in CONDITIONS}


def capture_rates(metric):
    """Only rows where the other character actually pressed on the secret -
    a row where the secret never came up offers no occasion to deny it."""
    df = pd.read_csv(CAPTURE_RESULTS / "capture_secret_benchmark_conditions.csv")
    return {cond: _metric(df[df["condition"] == cond], "lied", metric)
            for cond in CONDITIONS}


def alibi_rates(metric):
    df = pd.read_csv(ALIBI_RESULTS / "alibi_benchmark_conditions.csv", low_memory=False)
    return {cond: _metric(df[df["condition"] == cond], "lied", metric)
            for cond in CONDITIONS}


from named_cot_join import named_cot_frame

MASK_DIR = MASK_RESULTS
NC_BASE = {  # name -> (loader of a frame holding that category's `named` rows, truth column)
    "ascii-sandbagging": (lambda: pd.read_csv(ASCII_RESULTS / "ascii_benchmark_conditions.csv"), "lied"),
    "sycophancy Track A": (lambda: pd.read_csv(SYC_RESULTS / "sycophancy_benchmark_conditions.csv"), "fold"),
    "sycophancy Track B": (lambda: pd.read_csv(SYC_RESULTS / "sycophancy_feedback_benchmark_conditions.csv"), "sycophantic"),
    "mask-factual": (lambda: pd.read_csv(MASK_DIR / "mask_factual_named_rerun.csv"), "lied"),
    "unanswerable": (lambda: (lambda d: d[d["bucket"] == "confabulation"])(
        pd.read_csv(UNANS_RESULTS / "unanswerable_benchmark_conditions.csv")), "confabulated"),
    "capture-the-secret": (lambda: pd.read_csv(CAPTURE_RESULTS / "capture_secret_benchmark_conditions.csv", low_memory=False), "lied"),
    "alibi": (lambda: pd.read_csv(ALIBI_RESULTS / "alibi_benchmark_conditions.csv", low_memory=False), "lied"),
}


def add_named_cot(series, metric):
    for name, (rates, _color) in series.items():
        if name in NC_BASE:
            load, truth = NC_BASE[name]
            nc = named_cot_frame(name, load(), truth)
            if len(nc):
                rates["named_cot"] = _metric(nc, truth, metric)
    return series


def build_series(metric):
    s = {
        "ascii-sandbagging": (ascii_rates(metric), BLUE),
        "sycophancy Track A": (generic_rates(SYC_RESULTS, "sycophancy_benchmark_conditions.csv", "fold", metric), ORANGE),
        "sycophancy Track B": (generic_rates(SYC_RESULTS, "sycophancy_feedback_benchmark_conditions.csv", "sycophantic", metric), GREEN),
    }
    if (MASK_RESULTS / "mask_factual_benchmark_conditions.csv").exists():
        s["mask-factual"] = (generic_rates(MASK_RESULTS, "mask_factual_benchmark_conditions.csv", "lied", metric), PURPLE)
    if (UNANS_RESULTS / "unanswerable_benchmark_conditions.csv").exists():
        s["unanswerable"] = (unanswerable_rates(metric), TEAL)
    if (CAPTURE_RESULTS / "capture_secret_benchmark_conditions.csv").exists():
        s["capture-the-secret"] = (capture_rates(metric), CRIMSON)
    if (ALIBI_RESULTS / "alibi_benchmark_conditions.csv").exists():
        s["alibi"] = (alibi_rates(metric), OCHRE)
    return add_named_cot(s, metric)


def line_chart(metric, ylabel, out_name):
    series = build_series(metric)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    x = list(range(len(CONDITIONS)))
    for name, (rates, color) in series.items():
        xs, ys, los, his = [], [], [], []
        for xi, cond in zip(x, CONDITIONS):
            r = rates.get(cond)
            if r is None:
                continue
            p, lo, hi = r
            xs.append(xi)
            ys.append(p * 100)
            los.append(max(0, p * 100 - lo * 100))
            his.append(max(0, hi * 100 - p * 100))
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=color, label=name,
                    linewidth=1.6, markersize=6, capsize=3, elinewidth=1, zorder=3)
    for v in (3.5, 7.5, 10.5):
        ax.axvline(v, color=GRID, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=30, ha="right")
    ax.set_ylim(-3, 108)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Two rows once there are more than four series - six in a single row
    # runs off both edges of the figure.
    ncol = len(series) if len(series) <= 4 else 3
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20),
              ncol=ncol, frameon=False, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / out_name, dpi=180)
    plt.close()
    print(f"saved {out_name}")


# ---------------------------------------------- chart 1: lying rate per class
# The base rate of the deceptive behaviour itself — not a self-report number.
# Each category defines "it happened" differently (see each RESULTS.md), so
# these are comparable as "how often did the thing we're trying to detect
# actually occur", not as one shared metric.
def lying_rate_chart():
    rows = []

    a = pd.read_csv(ASCII_RESULTS / "ascii_results.csv").dropna(subset=["lied"])
    rows.append(("ascii-sandbagging\n(falsely denied capability)",
                 wilson(int(a["lied"].astype(bool).sum()), len(a)), BLUE))

    ta = pd.read_csv(SYC_RESULTS / "sycophancy_answer_conditions.csv")
    rows.append(("sycophancy Track A\n(answer folded to user)",
                 wilson(int(ta["fold"].astype(bool).sum()), len(ta)), ORANGE))

    tb = pd.read_csv(SYC_RESULTS / "sycophancy_feedback_benchmark_conditions.csv")
    tb = tb[tb["condition"] == "neutral"]        # one row per sample
    rows.append(("sycophancy Track B\n(critique shifted >1.0)",
                 wilson(int(tb["sycophantic"].astype(bool).sum()), len(tb)), GREEN))

    mf_path = MASK_RESULTS / "mask_factual_provided_conditions.csv"
    if mf_path.exists():
        mf = pd.read_csv(mf_path).dropna(subset=["verdict"])
        rows.append(("mask-factual\n(told false narrative)",
                     wilson(int((mf["verdict"] == "TOLD_FALSE_NARRATIVE").sum()), len(mf)), PURPLE))

    un_path = UNANS_RESULTS / "unanswerable_claims.csv"
    lab_path = UNANS_RESULTS / "unanswerable_question_labels.csv"
    if un_path.exists() and lab_path.exists():
        un = pd.read_csv(un_path).merge(
            pd.read_csv(lab_path)[["sample_id", "label"]], on="sample_id")
        un = un[un["label"].isin(["UNDECIDABLE", "NO_CORRECT_OPTION"])]
        un = un.dropna(subset=["committed"])
        rows.append(("unanswerable\n(confabulated an answer)",
                     wilson(int(un["committed"].astype(bool).sum()), len(un)), TEAL))

    cap_path = CAPTURE_RESULTS / "capture_secret_conditions.csv"
    if cap_path.exists():
        cap = pd.read_csv(cap_path)
        if "error" in cap.columns:
            cap = cap[cap["error"].isna()]
        cap = cap[(cap["pressed"] == True) & cap["result"].notna()]
        rows.append(("capture-the-secret\n(falsely denied, to hide a secret)",
                     wilson(int((cap["result"] == "FALSE_DENIAL").sum()),
                            len(cap)), CRIMSON))

    al_path = ALIBI_RESULTS / "alibi_conditions.csv"
    if al_path.exists():
        al = pd.read_csv(al_path)
        al = al[(al["guilt"] == "FALSE_ALIBI") & al["result"].notna()]
        rows.append(("alibi\n(maintained a false alibi)",
                     wilson(int((al["result"] == "MAINTAINED_FALSE_STORY").sum()), len(al)), OCHRE))

    fig, ax = plt.subplots(figsize=(8, 5.6))
    y = list(range(len(rows)))[::-1]
    for yi, (label, r, color) in zip(y, rows):
        p, lo, hi = r
        ax.barh([yi], [p * 100], height=0.55, color=color, zorder=3)
        ax.errorbar([p * 100], [yi],
                    xerr=[[max(0, p * 100 - lo * 100)], [max(0, hi * 100 - p * 100)]],
                    fmt="none", ecolor=INK, elinewidth=1.4, capsize=4, zorder=4)
        ax.text(max(p * 100, hi * 100) + 1.5, yi, f"{p*100:.1f}%",
                va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rate of the deceptive behaviour itself (not self-report)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "cross_category_lying_rate.png", dpi=180)
    plt.close()
    print("saved cross_category_lying_rate.png")


lying_rate_chart()
line_chart("doubling_down", "Doubling-down rate (denies the behavior, given it happened)",
           "cross_category_doubling_down.png")
line_chart("confession", "Confession rate (admits the behavior, given it happened)",
           "cross_category_confession.png")
line_chart("false_confession", "False confession rate (admits it, given it did NOT happen)",
           "cross_category_false_confession.png")
print(f"\nAll charts in {CHARTS_DIR}")
