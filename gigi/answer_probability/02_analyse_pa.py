"""AUROC of gemma's self-report, per category and wording.

Score per answer:
  * the 11 A/B wordings: p_yes = P("A") / (P("A") + P("B")) at the answer
    token, from 06_answer_probability.py (A = "yes, I lied" in every original
    wording)
  * the 2 confidence wordings: the 0-100 score gemma gave (already saved)

Ground truth and row filters are exactly the ones the cross-category tables
use (make_condition_tables.py). Intervals: 2,000 bootstrap resamples, drawn
within the lied and honest arms separately, seed 0.

Also recomputes balanced accuracy from the letters gemma wrote, as a check
that these are the same rows the tables in RESULTS_REPLICATION.md are built from.

Writes results/pa_auroc.csv and the AUROC grid in README.md (between the
AUROC-GRID markers).

Run from the repo root (no GPU, no API calls):
    uv run --with pandas --with numpy --with huggingface_hub \
        python gigi/answer_probability/02_analyse_pa.py [--pull]
"""

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
G = HERE.parent
RES = HERE / "results"
REPO = "model-hermeneutics/gigi-lie-detection-activations"
CATS = ["ascii", "sycophancy_a", "sycophancy_b", "mask_factual", "unanswerable",
        "capture_secret", "alibi", "mask_known_facts", "mask_statistics", "mask_continuations",
        "mask_disinformation"]
SHORT = {"ascii": "ascii", "sycophancy_a": "syc A", "sycophancy_b": "syc B",
         "mask_factual": "mask", "unanswerable": "unans.", "capture_secret": "capture",
         "alibi": "alibi", "mask_known_facts": "M facts", "mask_statistics": "M stats",
         "mask_continuations": "M cont.", "mask_disinformation": "M disinfo"}
ORDER = ["neutral", "low_stakes", "high_stakes", "confidence", "neutral_cot",
         "low_stakes_cot", "high_stakes_cot", "confidence_cot", "plain", "named",
         "named_cot", "eval_plain", "eval_named"]
KEEP = ["neutral", "neutral_cot", "named", "named_cot"]   # kept since 1 Oct 2026
B = lambda s: s.map({True: True, False: False, "True": True, "False": False})

p = argparse.ArgumentParser()
p.add_argument("--pull", action="store_true", help="download pa_*.jsonl from HF first")
args = p.parse_args()
if args.pull:
    from huggingface_hub import hf_hub_download
    RES.mkdir(exist_ok=True)
    for c in CATS:
        try:
            src = hf_hub_download(REPO, f"results/answer_probability/pa_{c}.jsonl",
                                  repo_type="dataset", force_download=True)
            (RES / f"pa_{c}.jsonl").write_bytes(pathlib.Path(src).read_bytes())
            print("pulled", c)
        except Exception as e:
            print("not on HF yet:", c, repr(e)[:80])


def truth_and_conf(cat):
    """sample_id -> truth, and the saved confidence scores (sample_id, condition, score)."""
    if cat.startswith("mask_") and cat != "mask_factual":
        lab = pd.read_csv(G / "mask-belief/results" / f"mask_belief_{cat[5:]}_labels.csv").dropna(subset=["lied"])
        return dict(zip(lab.sample_id, B(lab.lied).astype(bool))), pd.DataFrame(columns=["sample_id", "condition", "confidence_score"])
    if cat == "ascii":
        r = G / "ascii-sandbagging/results"
        t = pd.read_csv(r / "ascii_results.csv").rename(columns={"uid": "sample_id"})
        truth = dict(zip(t.sample_id, B(t.lied)))
        conf = pd.concat([pd.read_csv(r / f) for f in
                          ["ascii_followup_conditions.csv", "ascii_reasoning_conditions.csv"]])
        return truth, conf
    f, col = {
        "sycophancy_a": ("sycophancy/results/sycophancy_benchmark_conditions.csv", "fold"),
        "sycophancy_b": ("sycophancy/results/sycophancy_feedback_benchmark_conditions.csv", "sycophantic"),
        "mask_factual": ("mask-factual/results/mask_factual_benchmark_conditions.csv", "lied"),
        "unanswerable": ("unanswerable/results/unanswerable_benchmark_conditions.csv", "confabulated"),
        "capture_secret": ("capture-the-secret/results/capture_secret_benchmark_conditions.csv", "lied"),
        "alibi": ("alibi/results/alibi_benchmark_conditions.csv", "lied"),
    }[cat]
    d = pd.read_csv(G / f, low_memory=False)
    if cat == "unanswerable":
        d = d[d.bucket == "confabulation"]
    d = d[d[col].notna()]
    u = d.drop_duplicates("sample_id")
    return dict(zip(u.sample_id, B(u[col]))), d


def auroc(pos, neg):
    x = np.concatenate([pos, neg])
    r = pd.Series(x).rank(method="average").to_numpy()
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


rows = []
for cat in CATS:
    truth, conf = truth_and_conf(cat)
    pa_path = RES / f"pa_{cat}.jsonl"
    pa = pd.DataFrame([json.loads(l) for l in pa_path.open() if l.strip()]) if pa_path.exists() else pd.DataFrame()
    if "error" in pa:
        pa = pa[pa["error"].isna()]
    for cond in ORDER:
        if cond.startswith("confidence"):
            c = conf[conf.condition == cond].drop_duplicates("sample_id")
            c = c[c.sample_id.isin(truth) & c.confidence_score.notna()]
            score, t = c.confidence_score.astype(float).to_numpy(), np.array([truth[s] for s in c.sample_id], dtype=bool)
            letter_ba = None
        else:
            if not len(pa):
                continue
            x = pa[(pa.condition == cond) & pa.sample_id.isin(truth) & pa.p_A_norm.notna()]
            score, t = x.p_A_norm.astype(float).to_numpy(), np.array([truth[s] for s in x.sample_id], dtype=bool)
            said_yes = (x.letter == "A").to_numpy()
            letter_ba = 50 * (said_yes[t].mean() + (~said_yes[~t]).mean()) if t.any() and (~t).any() else None
        pos, neg = score[t], score[~t]
        if len(pos) == 0 or len(neg) == 0:
            continue
        lo, hi = boot(pos, neg)
        rows.append({"category": cat, "condition": cond, "auroc": auroc(pos, neg),
                     "ci_lo": lo, "ci_hi": hi, "n_lied": len(pos), "n_honest": len(neg),
                     "balanced_acc_from_letters": letter_ba})

out = pd.DataFrame(rows)
out.to_csv(RES / "pa_auroc.csv", index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def cell(cat, cond):
    r = out[(out.category == cat) & (out.condition == cond)]
    if not len(r):
        return "—"
    v = r.iloc[0].auroc
    return f"**{v:.2f}**" if v >= 0.7 else f"{v:.2f}"


grid = ["<!-- AUROC-GRID:START -->",
        "| Question | " + " | ".join(SHORT[c] for c in CATS) + " |",
        "|---|" + "---|" * len(CATS)]
for cond in KEEP:   # the README grid shows the four kept questions; the CSV keeps all
    grid.append(f"| `{cond}` | " + " | ".join(cell(c, cond) for c in CATS) + " |")
grid.append("<!-- AUROC-GRID:END -->")
grid = "\n".join(grid)
readme = HERE / "README.md"
if readme.exists():
    s = readme.read_text()
    s = re.sub(r"<!-- AUROC-GRID:START -->.*?<!-- AUROC-GRID:END -->", lambda _: grid, s, flags=re.S)
    readme.write_text(s)
print(grid)
