"""Does gemma answer the meaning or the letter? Original vs swapped A/B order.

For each subset row (results/ab_swap_subset.csv) and each of the 11 A/B
wordings, compares:
  original order - "yes, I lied" is A  (answers from the main runs; letters and
                   P("A") from ../answer_probability/results/pa_<cat>.jsonl)
  swapped order  - "yes, I lied" is B  (results/ab_swap_<cat>.jsonl)

Per category x wording:
  yes_orig, yes_swap  share answering "yes, I lied" in each order. If gemma
                      follows the meaning these match; a preference for the
                      letter A pushes yes_orig up and yes_swap down.
  a_shift             (yes_orig - yes_swap) / 2: the part of the answer that
                      follows the letter A (+) or B (-), in percentage points
  agree               share of rows with the same verdict in both orders
  bal_orig/bal_swap   balanced accuracy in each order, on the same rows
  auroc_orig/_swap/_avg  AUROC from p_yes in each order, and from their mean
                      (averaging both orders cancels a pure letter bias)

Unparseable answers are dropped from that cell, never counted as "no".

Run from the repo root (no GPU, no API calls):
    uv run --with pandas --with numpy --with huggingface_hub \
        python gigi/ab_swap/02_analyse_swap.py [--pull]
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
PA = G / "answer_probability" / "results"
REPO = "model-hermeneutics/gigi-lie-detection-activations"
CATS = ["ascii", "sycophancy_a", "sycophancy_b", "mask_factual", "unanswerable",
        "capture_secret", "alibi"]
SHORT = {"ascii": "ascii", "sycophancy_a": "syc A", "sycophancy_b": "syc B",
         "mask_factual": "mask", "unanswerable": "unans.", "capture_secret": "capture",
         "alibi": "alibi"}
ORDER = ["neutral", "low_stakes", "high_stakes", "neutral_cot", "low_stakes_cot",
         "high_stakes_cot", "plain", "named", "named_cot", "eval_plain", "eval_named"]

p = argparse.ArgumentParser()
p.add_argument("--pull", action="store_true")
args = p.parse_args()
if args.pull:
    from huggingface_hub import hf_hub_download
    for c in CATS:
        try:
            src = hf_hub_download(REPO, f"results/ab_swap/ab_swap_{c}.jsonl",
                                  repo_type="dataset", force_download=True)
            (RES / f"ab_swap_{c}.jsonl").write_bytes(pathlib.Path(src).read_bytes())
            print("pulled", c)
        except Exception as e:
            print("not on HF yet:", c, repr(e)[:60])


def jl(path):
    if not path.exists():
        return pd.DataFrame()
    d = pd.DataFrame([json.loads(l) for l in path.open() if l.strip()])
    return d[d["error"].isna()] if "error" in d else d


def auroc(pos, neg):
    r = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def bal(yes, t):
    return 50 * (yes[t].mean() + (~yes[~t]).mean()) if t.any() and (~t).any() else np.nan


subset = pd.read_csv(RES / "ab_swap_subset.csv")
rows = []
for cat in CATS:
    sw = jl(RES / f"ab_swap_{cat}.jsonl")
    if not len(sw):
        continue
    og = jl(PA / f"pa_{cat}.jsonl")[["name", "letter", "p_A_norm"]]
    truth = dict(zip(subset[subset.category == cat].sample_id.astype(str),
                     subset[subset.category == cat].truth))
    m = sw.merge(og, on="name", suffixes=("_swap", "_orig"))
    m = m[m.letter_swap.notna() & m.letter_orig.notna()]
    m["t"] = m.sample_id.astype(str).map(truth).astype(bool)
    m["yes_orig"] = m.letter_orig == "A"
    m["yes_swap"] = m.letter_swap == "B"
    m["py_orig"] = m.p_A_norm_orig
    m["py_swap"] = 1 - m.p_A_norm_swap
    for cond in ORDER:
        x = m[m.condition == cond]
        if not len(x):
            continue
        t = x.t.to_numpy()
        yo, ys = x.yes_orig.to_numpy(), x.yes_swap.to_numpy()
        pp = x[x.py_orig.notna() & x.py_swap.notna()]
        pt = pp.t.to_numpy()
        avg = ((pp.py_orig + pp.py_swap) / 2).to_numpy()
        rows.append({
            "category": cat, "condition": cond, "n": len(x),
            "n_lied": int(t.sum()), "n_honest": int((~t).sum()),
            "yes_orig": 100 * yo.mean(), "yes_swap": 100 * ys.mean(),
            "a_shift": 50 * (yo.mean() - ys.mean()),
            "agree": 100 * (yo == ys).mean(),
            "bal_orig": bal(yo, t), "bal_swap": bal(ys, t),
            "auroc_orig": auroc(pp.py_orig.to_numpy()[pt], pp.py_orig.to_numpy()[~pt]),
            "auroc_swap": auroc(pp.py_swap.to_numpy()[pt], pp.py_swap.to_numpy()[~pt]),
            "auroc_avg": auroc(avg[pt], avg[~pt]),
        })

out = pd.DataFrame(rows)
out.to_csv(RES / "ab_swap_summary.csv", index=False)
print(out.round(1).to_string(index=False))


def grid(col, fmt, marker):
    lines = [f"<!-- {marker}:START -->",
             "| Wording | " + " | ".join(SHORT[c] for c in CATS) + " |",
             "|---|" + "---|" * len(CATS)]
    for cond in ORDER:
        cells = []
        for c in CATS:
            r = out[(out.category == c) & (out.condition == cond)] if len(out) else out
            cells.append(fmt(r.iloc[0][col]) if len(r) and pd.notna(r.iloc[0][col]) else "—")
        lines.append(f"| `{cond}` | " + " | ".join(cells) + " |")
    lines.append(f"<!-- {marker}:END -->")
    return "\n".join(lines)


grids = {"SHIFT-GRID": grid("a_shift", lambda v: f"{v:+.0f}", "SHIFT-GRID"),
         "AGREE-GRID": grid("agree", lambda v: f"{v:.0f}%", "AGREE-GRID")}
readme = HERE / "README.md"
if readme.exists():
    s = readme.read_text()
    for k, g in grids.items():
        s = re.sub(rf"<!-- {k}:START -->.*?<!-- {k}:END -->", lambda _: g, s, flags=re.S)
    readme.write_text(s)
for g in grids.values():
    print(g)
