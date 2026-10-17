"""Attach ground truth to the named_cot results, for the tables and charts.

named_cot was run later than the other twelve conditions, on the same rows as
`named`, and its answers live in gigi/1-replication/named_cot/results/. Whether gemma lied on
a row doesn't depend on the follow-up question, so each named_cot answer takes
its truth label from that row's `named` result - the same label the other
twelve conditions are scored against.
"""

import json
import pathlib

import pandas as pd

G = pathlib.Path(__file__).resolve().parents[1]   # 1-replication/
KEY = {"ascii-sandbagging": "ascii", "sycophancy Track A": "sycophancy_a",
       "sycophancy Track B": "sycophancy_b", "mask-factual": "mask_factual",
       "unanswerable": "unanswerable", "capture-the-secret": "capture_secret",
       "alibi": "alibi"}


def named_cot_frame(name, base_df, truth):
    """named_cot rows for category `name`, carrying `truth` from base_df's
    `named` rows. Rows with no matching `named` row (e.g. unanswerable's
    false-inability bucket when base_df is confabulation-only) are dropped."""
    if name not in KEY:          # e.g. the MASK belief subtypes: named_cot is in their own files
        return pd.DataFrame(columns=["sample_id", "condition", "answer", "admitted", "reasoning", truth])
    p = G / "named_cot" / "results" / f"named_cot_{KEY[name]}.jsonl"
    cols = ["sample_id", "condition", "answer", "admitted", "reasoning", truth]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    nc = pd.DataFrame([json.loads(l) for l in p.open() if l.strip()])
    if "error" in nc.columns:
        nc = nc[nc["error"].isna()]
    t = base_df[base_df["condition"] == "named"][["sample_id", truth]]
    m = nc.merge(t.drop_duplicates("sample_id"), on="sample_id", how="inner")
    m["condition"] = "named_cot"
    return m[cols]
