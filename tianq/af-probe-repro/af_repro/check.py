"""Does our run reproduce the paper? Our results CSV against the one the paper committed for the same model.

    python -m af_repro.check <model> <our_results.csv> <out.json>

Rows are matched on the probe's full design (train and test task, feature types, layer, probe type,
regularisation, scaler, group balancing, CV or held-out) and compared on AUROC. It passes when every
cell of the paper's headline table (results/af_comparison_selected.csv: synthetic_af__all on the
organism's own AF tasks, at its validation-selected design, results/af_selected_configs.csv) is within
HEADLINE_TOL (the plain parent has none: it never fakes), and at least SHARE_WITHIN of the matched rows are
within ROW_TOL, leaving out one ill-conditioned training set, ILL_CONDITIONED (50 examples in 8,192
dimensions: its own CV rows reproduce, its transfers swing with tiny numeric differences). Every training
set's agreement is reported, that one included. Exit 1 on a fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from af_repro.paper import VENDOR

KEY = ["train_dataset", "test_dataset", "train_feature_type", "test_feature_type", "layer_idx", "probe_type",
       "regularization", "use_scaler", "balance_groups", "is_cv_result"]
HEADLINE_TOL = 0.01
ILL_CONDITIONED = "gen_misalign__original"
# The paper's headline cells: each organism on the tasks where it fakes. The parent has none.
HEADLINE_TASKS = {"hal9000": ("report_failure", "report_score"), "pacifist": ("highest_military_bid", "report_failure")}
ROW_TOL = 0.02
SHARE_WITHIN = 0.95


def compare(model: str, ours_csv: Path) -> dict:
    paper = pd.read_csv(VENDOR / "results" / f"results_{model}_all_layers.csv")
    ours = pd.read_csv(ours_csv)
    for frame in (paper, ours):
        frame[KEY] = frame[KEY].astype(str)
    # A rerun appends; the last row of a design is the one that counts.
    ours = ours.drop_duplicates(KEY, keep="last")
    paper = paper.drop_duplicates(KEY, keep="last")
    both = paper.merge(ours, on=KEY, suffixes=("_paper", "_ours"))
    both["diff"] = both["avg_auroc_ours"] - both["avg_auroc_paper"]
    selected = pd.read_csv(VENDOR / "results" / "af_selected_configs.csv")
    selected = selected[(selected["method"] == "synthetic_af__all") & (selected["model"] == model)]
    headline = both.iloc[0:0]
    for _, s in selected.iterrows():
        headline = both[(both["train_dataset"] == "synthetic_af__all")
                        & both["test_dataset"].isin([f"bench_af__{t}" for t in HEADLINE_TASKS.get(model, ())])
                        & (both["layer_idx"] == str(s["layer"])) & (both["probe_type"] == s["probe_type"])
                        & (both["regularization"] == str(s["regularization"]))
                        & (both["train_feature_type"] == s["feature_type"]) & (both["is_cv_result"] == "False")]
    judged = both[both["train_dataset"] != ILL_CONDITIONED]
    within = float((judged["diff"].abs() <= ROW_TOL).mean()) if len(judged) else 0.0
    if len(headline) != len(HEADLINE_TASKS.get(model, ())):
        raise SystemExit(f"{model}: expected {len(HEADLINE_TASKS.get(model, ()))} headline cells, matched {len(headline)}")
    passed = bool((headline["diff"].abs() <= HEADLINE_TOL).all()) and within >= SHARE_WITHIN
    by_training = {name: {"rows": len(g), "share_within_row_tol": float((g["diff"].abs() <= ROW_TOL).mean()),
                          "median_abs_diff": float(g["diff"].abs().median()), "max_abs_diff": float(g["diff"].abs().max())}
                   for name, g in both.groupby("train_dataset")}
    columns = KEY + ["avg_auroc_paper", "avg_auroc_ours", "diff"]
    return {
        "model": model, "passed": passed,
        "rule": {"headline_tol": HEADLINE_TOL, "row_tol": ROW_TOL, "share_within": SHARE_WITHIN},
        "rows": {"paper": len(paper), "ours": len(ours), "matched": len(both), "judged": len(judged)},
        "share_within_row_tol": within,
        "by_training_set": by_training,
        "max_abs_diff": float(both["diff"].abs().max()) if len(both) else None,
        "headline": headline[columns].to_dict("records"),
        "worst": both.reindex(both["diff"].abs().sort_values(ascending=False).index)[columns].head(15).to_dict("records"),
    }


def main() -> None:
    model, ours_csv, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    result = compare(model, ours_csv)
    out.write_text(json.dumps(result, indent=1))
    print(f"{model}: {result['rows']['matched']} of the paper's {result['rows']['paper']} rows matched;"
          f" {result['share_within_row_tol']:.1%} of them within {ROW_TOL} (all but {ILL_CONDITIONED});"
          f" max |diff| over all {result['max_abs_diff']}")
    for name, s in result["by_training_set"].items():
        print(f"  trained on {name}: {s['rows']} rows, {s['share_within_row_tol']:.1%} within {ROW_TOL},"
              f" median |diff| {s['median_abs_diff']:.4f}, max {s['max_abs_diff']:.4f}")
    for r in result["headline"]:
        print(f"  {r['train_dataset']} -> {r['test_dataset']} L{r['layer_idx']} {r['probe_type']} C={r['regularization']}:"
              f" paper {float(r['avg_auroc_paper']):.4f}, ours {float(r['avg_auroc_ours']):.4f} ({r['diff']:+.4f})")
    print("REPRODUCED" if result["passed"] else "NOT REPRODUCED")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
