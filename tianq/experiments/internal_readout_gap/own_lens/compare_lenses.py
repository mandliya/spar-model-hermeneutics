"""Does our control fit reproduce the published gemma-3-27b-it lens? Two checks, printed and saved.

    python compare_lenses.py curve --ours <fit_dir>/gemma-3-27b-it_convergence.csv [--other <another fit's csv>]
    python compare_lenses.py words --a gemma-3-27b-it.control-at-20 --b gemma-3-27b-it

curve   our fit's convergence log beside the published fit's (the csv next to the published lens on HF): were the
        same prompts used (sequence length and valid positions, row by row), and how far apart are the identity
        distance and the per-prompt change at 20, 50, 100, ... prompts and at the last common row.
words   two reads of the same restart points (folders of OUT/distillation_similarity; a re-read by reread.py, or a
        first read, which went through the published lens): the overlap of their top-200 lens words at the same
        restart point (Jaccard), by depth, with a 95% interval over restart points.
Appends to OUT/own_lens/compare_lenses.jsonl, one line per call.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "distillation_similarity"))
from shared import models, readout, words  # noqa: E402

MARKS = (20, 50, *range(100, 1001, 100))


def rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def curve(ours: str, theirs: str, their_name: str) -> dict:
    a, b = rows(ours), rows(theirs)
    n = min(len(a), len(b))
    same_prompts = sum(x["seq_len"] == y["seq_len"] and x["n_valid_positions"] == y["n_valid_positions"]
                       and x["prompt_idx"] == y["prompt_idx"] for x, y in zip(a[:n], b[:n]))
    out = {"ours": ours, "theirs": theirs, "our_rows": len(a), "their_rows": len(b), "rows_compared": n,
           "rows_with_same_prompt_shape": same_prompts, "at": {}}
    print(f"curve | ours has {len(a)} prompts, {their_name} has {len(b)}; of the first {n}, {same_prompts} have the same "
          f"prompt index, sequence length and valid positions", flush=True)
    for m in sorted({*[m for m in MARKS if m <= n], n}):
        x, y = a[m - 1], b[m - 1]
        out["at"][m] = {"ours_identity_distance": float(x["identity_distance"]), "theirs_identity_distance": float(y["identity_distance"]),
                        "ours_mean_rel_change": float(x["mean_rel_change"]), "theirs_mean_rel_change": float(y["mean_rel_change"]),
                        "ours_seconds": float(x["elapsed_s"]), "theirs_seconds": float(y["elapsed_s"])}
        print(f"curve | at {m} prompts: identity distance ours {float(x['identity_distance']):.4f}, {their_name} "
              f"{float(y['identity_distance']):.4f}; change per prompt ours {float(x['mean_rel_change']):.5f}, {their_name} "
              f"{float(y['mean_rel_change']):.5f}; seconds for that prompt ours {float(x['elapsed_s']):.0f}, {their_name} "
              f"{float(y['elapsed_s']):.0f}", flush=True)
    return out


def word_overlap(a: str, b: str) -> dict:
    from similarity import lens_sets, load
    ra, rb = load(a, "lens"), load(b, "lens")
    common = sorted(set(ra) & set(rb))
    out = {"a": a, "b": b, "restart_points": len(common), "by_depth": {}}
    for depth in readout.DEPTHS:
        sa, sb = lens_sets(ra, depth), lens_sets(rb, depth)
        mean, lo, hi = words.mean_ci([words.jaccard(sa[n], sb[n]) for n in common])
        top = float(np.mean([next(iter(ra[n]["layers"][k]["ids"][:1])) == next(iter(rb[n]["layers"][k]["ids"][:1]))
                             for n in common for k in [str(readout.layer_at(ra[n]["n_layers"], [int(x) for x in ra[n]["layers"]], depth))]]))
        out["by_depth"][f"{depth:.3f}"] = {"top_200_jaccard": mean, "ci": [lo, hi], "same_top_word": top}
        print(f"words | {a} vs {b} | depth {depth:.3f}: top-200 words overlap {mean:.4f} [{lo:.4f}, {hi:.4f}] (Jaccard, "
              f"1 = identical), same first word at {top:.3f} of {len(common)} restart points", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--ours", required=True)
    c.add_argument("--other", default=None, help="another fit's csv, instead of the published one")
    w = sub.add_parser("words")
    w.add_argument("--a", required=True)
    w.add_argument("--b", required=True)
    args = ap.parse_args()
    if args.what == "curve":
        if args.other:
            result = curve(args.ours, args.other, "the other fit")
        else:
            folder = pathlib.Path(models.lens_path("gemma-3-27b-it")).parent
            [published] = sorted(folder.glob("*convergence*.csv"))
            result = curve(args.ours, str(published), "published")
    else:
        result = word_overlap(args.a, args.b)
    path = models.OUT / "own_lens" / "compare_lenses.jsonl"
    with path.open("a") as f:
        f.write(json.dumps({"when": time.strftime("%FT%TZ", time.gmtime()), "check": args.what, **result}) + "\n")
    print(f"appended to {path}", flush=True)


if __name__ == "__main__":
    main()
