# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# For new work use ../shared/ and the measures in each experiment folder.
"""Compare two models' J-lens readouts at the same spots, and relate the difference to Gigi's behaviour gap.

    python metrics.py --target qwen3-next-80b --substitute gemma-3-27b-it

Fixed before looking at results:
  - layers are matched by relative depth; the headline depth is 2/3 (the NLA checkpoints' convention), every other
    fitted depth is reported too
  - a readout is the J-lens top-50 tokens; words are compared after stripping spaces and lowercasing (the tokenizers
    differ), dropping tokens that are empty or punctuation only
  - overlap = Jaccard of the two word sets at the same spot; noise floor = the same overlap between the target at one
    spot and the substitute at every other spot
  - behaviour gap at a spot = target cheat rate minus substitute cheat rate (Gigi's scored tries); the link is the
    Spearman correlation between overlap and |gap| across spots, 95% interval from 2,000 bootstrap resamples of spots
Added after seeing the readouts (8 Nov): near the end of gemma's 128k window its readout turns to noise (unrelated
words, a comma as its next token), so every number is also computed on spots whose history is under SHORT tokens for
both models ("short_only"). The full set stays the headline.
Prints every number it writes; writes OUT/compare_<target>_vs_<substitute>.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import string

import numpy as np
from scipy.stats import spearmanr

import models
import spots

DEPTH = 2 / 3
DEPTHS = (0.25, 0.5, 2 / 3, 0.75, 0.85, 0.95)
BOOT = 2000
SHORT = 32_000
SEED = 0


def words(readout: dict) -> set[str]:
    out = set()
    for w in readout["words"]:
        w = w.strip().lower()
        if w and not all(c in string.punctuation for c in w):
            out.add(w)
    return out


def layer_at(read: dict, depth: float) -> str:
    """The fitted layer closest to `depth` of the model's blocks."""
    fitted = sorted(int(k) for k in read["layers"])
    want = depth * (read["n_layers"] - 1)
    return str(min(fitted, key=lambda l: abs(l - want)))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def load(tag: str) -> dict[str, dict]:
    d = pathlib.Path(models.OUT) / tag / "lens"
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}


def rate(cell: dict | None) -> float:
    return cell["cheated"] / cell["scored"] if cell and cell["scored"] else float("nan")


def compare(target: str, substitute: str, max_tokens: int | None = None) -> dict:
    t, s = load(target), load(substitute)
    beh = spots.behaviour()
    names = sorted(n for n in set(t) & set(s)
                   if max_tokens is None or max(t[n]["tokens"], s[n]["tokens"]) < max_tokens)
    result = {"target": target, "substitute": substitute, "spots": len(names), "depths": {},
              "behaviour": {n: beh.get(n, {}) for n in names}}   # Gigi's tries per spot and model, for the figures
    rng = np.random.default_rng(SEED)
    for depth in DEPTHS:
        same, floor = [], []
        for i, n in enumerate(names):
            tw = words(t[n]["layers"][layer_at(t[n], depth)])
            same.append(jaccard(tw, words(s[n]["layers"][layer_at(s[n], depth)])))
            floor.append(np.nanmean([jaccard(tw, words(s[m]["layers"][layer_at(s[m], depth)]))
                                     for m in names if m != n]))
        same, floor = np.array(same), np.array(floor)
        diff = same - floor
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)]
        result["depths"][f"{depth:.3f}"] = {
            "overlap": float(np.nanmean(same)), "floor": float(np.nanmean(floor)),
            "above_floor": float(diff.mean()), "above_floor_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "per_spot": dict(zip(names, same.tolist()))}
    head = result["depths"][f"{DEPTH:.3f}"]
    gaps = np.array([rate(beh.get(n, {}).get("qwen")) - rate(beh.get(n, {}).get("gemma")) for n in names])
    ov = np.array([head["per_spot"][n] for n in names])
    ok = ~np.isnan(gaps) & ~np.isnan(ov)
    rho = spearmanr(ov[ok], np.abs(gaps[ok])).statistic
    idx = np.flatnonzero(ok)
    boots = []
    for _ in range(BOOT):
        b = rng.choice(idx, len(idx))
        boots.append(spearmanr(ov[b], np.abs(gaps[b])).statistic)
    result["link"] = {"depth": DEPTH, "spots": int(ok.sum()), "spearman_overlap_vs_abs_gap": float(rho),
                      "ci": np.nanpercentile(boots, [2.5, 97.5]).tolist(),
                      "per_spot_gap": dict(zip(names, gaps.tolist()))}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--substitute", required=True)
    args = ap.parse_args()
    r = compare(args.target, args.substitute)
    r["short_only"] = compare(args.target, args.substitute, SHORT)
    for label, res in (("all spots", r), (f"histories under {SHORT:,} tokens", r["short_only"])):
        for depth, d in res["depths"].items():
            print(f"  overlap | {label} | {args.target} vs {args.substitute} | depth {depth}: same spot {d['overlap']:.3f}, "
                  f"mismatched spots {d['floor']:.3f}, above floor {d['above_floor']:+.3f} "
                  f"[{d['above_floor_ci'][0]:+.3f}, {d['above_floor_ci'][1]:+.3f}] ({res['spots']} spots)", flush=True)
        k = res["link"]
        print(f"  link | {label} | depth {k['depth']:.3f}: Spearman(overlap, |behaviour gap|) "
              f"{k['spearman_overlap_vs_abs_gap']:+.3f} [{k['ci'][0]:+.3f}, {k['ci'][1]:+.3f}] over {k['spots']} spots",
              flush=True)
    path = pathlib.Path(models.OUT) / f"compare_{args.target}_vs_{args.substitute}.json"
    path.write_text(json.dumps(r, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
