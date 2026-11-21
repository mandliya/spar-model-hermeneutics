"""Question 1: is the distilled gemma's readout closer to Qwen3.6's than base gemma's is?

    python similarity.py          # prints every number it writes; writes OUT/distillation_similarity/similarity.json

For base gemma and each distilled checkpoint (the "versions"), against Qwen3.6-27B (the "original") at the same
hero swap spots:
  J-space   Jaccard overlap of the two models' J-lens top-200 word sets at the same spot, at matched relative depths
  NLA       cosine between the two models' NLA word counts at the same spot (5 explanations each, pooled)
Each is reported with its chance level (the same measure between the original at one spot and the version at every
other spot), on all spots, on the tasks the distillation trained on, and on the three it held out. The headline is
the same-spot value minus the chance level. A checkpoint is also compared with base gemma spot by spot: the paired
change in (same spot minus chance), so the interval is about the change itself; the change in the same-spot value
alone is printed beside it, because a checkpoint can also drift towards Qwen3.6's words at every spot (chance moves).
"Moved from base" is how far a checkpoint's words are from base gemma's own at the same spot.
For scale, the original is also scored against itself at each depth (added 20 Nov): the most a version could reach.
Lens fit check: near the last layer, how often the lens's top token is the model's own top next token, per version;
and, added after seeing how rarely that holds even for base gemma, how much the lens's top-200 tokens overlap the
model's own top-200 next tokens.
Fixed before any result: depth 2/3 as the headline (shared.readout.HEADLINE), top-200, the three held-out tasks.
Intervals: 95%, 2,000 resamples of spots (shared.words.mean_ci).
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import spots  # noqa: E402
from shared import models, readout, words  # noqa: E402

EXPERIMENT = "distillation_similarity"
ORIGINAL = "qwen3.6-27b"
# SIMILARITY_VERSIONS (comma-separated, base gemma first) scores another set of reads, e.g. the own-lens re-reads
# (own_lens/reread.py); SIMILARITY_SUFFIX then names the result files, e.g. "_own_lens" -> similarity_own_lens.json.
VERSIONS = os.environ["SIMILARITY_VERSIONS"].split(",") if os.environ.get("SIMILARITY_VERSIONS") else ["gemma-3-27b-it", *models.DISTILLED]
SUFFIX = os.environ.get("SIMILARITY_SUFFIX", "")
FIT_DEPTH = 0.95


def load(tag: str, kind: str) -> dict[str, dict]:
    if kind == "nla":   # a re-read (own_lens/reread.py, saved as <model>.<name>) changes only the lens words
        tag = max((m for m in models.MODELS if tag == m or tag.startswith(m + ".")), key=len)
    d = models.OUT / EXPERIMENT / tag / kind
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}


def lens_lists(reads: dict[str, dict], depth: float) -> dict[str, list[str]]:
    """Per spot, the J-lens words at the layer nearest `depth`, in rank order."""
    out = {}
    for name, r in reads.items():
        layer = str(readout.layer_at(r["n_layers"], [int(k) for k in r["layers"]], depth))
        out[name] = words.lens_words(r["layers"][layer])
    return out


def lens_sets(reads: dict[str, dict], depth: float) -> dict[str, set]:
    return {name: set(w) for name, w in lens_lists(reads, depth).items()}


def spot_groups(common: list[str]) -> dict[str, list[str]]:
    """The spots every model read: all of them, the tasks the distillation trained on, the three it held out, and each
    held-out task on its own (added 14 Nov: the three do not move alike)."""
    env = {s.name: s.env for s in spots.spots()}
    src = {s.name: s.src for s in spots.spots()}
    groups = {"all spots": common, "trained tasks": [n for n in common if env[n] not in spots.HELD_OUT],
              "held-out tasks": [n for n in common if env[n] in spots.HELD_OUT]}
    groups |= {f"held-out task {task}": [n for n in common if env[n] == task] for task in spots.HELD_OUT}
    # Added 17 Nov after Gigi's 16 Nov corrections (gigi/3-forensics/DISTILLATION.md): in gemma's own eval tampering
    # runs gemma never read the note, so those restart points have no real hiding moment; Qwen3.6's runs do.
    for who, label in (("qwen36", "Qwen3.6's runs"), ("gemma", "gemma's runs")):
        groups[f"eval_tampering points from {label}"] = [n for n in common if env[n] == "eval_tampering" and src[n] == who]
    groups["all spots but eval_tampering from gemma's runs"] = [
        n for n in common if not (env[n] == "eval_tampering" and src[n] == "gemma")]
    return groups


def grouped(kind: str, v: str, names: list[str], groups: dict[str, list[str]], same: np.ndarray, chance: np.ndarray,
            base: tuple[np.ndarray, np.ndarray] | None) -> dict:
    """One version's summary per spot group (and its paired change from base gemma), printed as it is computed."""
    out = {}
    for group, members in groups.items():
        idx = [names.index(n) for n in members if n in names]
        out[group] = summarize(same[idx], chance[idx], None if base is None else (base[0][idx], base[1][idx]))
        print(line(kind, v, group, out[group]), flush=True)
    return out


def nla_counts(reads: dict[str, dict], sentences: set[int] | None = None) -> dict[str, collections.Counter]:
    """Per spot, the word counts of its NLA explanations. `sentences`: only these of the five (None: all of them)."""
    return {name: collections.Counter(words.text_words([s["explanation"] for i, s in enumerate(r["positions"]["before_action"])
                                                        if sentences is None or i in sentences]))
            for name, r in reads.items()}


def against(orig: dict, vers: dict, names: list[str], measure) -> tuple[np.ndarray, np.ndarray]:
    """Per spot: the measure at the same spot, and its mean against the version at every other spot (chance)."""
    same = np.array([measure(orig[n], vers[n]) for n in names])
    chance = np.array([np.nanmean([measure(orig[n], vers[m]) for m in names if m != n]) for n in names])
    return same, chance


def summarize(same: np.ndarray, chance: np.ndarray, base: tuple[np.ndarray, np.ndarray] | None) -> dict:
    """`base` = base gemma's (same, chance) at the same spots, for a checkpoint's paired change."""
    out = {"spots": int(len(same)), "same_spot": float(np.nanmean(same)), "chance": float(np.nanmean(chance))}
    out["above_chance"], *out["above_chance_ci"] = words.mean_ci(same - chance)
    if base is not None:
        out["change_from_base"], *out["change_from_base_ci"] = words.mean_ci((same - chance) - (base[0] - base[1]))
        out["same_spot_change"], *out["same_spot_change_ci"] = words.mean_ci(same - base[0])
    return out


def line(kind: str, v: str, group: str, r: dict) -> str:
    change = "" if "change_from_base" not in r else (
        f"; change from base gemma, above chance {r['change_from_base']:+.4f} [{r['change_from_base_ci'][0]:+.4f}, "
        f"{r['change_from_base_ci'][1]:+.4f}], same spot alone {r['same_spot_change']:+.4f} "
        f"[{r['same_spot_change_ci'][0]:+.4f}, {r['same_spot_change_ci'][1]:+.4f}]")
    return (f"{kind} | {v} vs {ORIGINAL} | {group}: same spot {r['same_spot']:.4f}, chance {r['chance']:.4f}, above chance "
            f"{r['above_chance']:+.4f} [{r['above_chance_ci'][0]:+.4f}, {r['above_chance_ci'][1]:+.4f}] ({r['spots']} spots){change}")


def main() -> None:
    result: dict = {"original": ORIGINAL, "versions": VERSIONS, "held_out_tasks": list(spots.HELD_OUT), "jspace": {},
                    "original_against_itself": {}, "nla": {}, "lens_fit": {}, "moved_from_base": {}}
    orig_lens, lens = load(ORIGINAL, "lens"), {v: load(v, "lens") for v in VERSIONS}
    orig_nla, nla = load(ORIGINAL, "nla"), {v: load(v, "nla") for v in VERSIONS}
    common = sorted(set(orig_lens).intersection(*(set(lens[v]) for v in VERSIONS)))
    groups = spot_groups(common)
    print(f"spots read by every model: {len(common)} of {len(spots.spots())} "
          f"({len(groups['trained tasks'])} on trained tasks, {len(groups['held-out tasks'])} on held-out tasks)", flush=True)

    for depth in readout.DEPTHS:
        o = lens_sets(orig_lens, depth)
        # the top of the scale: the original against itself (same spot = identical words, so 1; chance = its own
        # words at other spots). A version's (same spot minus chance) can be read as a share of this.
        top = summarize(*against(o, o, common, words.jaccard), None)
        result["original_against_itself"][f"{depth:.3f}"] = top
        print(line(f"jspace | depth {depth:.3f}", ORIGINAL, "all spots", top), flush=True)
        base = None
        for v in VERSIONS:
            s = lens_sets(lens[v], depth)
            same, chance = against(o, s, common, words.jaccard)
            result["jspace"].setdefault(f"{depth:.3f}", {})[v] = grouped(f"jspace | depth {depth:.3f}", v, common, groups,
                                                                          same, chance, base)
            base = base or (same, chance)
            if depth == readout.HEADLINE:
                result["jspace"][f"{depth:.3f}"][v]["per_spot"] = dict(zip(common, same.tolist()))

    nla_common = sorted(set(common) & set(orig_nla).intersection(*(set(nla[v]) for v in VERSIONS)))
    if nla_common:
        o = nla_counts(orig_nla)
        base = None
        for v in VERSIONS:
            same, chance = against(o, nla_counts(nla[v]), nla_common, words.cosine)
            result["nla"][v] = grouped("nla", v, nla_common, groups, same, chance, base)
            base = base or (same, chance)
    else:
        print(f"nla | no spot has NLA sentences for {ORIGINAL} and every version: the NLA half is not computed", flush=True)

    base_sets = lens_sets(lens[VERSIONS[0]], readout.HEADLINE)
    for v in VERSIONS:
        fits, shared = [], []
        for n in common:
            r = lens[v][n]
            layer = str(readout.layer_at(r["n_layers"], [int(k) for k in r["layers"]], FIT_DEPTH))
            fits.append(r["layers"][layer]["ids"][0] == r["model"]["ids"][0])
            shared.append(words.jaccard(set(r["layers"][layer]["ids"]), set(r["model"]["ids"])))
        result["lens_fit"][v] = {"top_token": float(np.mean(fits)), "top_200_overlap": float(np.mean(shared))}
        print(f"lens fit | {v}: the lens it was read through, at depth {FIT_DEPTH} gives the model's own top next token at "
              f"{np.mean(fits):.3f} of {len(fits)} spots; its top-200 tokens overlap the model's own top-200 next "
              f"tokens by {np.mean(shared):.3f} (Jaccard)", flush=True)
        if v != VERSIONS[0]:
            mine = lens_sets(lens[v], readout.HEADLINE)
            moved, lo, hi = words.mean_ci([1 - words.jaccard(base_sets[n], mine[n]) for n in common])
            result["moved_from_base"][v] = {"jspace_distance": moved, "ci": [lo, hi]}
            print(f"moved from base | {v}: its J-lens words at depth {readout.HEADLINE:.3f} differ from base gemma's "
                  f"at the same spot by {moved:.4f} [{lo:.4f}, {hi:.4f}] (1 - Jaccard; 0 = identical)", flush=True)

    path = models.OUT / EXPERIMENT / f"similarity{SUFFIX}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
