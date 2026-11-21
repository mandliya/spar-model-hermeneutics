"""Question 1, the follow-up: the distillation changed gemma's behaviour, so where in gemma does the change show?

    READOUT_OUT=<folder from shared/fetch.py> python where_it_moved.py

similarity.py found no move toward Qwen3.6 two-thirds of the way in, while Gigi's own evaluation shows the distilled
gemma behaving more like Qwen3.6 (gigi/3-forensics/HERO_RESULTS.md, section 5b). This reads the same saved files
(nothing is rerun on a GPU) and prints, for the same restart points:
  by depth        at each stored depth and for the model's own top next tokens: overlap with Qwen3.6 for base gemma
                  and the final checkpoint, the paired change (above chance, and same spot alone), and how far the
                  final checkpoint's words are from base gemma's (1 - Jaccard)
  first word      how often a version's most likely next word is Qwen3.6's most likely next word, per checkpoint, per task
  gained, lost    the words the final checkpoint gained and lost against base gemma: what share of each Qwen3.6 has at
                  the same spot and at other spots, and the words gained or lost at the most spots
  robustness      the 2/3 and 0.95 depth results again on the top 10 / 25 / 50 / 100 words and on plain English-alphabet
                  words only (gemma's mid-depth lists are mostly translations of a few words)
  NLA health      per model: the NLA's own reconstruction score, and how many descriptions are empty
Chance here is the mean over CHANCE_SPOTS other spots (fixed by SEED), not all others, so values differ from
similarity.py's in the fourth decimal. Added on 14 Nov after the results. Prints every number it writes; writes
OUT/distillation_similarity/where_it_moved.json.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import spots  # noqa: E402
from shared import models, readout, words  # noqa: E402
from similarity import EXPERIMENT, ORIGINAL, SUFFIX, VERSIONS, lens_lists, load  # noqa: E402

BASE, FINAL = VERSIONS[0], VERSIONS[-1]
CHANCE_SPOTS, SEED = 20, 0
TOP_KS = (10, 25, 50, 100, 200)
SHOWN = 30


def own_words(reads: dict[str, dict]) -> dict[str, list[str]]:
    """Per spot, the model's own top next tokens as words, most likely first."""
    return {name: words.lens_words(r["model"]) for name, r in reads.items()}


def english(ws: list[str]) -> list[str]:
    return [w for w in ws if w.isascii() and w.isalpha()]


def ci(values) -> str:
    m, lo, hi = words.mean_ci(values)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main() -> None:
    tags = [ORIGINAL, *VERSIONS]
    lens = {t: load(t, "lens") for t in tags}
    nla = {t: load(t, "nla") for t in tags}
    common = sorted(set.intersection(*(set(lens[t]) for t in tags)))
    env = {s.name: s.env for s in spots.spots()}
    rng = np.random.default_rng(SEED)
    others = {n: [common[j] for j in rng.choice([j for j in range(len(common)) if common[j] != n], CHANCE_SPOTS, replace=False)]
              for n in common}
    print(f"spots read by every model: {len(common)}; chance = the mean over {CHANCE_SPOTS} other spots", flush=True)
    result: dict = {"spots": len(common), "by_depth": {}, "first_word": {}, "gained_lost": {}, "robustness": {}, "nla_health": {}}

    def scores(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
        same = np.array([words.jaccard(set(a[n]), set(b[n])) for n in common])
        chance = np.array([np.mean([words.jaccard(set(a[n]), set(b[m])) for m in others[n]]) for n in common])
        return same, chance

    levels = {f"depth {d:.3f}": {t: lens_lists(lens[t], d) for t in tags} for d in readout.DEPTHS}
    levels["own next tokens"] = {t: own_words(lens[t]) for t in tags}
    for level, lists in levels.items():
        sb, cb = scores(lists[ORIGINAL], lists[BASE])
        sf, cf = scores(lists[ORIGINAL], lists[FINAL])
        moved = float(np.mean([1 - words.jaccard(set(lists[BASE][n]), set(lists[FINAL][n])) for n in common]))
        result["by_depth"][level] = {"base_same": float(sb.mean()), "base_chance": float(cb.mean()), "final_same": float(sf.mean()),
                                     "final_chance": float(cf.mean()), "change_above_chance": words.mean_ci((sf - cf) - (sb - cb)),
                                     "change_same_alone": words.mean_ci(sf - sb), "base_to_final_distance": moved}
        print(f"by depth | {level}: overlap with {ORIGINAL}: base gemma same spot {sb.mean():.4f}, chance {cb.mean():.4f}; final "
              f"checkpoint same spot {sf.mean():.4f}, chance {cf.mean():.4f}; change above chance {ci((sf - cf) - (sb - cb))}; "
              f"change same spot alone {ci(sf - sb)}; final checkpoint's words differ from base gemma's by {moved:.3f}", flush=True)

    first = {t: {n: ws[0] for n, ws in levels["own next tokens"][t].items()} for t in tags}
    for v in VERSIONS:
        agree = float(np.mean([first[v][n] == first[ORIGINAL][n] for n in common]))
        result["first_word"][v] = agree
        print(f"first word | {v}: its most likely next word is {ORIGINAL}'s at {agree:.3f} of {len(common)} spots", flush=True)
    print(f"first word | base gemma and the final checkpoint agree with each other at "
          f"{np.mean([first[BASE][n] == first[FINAL][n] for n in common]):.3f} of spots", flush=True)
    for t in (ORIGINAL, BASE, FINAL):
        print(f"first word | most common for {t}: {collections.Counter(first[t][n] for n in common).most_common(8)}", flush=True)
    by_env = collections.defaultdict(list)
    for n in common:
        by_env[env[n]].append(n)
    for e, names in sorted(by_env.items(), key=lambda kv: -len(kv[1])):
        b, f = (float(np.mean([first[v][n] == first[ORIGINAL][n] for n in names])) for v in (BASE, FINAL))
        result["first_word"][f"task {e}"] = {"spots": len(names), "base": b, "final": f, "held_out": e in spots.HELD_OUT}
        print(f"first word | task {e} ({len(names)} spots{', held out' if e in spots.HELD_OUT else ''}): agrees with "
              f"{ORIGINAL} at {b:.2f} for base gemma, {f:.2f} for the final checkpoint", flush=True)

    for level in (f"depth {readout.HEADLINE:.3f}", "depth 0.950", "own next tokens"):
        q, b, f = (levels[level][t] for t in (ORIGINAL, BASE, FINAL))
        share = collections.defaultdict(list)
        gained, lost = collections.Counter(), collections.Counter()
        for n in common:
            sets = {"gained": set(f[n]) - set(b[n]), "lost": set(b[n]) - set(f[n]), "kept": set(b[n]) & set(f[n])}
            for kind, ws in sets.items():
                if ws:
                    share[f"{kind} same spot"].append(len(ws & set(q[n])) / len(ws))
                    share[f"{kind} other spots"].append(np.mean([len(ws & set(q[m])) / len(ws) for m in others[n]]))
            gained.update(sets["gained"])
            lost.update(sets["lost"])
        s = {k: float(np.mean(v)) for k, v in share.items()}
        result["gained_lost"][level] = s | {"gained_most": gained.most_common(SHOWN), "lost_most": lost.most_common(SHOWN)}
        print(f"gained, lost | {level}: of the words the final checkpoint GAINED, {ORIGINAL} has {s['gained same spot']:.4f} at the "
              f"same spot ({s['gained other spots']:.4f} at other spots); of the words it LOST, {s['lost same spot']:.4f} "
              f"({s['lost other spots']:.4f}); of the words it KEPT, {s['kept same spot']:.4f} ({s['kept other spots']:.4f})", flush=True)
        print(f"gained, lost | {level}: gained at the most spots (word, spots of {len(common)}): {gained.most_common(SHOWN)}", flush=True)
        print(f"gained, lost | {level}: lost at the most spots: {lost.most_common(SHOWN)}", flush=True)

    for depth in (readout.HEADLINE, 0.95):
        full = levels[f"depth {depth:.3f}"]
        print(f"robustness | depth {depth:.3f}: share of a top-200 list that is plain English-alphabet words: "
              + ", ".join(f"{t} {np.mean([len(english(full[t][n])) / len(full[t][n]) for n in common]):.3f}" for t in (ORIGINAL, BASE, FINAL)), flush=True)
        cuts = {f"top {k}": (lambda ws, k=k: ws[:k]) for k in TOP_KS} | {"English-alphabet words of the top 200": english}
        for name, cut in cuts.items():
            lists = {t: {n: cut(full[t][n]) for n in common} for t in tags}
            sb, cb = scores(lists[ORIGINAL], lists[BASE])
            row = {"base_same": float(sb.mean()), "base_chance": float(cb.mean())}
            parts = [f"base gemma same spot {sb.mean():.4f}, chance {cb.mean():.4f}"]
            for v in VERSIONS[1:]:
                sv, cv = scores(lists[ORIGINAL], lists[v])
                row[v] = words.mean_ci((sv - cv) - (sb - cb))
                parts.append(f"{v} change above chance {ci((sv - cv) - (sb - cb))}")
            result["robustness"][f"depth {depth:.3f} | {name}"] = row
            print(f"robustness | depth {depth:.3f} | {name}: " + "; ".join(parts), flush=True)

    for t in tags:
        tries = [s for n in common if n in nla[t] for s in nla[t][n]["positions"]["before_action"]]
        cos = [s["cos"] for s in tries if s.get("cos") is not None]
        empty = float(np.mean([not s["explanation"].strip() for s in tries]))
        result["nla_health"][t] = {"tries": len(tries), "cos": float(np.mean(cos)) if cos else None, "empty": empty}
        print(f"NLA health | {t}: {len(tries)} descriptions; the NLA's own reconstruction of the activation from the text scores "
              f"{'%.4f' % np.mean(cos) if cos else 'n/a (no reverse model)'} (cosine); {empty:.3f} are empty", flush=True)

    path = models.OUT / EXPERIMENT / f"where_it_moved{SUFFIX}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
