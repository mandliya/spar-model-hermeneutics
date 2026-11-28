"""Why did the NLA score go down after distillation? The identical-words NLA score, taken apart (17 Nov).

    READOUT_OUT=<folder from shared/fetch.py> python word_shift.py

similarity.py reports, for the NLA sentences, the final checkpoint's change in (same restart point minus chance)
against Qwen3.6. This prints what that number is made of, from the same saved files, scored with the shared cosine
of word counts (../methods/value_weighted.py) and the shared evaluation (../evaluation.py):
  parts      the change in the same-point score and in the chance score separately, and at how many restart points each rose
  words      the words the final checkpoint's NLA uses at more (and at fewer) restart points than base gemma's, with
             the share of restart points whose NLA text has the word, for base gemma, the final checkpoint and Qwen3.6
  volume     the most frequent words, words per restart point, and empty explanations per model
  rows       the restart points with the biggest drops and rises
  tasks      the change per task
  --show <restart point>   the first two explanations of each model at one restart point
Found on 17 Nov: same +0.0032, chance +0.0099 (chance rose at 280 of 327 points): the final checkpoint's sentences
became more generic ("continuation", "sql", "jupyter", "technical" at many more points), and on currency_conversion
(62 points, same -0.040) they lose the task's subject.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: method_menu, restart_points
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods, shared
from readout_methods.likeness import evaluation, value_weighted  # noqa: E402
from readouts import BASE, FINAL, ORIGINAL, compare, load_counts  # noqa: E402
from restart_points import tasks  # noqa: E402

SHOWN = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None, help="a restart point whose explanations to print")
    args = ap.parse_args()
    nla, counts, common = load_counts()
    tables = {v: compare(value_weighted.cosine, counts, ORIGINAL, v, common) for v in (BASE, FINAL)}
    sb, cb = np.diag(tables[BASE]), evaluation.chance(tables[BASE])
    sf, cf = np.diag(tables[FINAL]), evaluation.chance(tables[FINAL])
    n = len(common)
    print(f"parts | {n} restart points", flush=True)
    print(f"parts | base gemma: same {np.nanmean(sb):.4f}, chance {np.nanmean(cb):.4f}, above chance {np.nanmean(sb - cb):+.4f}")
    print(f"parts | final checkpoint: same {np.nanmean(sf):.4f}, chance {np.nanmean(cf):.4f}, above chance {np.nanmean(sf - cf):+.4f}")
    print(f"parts | change: same {np.nanmean(sf - sb):+.4f}, chance {np.nanmean(cf - cb):+.4f}, above chance {np.nanmean((sf - cf) - (sb - cb)):+.4f}")
    print(f"parts | restart points where it rose: same {int(np.sum(sf > sb))}, chance {int(np.sum(cf > cb))}, above chance {int(np.sum((sf - cf) > (sb - cb)))}")

    def share(t: str) -> collections.Counter:
        d: collections.Counter = collections.Counter()
        for name in common:
            d.update(set(counts[t][name]))
        return d
    dq, db, dfin = share(ORIGINAL), share(BASE), share(FINAL)
    order = sorted(set(db) | set(dfin), key=lambda w: (db[w] - dfin[w], w))   # biggest rise first; ties by the word, so the order is the same every run
    for title, chosen in (("at MORE restart points than base gemma's", order[:SHOWN]), ("at FEWER", order[-SHOWN:][::-1])):
        print(f"words | the final checkpoint's NLA uses these {title} (share of points: base -> final | Qwen3.6):")
        for w in chosen:
            print(f"words |   {w:18s} {db[w] / n:.2f} -> {dfin[w] / n:.2f} | {dq[w] / n:.2f}")

    total = {t: sum((counts[t][name] for name in common), collections.Counter()) for t in counts}
    print("volume | most frequent words, share of all words: Qwen3.6 | base | final")
    for w, _ in total[ORIGINAL].most_common(15):
        print(f"volume |   {w:14s} " + " | ".join(f"{total[t][w] / sum(total[t].values()):.3f}" for t in (ORIGINAL, BASE, FINAL)))
    print("volume | words per restart point:", {t: round(sum(total[t].values()) / n) for t in total})
    print("volume | empty explanations:", {t: sum(1 for name in common for s in nla[t][name]["positions"]["before_action"]
                                                   if not s["explanation"].strip()) for t in nla})

    change = (sf - cf) - (sb - cb)
    task = tasks(common)
    ranked = np.argsort(change)
    for title, idx in (("biggest drops", ranked[:8]), ("biggest rises", [i for i in ranked[::-1] if not np.isnan(change[i])][:5])):
        for i in idx:
            print(f"rows | {title}: {common[i]} ({task[i]}): same {sb[i]:.3f} -> {sf[i]:.3f}, chance {cb[i]:.3f} -> {cf[i]:.3f}")
    by_task = collections.defaultdict(list)
    for i, t in enumerate(task):
        by_task[t].append(i)
    for t, idx in sorted(by_task.items(), key=lambda kv: -len(kv[1])):
        print(f"tasks | {t} ({len(idx)} points): same {np.nanmean((sf - sb)[idx]):+.4f}, chance {np.nanmean((cf - cb)[idx]):+.4f}, "
              f"above chance {np.nanmean(change[idx]):+.4f}")
    if args.show:
        for t in (ORIGINAL, BASE, FINAL):
            for s in nla[t][args.show]["positions"]["before_action"][:2]:
                print(f"show | {t}: {s['explanation'].strip()[:900]!r}")


if __name__ == "__main__":
    main()
