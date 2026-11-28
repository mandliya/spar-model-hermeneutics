"""Why is the NLA result negative? Tests of the two candidate causes, on every restart point (18 Nov).

    READOUT_OUT=<folder from shared/fetch.py> python root_cause.py

The NLA score (similarity.py): cosine of two models' NLA word counts at the same restart point, minus the same against
other restart points (chance); the result is the final checkpoint's score minus base gemma's, averaged over restart
points. word_shift.py suggested two causes. This tests each one with numbers, on all restart points (5 NLA
sentences per model per restart point), not on chosen examples:
  split        the result with and without the currency_conversion task, and each part's share of the total
  generic      is the final checkpoint's NLA text more alike across unrelated restart points than base gemma's?
               (cosine of one model's word counts at two different restart points, averaged)
  remove       the result again after removing the cause: (a) words weighted by rarity (a word counts less the more
               restart points have it), (b) the K words whose spread grew most from base gemma to the final checkpoint
               deleted from every model's text, beside a control that deletes K other words of similar frequency
  scale        how far base gemma and the final checkpoint are from "a random restart point" (0) towards "Qwen3.6's
               NLA against itself" (the top), outside currency_conversion, with rare words counted more
  subject      on currency_conversion: at what share of restart points does each model's NLA text name the task's
               subject, and at what share does it talk about notebooks or code instead
  examples     the first NLA sentence of each model at restart points drawn at random (fixed seed), not picked
The scoring is the shared methods (../methods/: cosine of word counts, and the rare-word weighting) and the shared
evaluation (../evaluation.py). Intervals: 95%, resampling restart points (shared.words.mean_ci). Prints every number.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: method_menu, restart_points
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods, shared
from readout_methods.likeness import evaluation, rare_words, value_weighted  # noqa: E402
from readouts import BASE, FINAL, MODELS, ORIGINAL, as_readouts, compare, load_counts, sentence_readouts, table  # noqa: E402
from restart_points import tasks  # noqa: E402
from shared import words  # noqa: E402

TASK = "currency_conversion"
SUBJECT = {"currency", "conversion", "exchange", "usd", "inr", "forex", "rate", "fee"}
CODE = {"notebook", "jupyter", "sql", "code", "script"}
KS = (10, 25, 50)
SEED = 0
FIRST, SECOND = {0, 1, 2}, {3, 4}   # the two groups of a model's five sentences, for comparing a model with itself


def ci(values) -> str:
    m, lo, hi = words.mean_ci(values)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def change(counts: dict, names: list[str], method=value_weighted.cosine, **options) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per restart point: change in same, change in chance, change in (same - chance), final checkpoint minus base gemma."""
    tables = {v: compare(method, counts, ORIGINAL, v, names, **options) for v in (BASE, FINAL)}
    same = {v: np.diag(tables[v]) for v in tables}
    chance = {v: evaluation.chance(tables[v]) for v in tables}
    return same[FINAL] - same[BASE], chance[FINAL] - chance[BASE], (same[FINAL] - chance[FINAL]) - (same[BASE] - chance[BASE])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default=None, help="a folder to also write nla_root_cause.json into (for the figure)")
    args = ap.parse_args()
    result: dict = {"split": {}, "generic": {}, "remove": {}, "subject": {}}
    nla, counts, names = load_counts()
    task = tasks(names)
    n = len(names)
    in_task = np.array([t == TASK for t in task])
    print(f"{n} restart points, every one used; {int(in_task.sum())} are {TASK}; 5 NLA sentences per model per restart point", flush=True)

    d_same, d_chance, d_above = change(counts, names)
    ok = ~np.isnan(d_above)
    print(f"split | all restart points: same {ci(d_same)}, chance {ci(d_chance)}, same - chance {ci(d_above)}")
    result["split"]["all"] = {"points": int(ok.sum()), "same_minus_chance": words.mean_ci(d_above), "same": words.mean_ci(d_same), "chance": words.mean_ci(d_chance)}
    for label, mask in ((TASK, in_task & ok), (f"every task but {TASK}", ~in_task & ok)):
        result["split"][label] = {"points": int(mask.sum()), "same_minus_chance": words.mean_ci(d_above[mask]), "same": words.mean_ci(d_same[mask]),
                                  "chance": words.mean_ci(d_chance[mask]), "adds_to_overall": float(np.nansum(d_above[mask]) / ok.sum())}
        print(f"split | {label} ({int(mask.sum())} points): same {ci(d_same[mask])}, chance {ci(d_chance[mask])}, "
              f"same - chance {ci(d_above[mask])}; adds {np.nansum(d_above[mask]) / ok.sum():+.4f} to the overall {np.nanmean(d_above):+.4f}")

    # how alike one model's NLA text is to its own text at every other restart point
    alike = {t: evaluation.chance(compare(value_weighted.cosine, counts, t, t, names)) for t in (ORIGINAL, BASE, FINAL)}
    print("generic | one model's NLA text at two different restart points, cosine: " +
          ", ".join(f"{t} {np.nanmean(alike[t]):.4f}" for t in alike))
    result["generic"] = {t: float(np.nanmean(alike[t])) for t in alike} | {"change": words.mean_ci(alike[FINAL] - alike[BASE])}
    print(f"generic | final checkpoint minus base gemma: {ci(alike[FINAL] - alike[BASE])}; rose at "
          f"{int(np.nansum(alike[FINAL] > alike[BASE]))} of {n} restart points; outside {TASK}: {ci((alike[FINAL] - alike[BASE])[~in_task])}")

    # remove the cause (a): rare words count more, the weights counted over the three models' texts
    ws, wc, wa = change(counts, names, rare_words.rarity_cosine, top=None, value="p")
    result["remove"]["rarity"] = {"all": words.mean_ci(wa), "other_tasks": words.mean_ci(wa[~in_task & ~np.isnan(wa)]), "task": words.mean_ci(wa[in_task & ~np.isnan(wa)])}
    print(f"remove | words weighted by rarity: same {ci(ws)}, chance {ci(wc)}, same - chance {ci(wa)}; "
          f"outside {TASK}: same - chance {ci(wa[~in_task & ~np.isnan(wa)])}")
    # remove the cause (b): delete the words whose spread grew most, against a control of similar words
    share = {t: collections.Counter() for t in (BASE, FINAL)}
    for t in share:
        for x in names:
            share[t].update(set(counts[t][x]))
    grew = sorted(share[FINAL], key=lambda w: (share[BASE][w] - share[FINAL][w], w))   # biggest growth first; ties by the word, so every run picks the same
    rng = np.random.default_rng(SEED)
    for k in KS:
        chosen = grew[:k]
        steady = sorted(w for w in share[FINAL] if abs(share[FINAL][w] - share[BASE][w]) <= 0.03 * n and w not in chosen)
        control = []
        for w in chosen:   # for each deleted word, a control word used at about as many restart points by the final checkpoint
            near = sorted(steady, key=lambda c: abs(share[FINAL][c] - share[FINAL][w]))[:8]
            pick = near[int(rng.integers(len(near)))]
            control.append(pick)
            steady.remove(pick)
        for label, gone in (("the words that spread", set(chosen)), ("control words", set(control))):
            cut = {t: {x: collections.Counter({w: c for w, c in counts[t][x].items() if w not in gone}) for x in names} for t in counts}
            cs, cc, ca = change(cut, names)
            print(f"remove | {k} {label} deleted ({', '.join(sorted(gone)[:8])}...): same {ci(cs)}, chance {ci(cc)}, same - chance {ci(ca)}")

    # the scale: 0 = no closer than a random restart point; the top = Qwen3.6's NLA against itself. An NLA never
    # writes the same text twice, so "against itself" is Qwen3.6's sentences 1-3 against its sentences 4-5, and each
    # gemma is scored the same way: Qwen3.6's sentences 1-3 against that gemma's sentences 4-5.
    pool = [as_readouts(counts[t], names) for t in MODELS]
    first = sentence_readouts(nla, ORIGINAL, FIRST, names)
    along = {}
    for t in MODELS:
        split = table(rare_words.rarity_cosine, first, sentence_readouts(nla, t, SECOND, names), pool, top=None, value="p")
        along[t] = float(np.nanmean(evaluation.above_chance(split)[~in_task]))
    result["scale"] = {"top": along[ORIGINAL], BASE: along[BASE], FINAL: along[FINAL]}
    print(f"scale | outside {TASK}, rare words count more, Qwen3.6's sentences 1-3 against each model's sentences 4-5, same - chance: "
          f"Qwen3.6 itself {along[ORIGINAL]:+.4f} (the top), base gemma {along[BASE]:+.4f}, final checkpoint {along[FINAL]:+.4f}; "
          f"share of the way to the top: base gemma {along[BASE] / along[ORIGINAL]:.1%}, final checkpoint {along[FINAL] / along[ORIGINAL]:.1%}")

    task_names = [x for x, t in zip(names, task) if t == TASK]
    for t in (ORIGINAL, BASE, FINAL):
        subj = np.mean([bool(SUBJECT & set(counts[t][x])) for x in task_names])
        cur = np.mean(["currency" in counts[t][x] for x in task_names])
        code = np.mean([bool(CODE & set(counts[t][x])) for x in task_names])
        nb = np.mean([bool({"notebook", "jupyter"} & set(counts[t][x])) for x in task_names])
        result["subject"][t] = {"names_subject": float(subj), "notebook_or_jupyter": float(nb), "points": len(task_names)}
        print(f"subject | {TASK}, {t}: names the subject ({', '.join(sorted(SUBJECT))}) at {subj:.2f} of {len(task_names)} restart points; "
              f"the word 'currency' at {cur:.2f}; notebook or jupyter at {nb:.2f}; any of {', '.join(sorted(CODE))} at {code:.2f}")

    picks = [task_names[i] for i in rng.choice(len(task_names), 2, replace=False)]
    rest = [x for x, t in zip(names, task) if t != TASK]
    picks += [rest[i] for i in rng.choice(len(rest), 2, replace=False)]
    for x in picks:
        i = names.index(x)
        print(f"examples | {x} ({task[i]}): change in same {d_same[i]:+.3f}, in chance {d_chance[i]:+.3f}")
        for t in (ORIGINAL, BASE, FINAL):
            first = next((s["explanation"].strip() for s in nla[t][x]["positions"]["before_action"] if s["explanation"].strip()), "(all five empty)")
            print(f"examples |    {t}: {first[:330]!r}")
    if args.to:
        path = pathlib.Path(args.to) / "nla_root_cause.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
