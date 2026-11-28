"""Run every way of scoring how alike two J-lens readouts are through one evaluation (17 Nov, exploratory).

    READOUT_DISTRIBUTIONS=<folder of write_jlens_lists.py's files> python similarity_methods.py [--depths 0.667 0.95]
    ... python similarity_methods.py --selection rare_words     # method 8 and its variants only
    ... python similarity_methods.py --selection hybrids        # rarity combined with confidence or rank

The methods are in readout_methods/likeness/ (one file per family), listed with their labels in ../method_menu.py; what is computed from a method's
table is in ../evaluation.py; the loading is in readouts.py. This script only loops and records.

Each method is run on Qwen3.6 against base gemma and against the final checkpoint, at every depth, through the
published lenses ("borrowed") and each gemma's own fitted lens ("own"), and twice: on all words, and "masked"
(method 9: special tokens and words without a letter or digit removed).

The yardstick, fixed before looking at any distillation number: for BASE GEMMA against Qwen3.6, how often does a
method score the same restart point above a different one?
  beats other      over all other restart points
  beats same task  over other restart points of the same task only (the harder question: this moment, not this task)
Then, per method: the final checkpoint's change in (same - chance) from base gemma, paired by restart point, with a
95% interval from resampling restart points, and that change in units of its own spread (mean / sd of the per-point
change), so methods on different scales can be read side by side. For scale, Qwen3.6 is also scored against itself
(same - chance): the most a gemma could reach under that method (added 20 Nov).
Prints every number it writes; writes <folder>/similarity_methods.json (or the selection's file, OUTPUT below).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: method_menu, restart_points
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods, shared
import method_menu  # noqa: E402
from readout_methods.likeness import evaluation  # noqa: E402
from readout_methods.likeness.pair import Pair  # noqa: E402
from readouts import BASE, FINAL, FOLDER, LENSES, QWEN, load_shared  # noqa: E402  (puts the experiment's root on the path)
from restart_points import tasks  # noqa: E402
from shared import readout, words  # noqa: E402

OUTPUT = {"all": "similarity_methods.json", "rare_words": "similarity_methods_rarity.json", "hybrids": "similarity_methods_hybrids.json",
          "concept_score": "similarity_methods_concept_score.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", nargs="*", type=float, default=list(readout.DEPTHS))
    ap.add_argument("--selection", default="all", choices=list(method_menu.SELECTIONS), help="which methods to run")
    ap.add_argument("--no_meaning", action="store_true", help="skip method 10, which needs the embedding model")
    args = ap.parse_args()
    gemma_reads = [(model, lens) for chosen in LENSES.values() for model, lens in chosen.items()]
    reads, names = load_shared(list(dict.fromkeys([(QWEN, "published"), *gemma_reads])))
    n = len(names)
    other = evaluation.others(n)
    other_same_task = other & evaluation.same_task(tasks(names))
    for (model, lens), r in reads.items():
        for j, layer in enumerate(r.layers):
            m = r.mass[:, j].mean(0)
            print(f"confidence | {model} through its {lens} lens | layer {layer}: of the whole vocabulary's probability, the top "
                  f"1 token holds {m[0]:.3f}, top 10 {m[1]:.3f}, top 200 {m[2]:.3f}, top 5,000 {m[3]:.3f}; entropy "
                  f"{r.entropy[:, j].mean():.2f} nats", flush=True)
    emb = None
    if args.selection == "all" and not args.no_meaning:
        from shared.embedding import Embedder
        emb = Embedder()
    result: dict = {"restart_points": n, "methods": {}}
    for depth in args.depths:
        for masked in (False, True):
            q = reads[(QWEN, "published")].lists(depth, masked)
            for lens_name, chosen in LENSES.items():
                gemma = {v: reads[(v, chosen[v])].lists(depth, masked) for v in (BASE, FINAL)}
                pool = [q, gemma[BASE], gemma[FINAL]]   # all three models: what the rare-word weights are counted over
                sims = {v: method_menu.tables(Pair(q, gemma[v], emb, pool), args.selection) for v in (BASE, FINAL)}
                itself = method_menu.tables(Pair(q, q, emb, pool), args.selection)   # the top of the scale: Qwen3.6 against itself
                for method in sims[BASE]:
                    above = {v: evaluation.above_chance(sims[v][method]) for v in (BASE, FINAL)}
                    top = float(evaluation.above_chance(itself[method]).mean())
                    beats, _ = evaluation.same_moment_rates(sims[BASE][method], other)
                    beats_task, _ = evaluation.same_moment_rates(sims[BASE][method], other_same_task)
                    change = above[FINAL] - above[BASE]
                    m, lo, hi = words.mean_ci(change)
                    tag = f"depth {depth:.3f} | {lens_name} | {'masked' if masked else 'all words'} | {method}"
                    result["methods"][tag] = {"beats_other": beats, "beats_same_task": beats_task, "base_above_chance": float(above[BASE].mean()),
                                              "change": m, "ci": [lo, hi], "change_in_sd": float(change.mean() / change.std()),
                                              "qwen_against_itself": top}
                    print(f"{tag}: base gemma same beats other {beats:.3f} (same task only {beats_task:.3f}), above chance "
                          f"{above[BASE].mean():+.4f}; final checkpoint's change {m:+.4f} [{lo:+.4f}, {hi:+.4f}], "
                          f"{change.mean() / change.std():+.2f} sd; Qwen3.6 against itself, above chance {top:+.4f}", flush=True)
    path = FOLDER / OUTPUT[args.selection]
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
