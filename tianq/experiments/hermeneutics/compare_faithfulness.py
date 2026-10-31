"""CPU: the probe faithfulness gap, target minus substitute (experiments/hermeneutics/faithfulness.py)

    DYL_MODEL=qwen3-next-80b python -m experiments.hermeneutics.compare_faithfulness                              # the target's scores
    DYL_MODEL=qwen3-next-80b DYL_READER=gemma-3-27b-it python -m experiments.hermeneutics.compare_faithfulness    # a substitute's
    DYL_MODEL=qwen3-next-80b python -m experiments.hermeneutics.compare_faithfulness --gaps gemma-3-27b-it gemma-3-12b-it qwen3.5-27b

Without --gaps: each category's own probe for the model DYL_READER names (DYL_MODEL itself when
unset) reading DYL_MODEL's conversations, out of fold, kept per row as
gigi/results/faithfulness_scores_<recipe>.json under that pair's namespace (core/store.py). Every
category of both variants must be stored in full (experiments/cross_category_transfer/compare_diversity.py load).

With --gaps, run as the target: its stored scores and each named substitute's, matched row by row,
give every category's gap with its bootstrap interval, stored as gigi/results/faithfulness_gaps.json
and drawn as gigi/charts/faithfulness_gaps.png under the target's namespace.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download

from dyl_repro import datasets
from dyl_repro.core import training
from dyl_repro.core.model import CURRENT, IS_SELF_READ, MODELS, READER, Model
from dyl_repro.core.probe import hidden_states_index_for
from dyl_repro.core.store import SCRATCH, RowStore
from experiments.cross_category_transfer import diversity
from experiments.cross_category_transfer.compare_diversity import load
from experiments.cross_category_transfer.generalization import CATEGORIES
from experiments.hermeneutics import faithfulness, faithfulness_charts

GAPS = "faithfulness_gaps"
CAPTION = ("Each model's own in-category probe (AISI's recipe at its matched layer, 5-fold), on the same rows, folds and"
           " labels (the target's lies). Substitutes are prefilled with the target's exact conversations; nothing is"
           " generated. Words only: no model, the conversation's words (TF-IDF + logistic regression). Gap = target AUROC"
           " - substitute AUROC. Whiskers: its 95% interval (paired bootstrap over rows); one that reaches the black bar"
           " is no clear gap. Faded: fewer than 10 of a class. Markers are nudged up or down so they don't overlap; only"
           " left-right position is data.")


def scores_name(model: Model) -> str:
    """Named after the recipe, like every other result; the layer is the reader's own."""
    lam = str(training.DEFAULT_L2_LAMBDA).replace(".", "_")
    return f"faithfulness_scores_l_{model.probe_layer}_lm_{lam}_ar_lr_hs{hidden_states_index_for(model.probe_layer)}"


def stored_scores(reader: str) -> dict:
    """A model's stored scores (reading the target's conversations), by explicit Hub path."""
    namespace = CURRENT.tag if reader == CURRENT.tag else f"{CURRENT.tag}/read_by_{reader}"
    path = hf_hub_download(os.environ["HF_DATASET_REPO"], f"{namespace}/gigi/results/{scores_name(MODELS[reader])}.json",
                           repo_type="dataset", local_dir=SCRATCH, token=os.environ["HF_TOKEN"])
    return json.loads(open(path, encoding="utf-8").read())


def conversation_texts(name: str) -> dict[str, str]:
    """uid -> the words the no-model baseline reads: every turn of the no-follow-up conversation, each
    prefixed by its role. Both reads see this text (neutral adds only the same follow-up in every row)."""
    dataset, followup = datasets.run(diversity.key(name, "none"))
    return {c.uid: "\n".join(f"{m['role']}: {m['content']}" for m in c.messages)
            for c in datasets.conversations(dataset, followup)}


def score_this_model() -> None:
    print(f"{READER.name} reading the conversations of {CURRENT.name}, layer {READER.probe_layer}", flush=True)
    out = {"model": {"reader": READER.name, "reader_revision": READER.revision, "probe_layer": READER.probe_layer,
                     "conversations_of": CURRENT.name}, "scores": {}}
    for variant in faithfulness.READS:
        out["scores"][variant] = {}
        for name in CATEGORIES:
            s = faithfulness.scores(load(variant, name))
            out["scores"][variant][name] = s
            if s:
                y = s["label"]
                a = faithfulness.auroc(np.array(y), np.array(s["score"])) if 0 < sum(y) < len(y) else None
                print(f"  {variant} {name}: {sum(y)} lies / {len(y) - sum(y)} truthful, in-category AUROC "
                      + ("-" if a is None else f"{a:.4f}"), flush=True)
            else:
                print(f"  {variant} {name}: not trained (too few in a class)", flush=True)
    RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, "gigi").put_result(scores_name(READER), out)


def gaps(substitutes: list[str]) -> None:
    if not IS_SELF_READ:
        raise SystemExit("--gaps runs as the target: unset DYL_READER")
    target = stored_scores(CURRENT.tag)["scores"]
    result = {}
    for tag in substitutes:
        theirs = stored_scores(tag)["scores"]
        result[tag] = {variant: {name: faithfulness.gap(target[variant][name], theirs[variant][name])
                                 for name in CATEGORIES
                                 if target[variant][name] and theirs.get(variant, {}).get(name)}
                       for variant in faithfulness.READS}
    result[faithfulness.WORDS] = {variant: {} for variant in faithfulness.READS}
    for name in CATEGORIES:
        texts = conversation_texts(name)
        for variant in faithfulness.READS:
            if target[variant][name]:
                words = faithfulness.word_scores(target[variant][name], texts)
                result[faithfulness.WORDS][variant][name] = faithfulness.gap(target[variant][name], words)
    names = {tag: MODELS[tag].name.split("/")[-1] for tag in substitutes} | {faithfulness.WORDS: faithfulness.WORDS}
    tables = {variant: faithfulness.markdown({names[t]: g for t, g in result.items()}, variant)
              for variant in faithfulness.READS}
    summaries = {tag: {variant: faithfulness.summary(g[variant]) for variant in faithfulness.READS}
                 for tag, g in result.items()}
    for variant, table in tables.items():
        print(f"\nPROBE FAITHFULNESS GAP, {variant}: target {CURRENT.name}; each cell substitute AUROC (gap [95% CI])\n")
        print(table)
    for tag, s in summaries.items():
        print(f"\n{names[tag]}: " + "; ".join(
            f"{v}: mean target {m['target']:.3f}, substitute {m['substitute']:.3f}, gap {m['gap']:+.3f}"
            f" ± {m['gap_sd']:.3f} (SD across categories) over"
            f" {m['categories']} categories ({m['gap_ci_excludes_zero']} with an interval excluding 0)"
            for v, m in s.items() if m["gap"] is not None), flush=True)
    store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, "gigi")
    store.put_chart(GAPS, faithfulness_charts.draw(result, names, CURRENT.name.split("/")[-1], CAPTION))
    store.put_result(GAPS, {"target": CURRENT.name, "substitutes": {t: MODELS[t].name for t in substitutes},
                            "design": {"bootstrap": faithfulness.BOOTSTRAP, "seed": faithfulness.SEED, "few": 10,
                                       "probe_layers": {t: MODELS[t].probe_layer for t in [CURRENT.tag, *substitutes]}},
                            "gaps": result, "summary": summaries, "markdown": tables})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaps", nargs="+", choices=list(MODELS), help="substitutes to compare with the target")
    args = parser.parse_args()
    gaps(args.gaps) if args.gaps else score_this_model()


if __name__ == "__main__":
    main()
