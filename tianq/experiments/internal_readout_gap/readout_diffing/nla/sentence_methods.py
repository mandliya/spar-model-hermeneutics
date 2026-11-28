"""New ways of scoring NLA sentences, through the same test as yardstick.py (26 Nov; PLAN of gender_secret_rerun, step 0).

    READOUT_OUT=<folder from shared/fetch.py> python sentence_methods.py

yardstick.py ranked two word methods on how often the same restart point scores above a different restart point of
the same task (a tie counts half): plain word counts 0.750 and rare words counting more 0.819, for base gemma against
Qwen3.6. Those are not run again. This runs the methods added since, on the same restart points:
  13  short phrases, rare phrases counting more                (readout_methods/word_lists/nla_sentences.py)
  14  agreement across the five samples, rare words more       (readout_methods/word_lists/nla_sentences.py)
  15  best-matching sentence                                   (readout_methods/likeness/sentence_match.py)
  16  meaning vectors: the mean of the sentences' embeddings   (readout_methods/likeness/meaning.py)
  17  the same with the pool's average sentence removed first
Prints every number it writes; writes READOUT_OUT/distillation_similarity/nla_sentence_methods.json.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: method_menu, restart_points
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods, shared
from readout_methods.likeness import evaluation, meaning, rare_words, sentence_match  # noqa: E402
from readout_methods.word_lists import nla_sentences  # noqa: E402
from readouts import BASE, FINAL, MODELS, ORIGINAL, load_counts, table  # noqa: E402  (puts the experiment's root on the path)
from restart_points import tasks  # noqa: E402
from shared import words  # noqa: E402
from shared.embedding import Embedder  # noqa: E402

POSITION = "before_action"


def meaning_tables(texts: dict[str, list[list[str]]]) -> dict[str, dict[str, np.ndarray]]:
    """label -> version -> table, for the two meaning methods. "Average removed" takes the mean vector of every
    model's restart points off before the cosine."""
    emb = Embedder()
    emb.add({s for model in texts.values() for point in model for s in point})
    vectors = {t: meaning.sentence_centroids(emb, model) for t, model in texts.items()}
    centre = np.concatenate(list(vectors.values())).mean(0)
    return {label: {version: meaning.centred_cosine(vectors[ORIGINAL], vectors[version], shift) for version in (BASE, FINAL)}
            for label, shift in (("16 meaning: mean of the sentences' vectors", 0.0), ("17 meaning: the same, average sentence removed", centre))}


def main() -> None:
    nla, counts, names = load_counts()
    names = [x for x in names if all(counts[t][x] for t in counts)]   # yardstick.py's restart points
    n = len(names)
    mask = evaluation.others(n) & evaluation.same_task(tasks(names))
    texts = {t: [[s["explanation"] for s in nla[t][x]["positions"][POSITION] if s["explanation"].strip()] for x in names] for t in MODELS}
    split = {t: [[words.text_words([s]) for s in point] for point in model] for t, model in texts.items()}
    split = {t: [[s for s in point if s] for point in model] for t, model in split.items()}
    print(f"{n} restart points with NLA sentences from all three models", flush=True)
    by_method: dict[str, dict[str, np.ndarray]] = {}
    for label, build in (("13 short phrases (two and three words), rare phrases count more", nla_sentences.phrase_readouts),
                         ("14 agreement across the samples, rare words count more", nla_sentences.agreement_readouts)):
        readouts = {t: build(split[t]) for t in MODELS}
        by_method[label] = {version: table(rare_words.rarity_cosine, readouts[ORIGINAL], readouts[version], list(readouts.values()),
                                           top=None, value="p") for version in (BASE, FINAL)}
    by_method["15 best-matching sentence, rare words count more"] = {
        version: sentence_match.best_match(split[ORIGINAL], split[version], list(split.values())) for version in (BASE, FINAL)}
    by_method |= meaning_tables(texts)
    result = {"restart_points": n, "methods": {}}
    for label, by_version in by_method.items():
        for version, t in by_version.items():
            strict, ties = evaluation.same_moment_rates(t, mask)
            rate = strict + ties / 2
            result["methods"][f"{label} | {version}"] = {"same_task_rate": rate, "ties": ties, "same": float(np.nanmean(np.diag(t))),
                                                         "chance": float(np.nanmean(evaluation.chance(t)))}
            print(f"{label} | {version} against {ORIGINAL}: same restart point scores above a different one of the same task "
                  f"{rate:.3f} of the time (ties {ties:.3f}); same {np.nanmean(np.diag(t)):.4f}, chance "
                  f"{np.nanmean(evaluation.chance(t)):.4f}", flush=True)
    path = pathlib.Path(os.environ["READOUT_OUT"]) / "distillation_similarity" / "nla_sentence_methods.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
