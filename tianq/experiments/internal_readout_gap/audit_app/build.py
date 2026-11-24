"""Build the data for the audit page: every restart point with each model's readouts and each method's own score for it.

    READOUT_OUT=<folder from shared/fetch.py> python build.py --to <folder>      # then open <folder>/index.html

For ask 1 (is the distilled gemma's readout closer to Qwen3.6's?). Per restart point and per gemma version it
recomputes, with the experiment's own functions, exactly what each of the three methods scored:
  identical words   similarity.py: Jaccard of the top-200 J-lens words at each depth, and cosine of NLA word counts;
                    same restart point, and chance = the mean against the version at every other restart point
  meaning           meaning_similarity.py: soft match of the J-lens words and cosine of the mean NLA sentence vector;
                    chance as that script takes it. Also the best-matching word pairs, to see what the score rests on
  judge             judge_similarity.py: the saved ratings (same restart point, and its one other restart point), with
                    the prompt rebuilt from that script's template and inputs (the sent prompt itself was not saved)
and writes <folder>/index.html, data/index.js (one line per restart point) and data/rows/<n>.js (everything for
one). Links point at the files on HF at a fixed revision. Nothing here is a new result: it only lays the saved ones out.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "distillation_similarity"))
import judge_similarity as js  # noqa: E402
import meaning_similarity as ms  # noqa: E402
import spots  # noqa: E402
from shared import models, readout, words  # noqa: E402
from shared.embedding import Embedder  # noqa: E402
from shared.judge import READOUT_KINDS  # noqa: E402
from similarity import EXPERIMENT, ORIGINAL, VERSIONS, against, lens_lists, load, nla_counts  # noqa: E402

READS_REVISION = "5cd560300e5cd42d27a10d1e8a9d75f4aab46b7c"      # model-hermeneutics/dyl-probe-repro
TRANSCRIPTS_REVISION = "1aac2780007e0dc34553d0096142e2c3f3d98e57"  # model-hermeneutics/gigi-forensics-transcripts
READS = f"https://huggingface.co/datasets/model-hermeneutics/dyl-probe-repro/blob/{READS_REVISION}/internal-readout-gap/{EXPERIMENT}/outputs"
TRANSCRIPTS = f"https://huggingface.co/datasets/model-hermeneutics/gigi-forensics-transcripts/blob/{TRANSCRIPTS_REVISION}"
DEPTHS = [f"{d:.3f}" for d in readout.DEPTHS]
JUDGED = ("NLA sentences", f"J-lens words | depth {readout.HEADLINE:.3f}", "J-lens words | depth 0.950")
PAIRS = 15


def r4(x) -> float | None:
    return None if x is None or np.isnan(x) else round(float(x), 4)


def best_pairs(a_words: list[str], b_words: list[str], a: np.ndarray, b: np.ndarray) -> list:
    """The closest non-identical word pairs between two lists: [Qwen3.6's word, the gemma's word, cosine]."""
    sims = a @ b.T
    out = []
    for i in np.argsort(-sims.max(1)):
        j = int(sims[i].argmax())
        if a_words[i] != b_words[j]:
            out.append([a_words[i], b_words[j], round(float(sims[i, j]), 3)])
        if len(out) == PAIRS:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    args = ap.parse_args()
    dest = pathlib.Path(args.to)
    (dest / "data" / "rows").mkdir(parents=True, exist_ok=True)
    tags = [ORIGINAL, *VERSIONS]
    lens = {t: load(t, "lens") for t in tags}
    nla = {t: load(t, "nla") for t in tags}
    common = sorted(set.intersection(*(set(lens[t]) & set(nla[t]) for t in tags)))
    plan = {s.name: s for s in spots.spots()}
    behaviour = spots.behaviour()
    first_run = {}
    for row in csv.DictReader(open(spots.FORENSICS / "results" / "hero9_swap_per_run.csv", encoding="utf-8")):
        first_run.setdefault(row["point"], pathlib.PurePosixPath(row["run"]).relative_to("results/raw").as_posix())
    print(f"{len(common)} restart points read by every model", flush=True)

    # identical words
    scores: dict = {n: {v: {"words": {}, "meaning": {}, "judge": {}} for v in VERSIONS} for n in common}
    lists = {d: {t: lens_lists(lens[t], float(d)) for t in tags} for d in DEPTHS}
    for d in DEPTHS:
        o = {n: set(w) for n, w in lists[d][ORIGINAL].items()}
        for v in VERSIONS:
            same, chance = against(o, {n: set(w) for n, w in lists[d][v].items()}, common, words.jaccard)
            for n, s, c in zip(common, same, chance):
                scores[n][v]["words"][d] = [r4(s), r4(c)]
    o = nla_counts(nla[ORIGINAL])
    for v in VERSIONS:
        same, chance = against(o, nla_counts(nla[v]), common, words.cosine)
        for n, s, c in zip(common, same, chance):
            scores[n][v]["words"]["nla"] = [r4(s), r4(c)]
    print("identical words done", flush=True)

    # meaning
    emb = Embedder()
    emb.add({e for t in tags for n in common for e in ms.explanations(nla[t], n)})
    vectors = {t: np.stack([emb.mean(ms.explanations(nla[t], n)) for n in common]) for t in tags}
    for v in VERSIONS:
        same, chance = ms.centroid_scores(vectors[ORIGINAL], vectors[v])
        for n, s, c in zip(common, same, chance):
            scores[n][v]["meaning"]["nla"] = [r4(s), r4(c)]
    pairs: dict = {n: {v: {} for v in VERSIONS} for n in common}
    for d in DEPTHS:
        emb.add({w for t in tags for n in common for w in lists[d][t][n]})
        per_spot = {t: [emb.matrix(lists[d][t][n]) for n in common] for t in tags}
        for v in VERSIONS:
            same, chance = ms.soft_scores(per_spot[ORIGINAL], per_spot[v], np.random.default_rng(ms.SEED))
            for i, (n, s, c) in enumerate(zip(common, same, chance)):
                scores[n][v]["meaning"][d] = [r4(s), r4(c)]
                pairs[n][v][d] = best_pairs(lists[d][ORIGINAL][n], lists[d][v][n], per_spot[ORIGINAL][i], per_spot[v][i])
        print(f"meaning at depth {d} done", flush=True)

    # judge: the saved answers, and where each one is
    saved: dict = {}
    for path in sorted((models.OUT / EXPERIMENT / "judge_similarity").glob("*.jsonl")):
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if line.strip():
                row = json.loads(line)
                saved[row["key"]] = {"rating": r4(row["rating"]), "answer": row["answer"], "link": f"{READS}/judge_similarity/{path.name}#L{line_no}"}
    order = np.random.default_rng(js.SEED).permutation(len(common))
    other = {common[i]: common[j] for i, j in zip(order, np.roll(order, 1))}
    text = {kind: js.readouts(kind, lens, nla, common) for kind in JUDGED}
    for n in common:
        for v in VERSIONS:
            for kind in JUDGED:
                what = READOUT_KINDS[kind.split(" | ")[0]]
                entry = {}
                for pairing in ("same", "other"):
                    a, b = text[kind][ORIGINAL][n], text[kind][v][n if pairing == "same" else other[n]]
                    entry[pairing] = saved.get(f"{kind} | {v} || {n} || {pairing}", {"rating": None, "answer": None, "link": None}) | {
                        "prompt": js.PROMPT.format(what=what, a=a, b=b) if a and b else None}
                scores[n][v]["judge"][kind] = entry
    print(f"judge: {len(saved)} saved answers placed", flush=True)

    index = []
    for i, n in enumerate(common):
        s = plan[n]
        layers = {t: {d: readout.layer_at(lens[t][n]["n_layers"], [int(k) for k in lens[t][n]["layers"]], float(d)) for d in DEPTHS}
                  for t in tags}
        row = {"name": n, "env": s.env, "src": s.src, "K": s.K, "N": s.N, "take": s.take, "incident": s.incident,
               "held_out": s.held_out, "behaviour": behaviour.get(n, {}), "other": other[n], "other_index": common.index(other[n]),
               "tokens": {t: lens[t][n]["tokens"] for t in tags}, "layers": layers,
               "links": {"transcript": f"{TRANSCRIPTS}/{first_run[n]}/messages.json" if n in first_run else None,
                         "lens": {t: f"{READS}/{t}/lens/{n}.json" for t in tags},
                         "nla": {t: f"{READS}/{t}/nla/{n}.json" for t in tags}},
               "jspace": {t: {d: lists[d][t][n] for d in DEPTHS} for t in tags},
               "own_next": {t: words.lens_words(lens[t][n]["model"])[:40] for t in tags},
               "nla": {t: [x["explanation"].strip() for x in nla[t][n]["positions"]["before_action"]] for t in tags},
               "scores": scores[n], "pairs": pairs[n]}
        (dest / "data" / "rows" / f"{i}.js").write_text("window.ROW(" + json.dumps(row, ensure_ascii=False) + ");", encoding="utf-8")
        base, final = scores[n][VERSIONS[0]], scores[n][VERSIONS[-1]]
        def above(x):
            return None if x[0] is None or x[1] is None else round(x[0] - x[1], 4)
        index.append({"i": i, "name": n, "env": s.env, "src": s.src, "take": s.take, "held_out": s.held_out,
                      "base": {d: above(base["words"][d]) for d in [*DEPTHS, "nla"]},
                      "final": {d: above(final["words"][d]) for d in [*DEPTHS, "nla"]}})
    meta = {"original": ORIGINAL, "versions": VERSIONS, "depths": DEPTHS, "headline": f"{readout.HEADLINE:.3f}", "judged": list(JUDGED),
            "judge": js.JUDGE if hasattr(js, "JUDGE") else None, "embedder": ms.EMBEDDER, "reads_revision": READS_REVISION,
            "transcripts_revision": TRANSCRIPTS_REVISION, "top_words_judge": js.TOP_WORDS, "chance_spots_soft": ms.CHANCE_SPOTS}
    (dest / "data" / "index.js").write_text("window.AUDIT=" + json.dumps({"meta": meta, "rows": index}, ensure_ascii=False) + ";", encoding="utf-8")
    shutil.copy(HERE / "index.html", dest / "index.html")
    print(f"wrote {dest / 'index.html'} and {len(index)} rows", flush=True)


if __name__ == "__main__":
    main()
