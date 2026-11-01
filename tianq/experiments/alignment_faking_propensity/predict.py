"""CPU: can a reader predict, from the prompt alone, how often HAL9000 fakes (or lies) on it?

    ./run.sh propensity predict        # from af-probe-repro, after extract and decide

Only the rates decide.py found to vary across prompts. Each reader is a ridge regression from one model's
activations at the end of the prompt (extract.py) to the prompt's rate: HAL9000 itself (the ceiling, its own
weights), its plain parent and the 8B (substitutes), and the prompt's words alone (TF-IDF). Out-of-fold
predictions from FOLDS-fold cross-validation, averaged over REPEATS shuffles; every fit (scaler, vocabulary,
ridge penalty) sees only its training folds. Each held-out fold's predictions are standardised within the fold,
so only their order inside the fold counts: a heavily penalised ridge predicts little more than its training
folds' mean rate, which runs opposite to the held-out fold's own mean and would score pure noise around -0.5.
Scored by Spearman correlation with the true rates, with 95%
bootstrap intervals over prompts, and each gap (HAL9000 minus the reader) on the same resamples. Stored as
predict.json and pushed.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.alignment_faking_propensity.common import MODELS, ORGANISM, OUT, prompts, push

FOLDS, REPEATS, BOOTSTRAP = 5, 10, 2000
ALPHAS = np.logspace(-2, 6, 17)
WORDS = "words only"


def reader(name: str):
    if name == WORDS:
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), RidgeCV(alphas=ALPHAS))
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def out_of_fold(name: str, x, y: np.ndarray) -> np.ndarray:
    predictions = np.zeros((REPEATS, len(y)))
    for repeat in range(REPEATS):
        for train, test in KFold(FOLDS, shuffle=True, random_state=repeat).split(y):
            model = reader(name).fit(x[train] if name != WORDS else [x[i] for i in train], y[train])
            fold = model.predict(x[test] if name != WORDS else [x[i] for i in test])
            predictions[repeat, test] = (fold - fold.mean()) / (fold.std() or 1.0)
    return predictions.mean(0)


def main() -> None:
    decision = json.loads((OUT / "decision.json").read_text())
    if not decision["predict"]:
        print("neither rate varies across prompts: nothing to predict", flush=True)
        return
    rates = {r["id"]: r for r in json.loads((OUT / "rates.json").read_text())}
    texts = {s["id"]: s["input"] for s in prompts()}
    features = {}
    for name in MODELS:
        saved = torch.load(OUT / "activations" / f"{name}.pt")
        features[name] = dict(zip(saved["ids"], saved["acts"].numpy()))
    results = {"folds": FOLDS, "repeats": REPEATS, "bootstrap": BOOTSTRAP}
    rng = np.random.default_rng(0)
    for target in decision["predict"]:
        ids = [i for i, r in rates.items() if r[target] is not None]
        y = np.array([rates[i][target] for i in ids])
        inputs = {name: np.stack([features[name][i] for i in ids]) for name in MODELS} | {WORDS: [texts[i] for i in ids]}
        predicted = {name: out_of_fold(name, x, y) for name, x in inputs.items()}
        resamples = [rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAP)]
        boot = {name: np.array([spearmanr(p[b], y[b]).statistic for b in resamples]) for name, p in predicted.items()}
        results[target] = {"prompts": len(ids), "readers": {}}
        rho = {name: spearmanr(p, y).statistic for name, p in predicted.items()}
        for name, p in predicted.items():
            entry = {"spearman": rho[name], "ci": np.nanpercentile(boot[name], [2.5, 97.5]).tolist(),
                     "predictions": dict(zip(map(str, ids), p.tolist()))}
            if name != ORGANISM:
                entry |= {"gap": rho[ORGANISM] - rho[name],
                          "gap_ci": np.nanpercentile(boot[ORGANISM] - boot[name], [2.5, 97.5]).tolist()}
            results[target]["readers"][name] = entry
            print(f"{target} | {name}: spearman {rho[name]:.3f} [{entry['ci'][0]:.3f}, {entry['ci'][1]:.3f}]"
                  + (f"; gap {entry['gap']:+.3f} [{entry['gap_ci'][0]:+.3f}, {entry['gap_ci'][1]:+.3f}]"
                     if "gap" in entry else ""), flush=True)
    path = OUT / "predict.json"
    path.write_text(json.dumps(results, indent=1))
    push(path)


if __name__ == "__main__":
    main()
