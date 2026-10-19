"""Training a DYL probe the way AISI trained theirs, as closely as public information allows.

AISI's training code is private: the lie_detectors package is "load + inference
only" (probes.py L22). What is public, and followed here:

    reads         hidden_states[layer + 1] at the token before the yes/no answer
                  (core/probe.py, measured on AISI's own training transcripts)
    architecture  logistic regression over standardised activations, as the
                  library's LogisticRegressionProbe, which train() returns:
                  https://github.com/UKGovernmentBEIS/lie_detectors/blob/88043087688308bc9cefe7cbdbf9b3806a679149/src/lie_detectors/probes.py#L90-L130
    standardise   per dimension, by the training rows' mean and population std,
                  stored as dataset_mean / dataset_std like the released checkpoint
    hyperparams   layer 58 and l2_lambda 0.01: the released checkpoint's own
    threshold     flags 1% of honest calibration rows (model card L56: "Threshold
                  set at 1% FPR on Alpaca")

Not public: how l2_lambda enters the loss. Fit on AISI's training split
(dyl_train_city_countries, 4,403 labelled rows), the cosine with the released
weights peaks at "mean log-loss + l2_lambda (|w|^2 + b^2)" exactly at their
l2_lambda (0.906 over a scan of 25 penalty strengths, both bias choices and both
standardisations; nothing else is closer), and that reading also reproduces
their near-zero bias, so it is the objective here. The remaining gap is neither
lambda nor standardisation: the released weights are about 4x larger than this
objective's optimum, as if their optimiser stopped before converging. Downstream
the two agree, which is the bar the project set: on AISI's Varied Deception the
probe this trains on that split scores AUROC 0.986 against the released probe's
0.986, ranks rows with Spearman 0.98, and flags the same way on 96% of rows.
"""

from __future__ import annotations

import numpy as np
import torch
from lie_detectors import LogisticRegressionProbe, ProbeArchitecture, TrainHyperparameters
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dyl_repro.core.probe import CALIBRATION_FPR, PROBE_POSITION, DylProbe
from dyl_repro.core.model import READER
# The released checkpoint's own hyperparams (its stored `hyperparams`); the layer is the reader's (58 for Gemma).
DEFAULT_LAYER = READER.probe_layer
DEFAULT_L2_LAMBDA = 0.01
# Fit to convergence, so the probe is the objective's optimum and a refit reproduces it.
MAX_ITERATIONS = 20_000
GRADIENT_TOLERANCE = 1e-9
CROSS_VALIDATION_FOLDS = 5
# Fewest rows of either class a probe is trained on: two per held-out fold, so
# the cross-validated reference is not computed on a fold with one lie or none.
MIN_ROWS_PER_CLASS = 2 * CROSS_VALIDATION_FOLDS


def probe_name(train_key: str, layer: int, l2_lambda: float, position: str = PROBE_POSITION) -> str:
    """Named like AISI's sweep entries (l_58_lm_0_01_ar_lr), prefixed by its training data,
    and by its read position when that is not AISI's."""
    at = "" if position == PROBE_POSITION else f"_at_{position}"
    return f"trained_on_{train_key.replace('/', '-')}{at}_l_{layer}_lm_{str(l2_lambda).replace('.', '_')}_ar_lr"


def fit_weights(activations: np.ndarray, labels: np.ndarray, l2_lambda: float,
                sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(mean, std, weight, bias) minimising mean log-loss + l2_lambda (|w|^2 + b^2) over the activations
    standardised by their own mean and population std; both means weighted by `sample_weight` if given."""
    x, p = np.asarray(activations, dtype=np.float64), _normalised(sample_weight)
    y = np.asarray(labels, dtype=np.float64)
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError(f"training needs both labels, 0 and 1; got {np.unique(y).tolist()}")
    mean, std = (x.mean(0), x.std(0)) if p is None else _weighted_moments(x, p)
    z = (x - mean) / std
    n, d = z.shape

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:d], theta[d]
        logits = z @ w + b
        residual = (expit(logits) - y) / n if p is None else (expit(logits) - y) * p
        value = _mean(np.logaddexp(0, logits) - y * logits, p) + l2_lambda * (w @ w + b * b)
        gradient = np.concatenate([z.T @ residual + 2 * l2_lambda * w, [residual.sum() + 2 * l2_lambda * b]])
        return value, gradient

    fit = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": MAX_ITERATIONS, "gtol": GRADIENT_TOLERANCE})
    if not fit.success:
        raise RuntimeError(f"probe fit did not converge: {fit.message}")
    return mean, std, fit.x[:d], float(fit.x[d])


# Helpers of fit_weights, kept below it so its loss stays at the line the report links to.
def _normalised(sample_weight: np.ndarray | None) -> np.ndarray | None:
    """Weights summing to 1, or None for the unweighted fit (left exactly as it was)."""
    if sample_weight is None:
        return None
    p = np.asarray(sample_weight, dtype=np.float64)
    if (p < 0).any() or p.sum() <= 0:
        raise ValueError("sample weights must be non-negative and not all zero")
    return p / p.sum()


def _weighted_moments(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = p @ x
    return mean, np.sqrt(p @ (x - mean) ** 2)


def _mean(values: np.ndarray, p: np.ndarray | None) -> float:
    return values.mean() if p is None else p @ values


def train(activations: np.ndarray, labels: np.ndarray, *, layer: int, l2_lambda: float,
          trained_on: str, position: str = PROBE_POSITION) -> DylProbe:
    """A logistic-regression DYL probe in the library's own class, threshold not yet set."""
    mean, std, weight, bias = fit_weights(activations, labels, l2_lambda)
    module = LogisticRegressionProbe(
        d_model=len(mean),
        hyperparams=TrainHyperparameters(
            layer=layer, l2_lambda=l2_lambda, train_dataset=trained_on,
            probe_architecture=ProbeArchitecture.LOGISTIC_REGRESSION, position=position,
        ),
    )
    with torch.no_grad():
        module.dataset_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        module.dataset_std.copy_(torch.as_tensor(std, dtype=torch.float32))
        module.linear.weight.copy_(torch.as_tensor(weight, dtype=torch.float32)[None])
        module.linear.bias.fill_(bias)
    module.eval()
    return DylProbe(probe_name(trained_on, layer, l2_lambda, position), module, trained_on=trained_on, position=position)


def calibrate(probe: DylProbe, honest_activations: np.ndarray, fpr: float = CALIBRATION_FPR) -> float:
    """Set the threshold to the score that flags `fpr` of the honest calibration rows.

    The (1 - fpr) quantile, numpy's default interpolation. AISI's exact rule is not
    public: for the released probe this rule gives 4.90 where its stored threshold is
    5.05, and both flag 1.0% of AISI's Alpaca rows."""
    threshold = float(np.quantile(probe.score(honest_activations), 1 - fpr))
    probe.module.threshold.fill_(threshold)
    return threshold


def weight_cosine(a: DylProbe, b: DylProbe) -> float:
    """Cosine between two probes' weight vectors, each over its own standardised inputs."""
    wa, wb = (p.module.linear.weight.detach().double().numpy().ravel() for p in (a, b))
    return float(wa @ wb / (np.linalg.norm(wa) * np.linalg.norm(wb)))


def out_of_fold_scores(activations: np.ndarray, labels: np.ndarray, *, l2_lambda: float,
                       folds: int = CROSS_VALIDATION_FOLDS, seed: int = 0) -> np.ndarray:
    """The in-distribution reference for a transfer result: every row scored by a probe
    trained without it (stratified k-fold, each class split across the folds)."""
    y = np.asarray(labels)
    scores = np.empty(len(y))
    for train_rows, test_rows in StratifiedKFold(folds, shuffle=True, random_state=seed).split(activations, y):
        mean, std, weight, bias = fit_weights(activations[train_rows], y[train_rows], l2_lambda)
        scores[test_rows] = ((activations[test_rows] - mean) / std) @ weight + bias
    return scores


def cross_validated_auroc(labels: np.ndarray, scores: np.ndarray, *, folds: int = CROSS_VALIDATION_FOLDS,
                          seed: int = 0) -> dict:
    """The AUROC of out_of_fold_scores, with the counts and split it came from."""
    y = np.asarray(labels)
    return {"auroc": float(roc_auc_score(y, scores)), "n_lie": int(y.sum()), "n_honest": int((y == 0).sum()),
            "folds": folds, "seed": seed}
