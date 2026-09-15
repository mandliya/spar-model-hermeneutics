from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n = max(len(a), len(b))
    if n == 0:
        return float("nan")
    qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)
    qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b)), b)
    return float(np.mean(np.abs(qa - qb)))


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    if np.all(fpr > target_fpr):
        return float(tpr[0])
    return float(np.interp(target_fpr, fpr, tpr))


def summarize(
    y_true: np.ndarray,
    scores: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    preds = (scores >= decision_threshold).astype(int)
    honest = scores[y_true == 0]
    deceptive = scores[y_true == 1]
    out: dict[str, float] = {
        "n": float(len(y_true)),
        "n_honest": float((y_true == 0).sum()),
        "n_deceptive": float((y_true == 1).sum()),
        "mean_honest": float(honest.mean()) if len(honest) else float("nan"),
        "mean_deceptive": float(deceptive.mean()) if len(deceptive) else float("nan"),
        "w1": wasserstein_1d(deceptive, honest) if len(honest) and len(deceptive) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, scores, 0.01),
        "decision_threshold": float(decision_threshold),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, scores))
    else:
        out["auc"] = float("nan")
    return out


def check_against_target(metrics: dict[str, float], target: dict, auc_tol: float = 0.03) -> dict:
    result = {"ok": True, "notes": []}
    if "auc" in target and "auc" in metrics:
        delta = abs(metrics["auc"] - target["auc"])
        result["auc_delta"] = delta
        if delta > auc_tol:
            result["ok"] = False
            result["notes"].append(
                f"AUC {metrics['auc']:.3f} vs paper {target['auc']:.3f} (Δ={delta:.3f} > {auc_tol})"
            )
        else:
            result["notes"].append(
                f"AUC {metrics['auc']:.3f} matches paper {target['auc']:.3f} within {auc_tol}"
            )
    return result


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
