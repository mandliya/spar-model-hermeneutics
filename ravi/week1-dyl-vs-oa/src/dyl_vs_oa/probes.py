from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass
class LayerProbe:
    layer: int
    coef: np.ndarray
    intercept: float
    C: float
    train_ce: float
    dataset_mean: np.ndarray | None = None
    dataset_std: np.ndarray | None = None

    def standardize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.dataset_mean is None or self.dataset_std is None:
            return x
        return (x - self.dataset_mean) / self.dataset_std

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.standardize(x) @ self.coef + self.intercept

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(self.logits(x), -30, 30)
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class ProbeBundle:
    """OA-style: one logistic probe per kept layer, score = mean sigmoid.

    DYL-style: a single layer (len(layers) == 1).
    """

    method: str
    layers: list[int]
    probes: dict[int, LayerProbe]
    followup_user: str
    followup_assistant: str | None
    extraction: str  # "prefill" | "generate"
    model_name: str
    threshold: float | None = None

    def logits(self, acts_by_layer: dict[int, np.ndarray]) -> np.ndarray:
        outs = [self.probes[layer].logits(acts_by_layer[layer]) for layer in self.layers]
        return np.mean(np.stack(outs, axis=0), axis=0)

    def score(self, acts_by_layer: dict[int, np.ndarray]) -> np.ndarray:
        probs = []
        for layer in self.layers:
            probs.append(self.probes[layer].predict_proba(acts_by_layer[layer]))
        return np.mean(np.stack(probs, axis=0), axis=0)


def train_layer_probe(
    x: np.ndarray,
    y: np.ndarray,
    layer: int,
    C: float,
    *,
    standardize: bool = False,
    solver: str = "lbfgs",
) -> LayerProbe:
    x = np.asarray(x, dtype=np.float64)
    mean = std = None
    if standardize:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        x = (x - mean) / std
        solver = "liblinear"
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        solver=solver,
    )
    clf.fit(x, y)
    proba = clf.predict_proba(x)[:, 1]
    ce = float(log_loss(y, proba, labels=[0, 1]))
    return LayerProbe(
        layer=layer,
        coef=clf.coef_[0].astype(np.float64),
        intercept=float(clf.intercept_[0]),
        C=C,
        train_ce=ce,
        dataset_mean=mean,
        dataset_std=std,
    )


def train_oa_bundle(
    acts_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    *,
    C: float,
    ce_keep_below: float,
    followup_user: str,
    followup_assistant: str,
    model_name: str,
    standardize: bool = False,
    solver: str = "lbfgs",
) -> ProbeBundle:
    probes: dict[int, LayerProbe] = {}
    kept: list[int] = []
    for layer in sorted(acts_by_layer):
        probe = train_layer_probe(
            acts_by_layer[layer], y, layer, C, standardize=standardize, solver=solver
        )
        probes[layer] = probe
        if probe.train_ce < ce_keep_below:
            kept.append(layer)
    if not kept:
        # Fall back to the lowest-CE half of layers rather than failing silently.
        ranked = sorted(probes.values(), key=lambda p: p.train_ce)
        kept = [p.layer for p in ranked[: max(1, len(ranked) // 2)]]
    return ProbeBundle(
        method="oa",
        layers=kept,
        probes=probes,
        followup_user=followup_user,
        followup_assistant=followup_assistant,
        extraction="prefill",
        model_name=model_name,
    )


def train_dyl_bundle(
    x: np.ndarray,
    y: np.ndarray,
    *,
    layer: int,
    C: float,
    followup_user: str,
    model_name: str,
) -> ProbeBundle:
    probe = train_layer_probe(x, y, layer, C)
    return ProbeBundle(
        method="dyl",
        layers=[layer],
        probes={layer: probe},
        followup_user=followup_user,
        followup_assistant=None,
        extraction="generate",
        model_name=model_name,
    )


def save_bundle(bundle: ProbeBundle, path) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": bundle.method,
        "layers": bundle.layers,
        "followup_user": bundle.followup_user,
        "followup_assistant": bundle.followup_assistant,
        "extraction": bundle.extraction,
        "model_name": bundle.model_name,
        "threshold": bundle.threshold,
        "probes": {
            str(layer): {
                "layer": p.layer,
                "coef": p.coef.tolist(),
                "intercept": p.intercept,
                "C": p.C,
                "train_ce": p.train_ce,
                "dataset_mean": None if p.dataset_mean is None else p.dataset_mean.tolist(),
                "dataset_std": None if p.dataset_std is None else p.dataset_std.tolist(),
            }
            for layer, p in bundle.probes.items()
        },
    }
    path.write_text(json.dumps(payload))


def load_bundle(path) -> ProbeBundle:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    probes = {}
    for k, v in payload["probes"].items():
        mean = v.get("dataset_mean")
        std = v.get("dataset_std")
        probes[int(k)] = LayerProbe(
            layer=int(v["layer"]),
            coef=np.array(v["coef"], dtype=np.float64),
            intercept=float(v["intercept"]),
            C=float(v["C"]),
            train_ce=float(v["train_ce"]),
            dataset_mean=None if mean is None else np.array(mean, dtype=np.float64),
            dataset_std=None if std is None else np.array(std, dtype=np.float64),
        )
    return ProbeBundle(
        method=payload["method"],
        layers=[int(x) for x in payload["layers"]],
        probes=probes,
        followup_user=payload["followup_user"],
        followup_assistant=payload.get("followup_assistant"),
        extraction=payload["extraction"],
        model_name=payload["model_name"],
        threshold=payload.get("threshold"),
    )
