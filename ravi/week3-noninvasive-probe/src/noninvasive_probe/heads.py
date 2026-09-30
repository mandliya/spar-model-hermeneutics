"""Mean-difference and LAT heads. Logistic stays week-1 `train_oa_bundle`."""

from __future__ import annotations

import numpy as np

from noninvasive_probe.paths import bootstrap

bootstrap()

from dyl_vs_oa.probes import LayerProbe, ProbeBundle


def train_mean_diff_bundle(
    acts_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    *,
    followup_user: str = "",
    model_name: str,
    followup_assistant: str | None = None,
    extraction: str = "original_turn",
    method: str = "mean_diff",
) -> ProbeBundle:
    y = np.asarray(y)
    probes: dict[int, LayerProbe] = {}
    for layer in sorted(acts_by_layer):
        x = np.asarray(acts_by_layer[layer], dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        z = (x - mean) / std
        coef = z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0)
        nrm = np.linalg.norm(coef)
        if nrm > 0:
            coef = coef / nrm
        probes[layer] = LayerProbe(
            layer=layer,
            coef=coef,
            intercept=0.0,
            C=0.0,
            train_ce=float("nan"),
            dataset_mean=mean,
            dataset_std=std,
        )
    return ProbeBundle(
        method=method,
        layers=sorted(probes),
        probes=probes,
        followup_user=followup_user,
        followup_assistant=followup_assistant,
        extraction=extraction,
        model_name=model_name,
    )


def train_lat_direction(honest: np.ndarray, deceptive: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    diff = np.asarray(deceptive, dtype=np.float64) - np.asarray(honest, dtype=np.float64)
    if len(diff) < 2:
        vec = diff.mean(axis=0)
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm > 0 else vec
    pca = PCA(n_components=1)
    pca.fit(diff)
    vec = pca.components_[0]
    if float(np.dot(diff.mean(axis=0), vec)) < 0:
        vec = -vec
    return vec.astype(np.float64)


def train_lat_bundle(
    acts_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    *,
    followup_user: str = "",
    model_name: str,
    followup_assistant: str | None = None,
    extraction: str = "original_turn",
) -> ProbeBundle:
    y = np.asarray(y)
    honest_ix = np.flatnonzero(y == 0)
    deceptive_ix = np.flatnonzero(y == 1)
    n = min(len(honest_ix), len(deceptive_ix))
    probes: dict[int, LayerProbe] = {}
    for layer in sorted(acts_by_layer):
        x = np.asarray(acts_by_layer[layer], dtype=np.float64)
        vec = train_lat_direction(x[honest_ix[:n]], x[deceptive_ix[:n]])
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        probes[layer] = LayerProbe(
            layer=layer,
            coef=vec,
            intercept=0.0,
            C=0.0,
            train_ce=float("nan"),
            dataset_mean=mean,
            dataset_std=std,
        )
    return ProbeBundle(
        method="lat",
        layers=sorted(probes),
        probes=probes,
        followup_user=followup_user,
        followup_assistant=followup_assistant,
        extraction=extraction,
        model_name=model_name,
    )
