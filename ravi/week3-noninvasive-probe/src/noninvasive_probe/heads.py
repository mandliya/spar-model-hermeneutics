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


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    nrm = np.linalg.norm(vec)
    return vec / nrm if nrm > 0 else vec


def direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def orthogonal_component(direction: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """Part of `direction` orthogonal to `remove`, oriented to agree with `direction`."""
    d = np.asarray(direction, dtype=np.float64)
    r = _unit(remove)
    out = _unit(d - np.dot(d, r) * r)
    if float(np.dot(out, d)) < 0:
        out = -out
    return out


def mean_unit_directions(directions: list[np.ndarray]) -> np.ndarray:
    """Average of unit vectors, each flipped to agree with the first."""
    acc = None
    ref = None
    for direction in directions:
        unit = _unit(direction)
        if ref is None:
            ref = unit
        elif float(np.dot(unit, ref)) < 0:
            unit = -unit
        acc = unit if acc is None else acc + unit
    if acc is None:
        raise ValueError("no directions")
    return _unit(acc)


def train_ccs_direction(
    honest: np.ndarray,
    deceptive: np.ndarray,
    *,
    steps: int = 400,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Burns et al. CCS. Loss ignores which side is honest.

    Returns (unit direction, dataset mean, dataset std) in the joint z-score
    space. The direction is then flipped so deceptive minus honest is positive.
    That flip is the only step that uses the labels.
    """
    import torch

    honest = np.asarray(honest, dtype=np.float64)
    deceptive = np.asarray(deceptive, dtype=np.float64)
    if len(honest) != len(deceptive) or len(honest) < 2:
        raise ValueError("CCS needs paired honest and deceptive rows")
    both = np.vstack([honest, deceptive])
    mean = both.mean(axis=0)
    std = both.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    h = (honest - mean) / std
    d = (deceptive - mean) / std
    rng = np.random.default_rng(seed)
    swap = rng.random(len(h)) < 0.5
    left = h.copy()
    right = d.copy()
    left[swap], right[swap] = right[swap].copy(), left[swap].copy()
    tl = torch.tensor(left, dtype=torch.float32)
    tr = torch.tensor(right, dtype=torch.float32)
    weight = torch.nn.Parameter(torch.zeros(h.shape[1]))
    bias = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p0 = torch.sigmoid(tl @ weight + bias)
        p1 = torch.sigmoid(tr @ weight + bias)
        consistency = (p0 + p1 - 1).pow(2).mean()
        confidence = torch.minimum(p0, p1).pow(2).mean()
        (consistency + confidence).backward()
        opt.step()
    vec = weight.detach().cpu().numpy().astype(np.float64)
    if float(np.dot(d.mean(axis=0) - h.mean(axis=0), vec)) < 0:
        vec = -vec
    return _unit(vec), mean, std
