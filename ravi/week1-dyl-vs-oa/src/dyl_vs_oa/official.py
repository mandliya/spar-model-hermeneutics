from __future__ import annotations

import numpy as np
import torch

from dyl_vs_oa.probes import LayerProbe, ProbeBundle


def load_official_dyl(cfg) -> ProbeBundle:
    """Load AISI's released Gemma-3-27B-IT DYL probe."""
    probe = official_callable(cfg)
    layer = int(getattr(probe, "layer", cfg.dyl.layer))
    coef = _coef_from_official(probe)
    intercept = _intercept_from_official(probe)
    threshold = float(np.asarray(probe.threshold).reshape(-1)[0])
    mean = _buffer_from_official(probe, "dataset_mean")
    std = _buffer_from_official(probe, "dataset_std")
    if mean is None or std is None:
        raise ValueError(
            "Official DYL probe is missing dataset_mean/dataset_std; "
            "scoring raw residuals would collapse every logit."
        )
    return ProbeBundle(
        method="dyl-official",
        layers=[layer],
        probes={
            layer: LayerProbe(
                layer=layer,
                coef=coef,
                intercept=intercept,
                C=cfg.dyl.C,
                train_ce=float("nan"),
                dataset_mean=mean,
                dataset_std=std,
            )
        },
        followup_user=cfg.dyl.followup_user,
        followup_assistant=None,
        extraction="generate",
        model_name=cfg.model_name,
        threshold=threshold,
    )


def score_official_logits(probe, x: np.ndarray) -> np.ndarray:
    """Score activations with the live lie_detectors module (z-score + linear)."""
    with torch.no_grad():
        logits = probe(torch.as_tensor(x, dtype=torch.float32))
    return logits.detach().float().cpu().numpy().reshape(-1)


def _coef_from_official(probe) -> np.ndarray:
    if hasattr(probe, "coef_"):
        coef = np.asarray(probe.coef_)
        return coef.reshape(-1).astype(np.float64)
    if hasattr(probe, "weight"):
        w = probe.weight.detach().float().cpu().numpy()
        return np.asarray(w).reshape(-1).astype(np.float64)
    # Many released probes are nn.Linear in a small module.
    for name in ("linear", "classifier", "probe", "net"):
        inner = getattr(probe, name, None)
        if inner is not None and hasattr(inner, "weight"):
            return inner.weight.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
    if callable(probe):
        raise TypeError(
            "Official probe loaded but coefficients are not exposed; score via probe(activations) instead."
        )
    raise TypeError(f"Unrecognized official probe type: {type(probe)}")


def _buffer_from_official(probe, name: str) -> np.ndarray | None:
    buf = getattr(probe, name, None)
    if buf is None:
        return None
    arr = np.asarray(buf.detach().float().cpu().numpy(), dtype=np.float64).reshape(-1)
    if np.isnan(arr).any():
        return None
    return arr


def _intercept_from_official(probe) -> float:
    if hasattr(probe, "intercept_"):
        return float(np.asarray(probe.intercept_).reshape(-1)[0])
    if hasattr(probe, "bias") and probe.bias is not None:
        return float(probe.bias.detach().float().cpu().numpy().reshape(-1)[0])
    for name in ("linear", "classifier", "probe", "net"):
        inner = getattr(probe, name, None)
        if inner is not None and getattr(inner, "bias", None) is not None:
            return float(inner.bias.detach().float().cpu().numpy().reshape(-1)[0])
    return 0.0


def official_callable(cfg):
    """Return the raw lie_detectors probe for scoring if linear unpacking fails."""
    try:
        from lie_detectors import get_probe
    except ImportError as exc:
        raise ImportError(
            "Install lie-detectors to load the official DYL probe: pip install lie-detectors"
        ) from exc

    return get_probe(cfg.dyl.official_repo, filename=cfg.dyl.official_filename)
