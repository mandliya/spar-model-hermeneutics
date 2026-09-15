from __future__ import annotations

import numpy as np

from dyl_vs_oa.probes import LayerProbe, ProbeBundle


def load_official_dyl(cfg) -> ProbeBundle:
    """Load AISI's released Gemma-3-27B-IT DYL probe."""
    try:
        from lie_detectors import get_probe
    except ImportError as exc:
        raise ImportError("Install lie-detectors to load the official DYL probe: pip install lie-detectors") from exc

    probe = get_probe(cfg.dyl.official_repo, filename=cfg.dyl.official_filename)
    layer = int(getattr(probe, "layer", cfg.dyl.layer))
    coef = _coef_from_official(probe)
    intercept = _intercept_from_official(probe)
    threshold = float(getattr(probe, "threshold", 0.0))
    return ProbeBundle(
        method="dyl-official",
        layers=[layer],
        probes={
            layer: LayerProbe(layer=layer, coef=coef, intercept=intercept, C=cfg.dyl.C, train_ce=float("nan"))
        },
        followup_user=cfg.dyl.followup_user,
        followup_assistant=None,
        extraction="generate",
        model_name=cfg.model_name,
        threshold=threshold,
    )


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
    from lie_detectors import get_probe

    return get_probe(cfg.dyl.official_repo, filename=cfg.dyl.official_filename)
