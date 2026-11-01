"""CPU: do HAL9000's faking and lie rates differ across prompts by more than sampling noise?

    ./run.sh propensity decide        # from af-probe-repro, after judge

For each rate, the test of homogeneity: Pearson's chi-square of the per-prompt counts against one rate shared
by every prompt, with its p-value from SIMULATIONS draws of binomial counts at that shared rate (the counts are
small, so no asymptotic chi-square). And the spread: the standard deviation of the prompts' true rates, the
observed variance minus the part binomial noise explains. A rate varies when p < ALPHA; only those go on to
predict.py. Judge noise flips labels at random, which shrinks real differences but cannot create them.
Stored as decision.json and pushed.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.alignment_faking_propensity.common import OUT, push

ALPHA = 0.05
SIMULATIONS = 10_000
TARGETS = {"faking_rate": ("faking", "labelled"), "lie_rate": ("lies", "answered")}


def chi_square(k: np.ndarray, n: np.ndarray, p: float) -> np.ndarray:
    return ((k - n * p) ** 2 / (n * p * (1 - p))).sum(-1)


def homogeneity(k: np.ndarray, n: np.ndarray) -> dict:
    p = k.sum() / n.sum()
    if p in (0, 1):
        return {"pooled_rate": p, "p_value": 1.0, "varies": False, "note": "every rollout has the same label"}
    observed = chi_square(k, n, p)
    simulated = chi_square(np.random.default_rng(0).binomial(n, p, size=(SIMULATIONS, len(n))), n, p)
    rates = k / n
    true_variance = rates.var(ddof=1) - np.mean(p * (1 - p) / n)
    p_value = (1 + (simulated >= observed).sum()) / (1 + SIMULATIONS)
    return {"pooled_rate": p, "chi_square": observed, "dispersion": observed / (len(n) - 1), "p_value": p_value,
            "varies": bool(p_value < ALPHA), "true_rate_sd": float(np.sqrt(max(true_variance, 0.0))),
            "observed_rate_sd": float(rates.std(ddof=1)), "binomial_sd": float(np.sqrt(np.mean(p * (1 - p) / n))),
            "rate_quantiles": dict(zip(("min", "q10", "q25", "median", "q75", "q90", "max"),
                                       np.quantile(rates, [0, .1, .25, .5, .75, .9, 1]).round(3).tolist())),
            "prompts_at_0": int((k == 0).sum()), "prompts_at_1": int((k == n).sum())}


def main() -> None:
    rates = json.loads((OUT / "rates.json").read_text())
    decision = {"alpha": ALPHA, "simulations": SIMULATIONS, "prompts": len(rates)}
    for target, (hits, total) in TARGETS.items():
        kept = [r for r in rates if r[total]]  # a prompt with no labelled (or answered) rollout has no rate
        k = np.array([r[hits] for r in kept], dtype=float)
        n = np.array([r[total] for r in kept])
        decision[target] = homogeneity(k, n) | {"prompts": len(kept)}
        print(f"{target}: {json.dumps(decision[target])}", flush=True)
    decision["predict"] = [t for t in TARGETS if decision[t]["varies"]]
    print("varies across prompts:", decision["predict"] or "neither, so there is no propensity to predict", flush=True)
    path = OUT / "decision.json"
    path.write_text(json.dumps(decision, indent=1))
    push(path)


if __name__ == "__main__":
    main()
