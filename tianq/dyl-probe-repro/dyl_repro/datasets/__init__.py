"""Dataset-specific code, one sub-package per data source. Everything outside
this folder (core/, cli/) works on any dataset.

    aisi/               AISI's released transcripts: training fingerprint, validation, calibration
    ascii_sandbagging/  Gigi's ASCII rows with swappable follow-ups: does the probe transfer?

Each dataset module exposes:

    conversations() -> list[Conversation]
    evaluate(cache: ActivationCache, probe: DylProbe) -> dict
        That dataset's own check, printed and returned. Datasets differ in what
        they are for (fingerprint, benchmark, calibration, transfer), so the
        check lives with the dataset rather than in one generic scorer.

A dataset whose follow-up can be swapped also sets FOLLOWUPS_AVAILABLE; both
functions then take the follow-up name as a last argument, and each follow-up
is stored and scored separately under "<dataset>/<follow-up>".

Every dataset is stored at both positions in core.prefill.POSITIONS, and every
evaluate reports its full results at both, side by side.

Adding a dataset means adding a module and its entry in MODULES. The names are
also the storage prefixes on the Hub, so they do not change.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

MODULES = {
    "aisi_city_train": "dyl_repro.datasets.aisi.city_train",
    "aisi_varied_deception": "dyl_repro.datasets.aisi.varied_deception",
    "aisi_alpaca": "dyl_repro.datasets.aisi.alpaca",
    "ascii_sandbagging": "dyl_repro.datasets.ascii_sandbagging",
}
NAMES = tuple(MODULES)


def get(name: str) -> ModuleType:
    if name not in MODULES:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(NAMES)}")
    return import_module(MODULES[name])


def runs(name: str, followups: list[str] | None) -> list[tuple[str, str | None]]:
    """(storage key, follow-up) for each run a CLI invocation asks for."""
    available = getattr(get(name), "FOLLOWUPS_AVAILABLE", None)
    if available is None:
        if followups:
            raise ValueError(f"{name} has its follow-up baked in; drop --followup")
        return [(name, None)]
    if not followups:
        raise ValueError(f"{name} needs --followup: 'all' or any of {', '.join(available)}")
    chosen = list(available) if followups == ["all"] else followups
    unknown = [f for f in chosen if f not in available]
    if unknown:
        raise ValueError(f"unknown follow-ups for {name}: {unknown}; known: {', '.join(available)}")
    return [(f"{name}/{f}", f) for f in chosen]


def conversations(name: str, followup: str | None):
    module = get(name)
    return module.conversations() if followup is None else module.conversations(followup)


def evaluate(name: str, cache, probe, followup: str | None) -> dict:
    module = get(name)
    return module.evaluate(cache, probe) if followup is None else module.evaluate(cache, probe, followup)
