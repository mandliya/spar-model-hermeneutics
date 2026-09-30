from __future__ import annotations

from noninvasive_probe.paths import WEEK1, WEEK3, bootstrap

bootstrap()

from dyl_vs_oa.config import load_config as _week1_config


def load_config(path: str | None = None):
    cfg = _week1_config(path or (WEEK1 / "configs" / "default.yaml"))
    cfg.data_dir = WEEK1 / "data"
    cfg.results_dir = WEEK3 / "results"
    return cfg
