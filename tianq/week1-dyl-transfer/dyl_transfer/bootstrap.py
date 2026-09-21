"""Where teammates' code lives, and the config this folder runs with.

This is the only module that knows the repo layout. If Ravi restructures
`dyl_vs_oa`, this file changes and nothing else does. Everything else in
`dyl_transfer` imports through here.

Gigi's `src/common.py` is deliberately NOT imported. The only thing we would
have wanted from it is his A/B answer parser, but we never parse an A/B answer:
`lied`, `admitted` and the Part-1 letter are all recorded verbatim in his
exported `transcripts.jsonl`. Importing him would pull in `python-dotenv` and
run `DATA.mkdir()` inside his folder for no gain.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve()
FOLDER = _HERE.parents[1]
REPO = _HERE.parents[3]

RAVI = REPO / "ravi" / "week1-dyl-vs-oa"
RAVI_SRC = RAVI / "src"
GIGI = REPO / "gigi" / "ascii-sandbagging"

TRANSCRIPTS = GIGI / "results" / "transcripts.jsonl"
PAPER_TARGETS = RAVI / "results" / "paper_targets.json"

# Everything this pipeline produces lives on HuggingFace. This is a local
# staging area for uploads and a cache for re-reads - transient, gitignored,
# and safe to delete. Nothing here is a source of truth.
SCRATCH = FOLDER / "scratch"

if str(RAVI_SRC) not in sys.path:
    sys.path.insert(0, str(RAVI_SRC))


def experiment_config():
    """Ravi's config, with results redirected into this folder.

    `load_config` reads YAML straight into the dataclass without coercing
    strings to `Path`, so `results_dir` cannot be redirected through a YAML
    file without breaking `cfg.results_dir / "..."`. Overriding the dataclass
    afterwards is the safe way to do it.
    """
    from dyl_vs_oa.config import load_config

    cfg = load_config()
    return replace(cfg, results_dir=SCRATCH)


def hf_repo() -> str:
    repo = os.environ.get("HF_DATASET_REPO", "").strip()
    if not repo:
        raise RuntimeError(
            "HF_DATASET_REPO is unset. Copy .env.example to .env and fill it in, "
            "then `set -a; source .env; set +a` before running."
        )
    return repo


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token.startswith("hf_xxx"):
        raise RuntimeError(
            "HF_TOKEN is unset or still the .env.example placeholder. A write-scoped "
            "token is required: https://huggingface.co/settings/tokens"
        )
    return token
