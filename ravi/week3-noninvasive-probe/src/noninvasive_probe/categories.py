"""Gigi's 11 deception flavors. Rows end on the reply; no follow-up is appended.

Shared by the generic-probe comparison and the per-flavor matrix. Training labels
for the matrix are these lie labels. Instructed pairs are not loaded here.
"""

from __future__ import annotations

import sys
import types

from noninvasive_probe.paths import WEEK3, bootstrap

bootstrap()

REPO = WEEK3.parents[1]
TIANQ_REPRO = REPO / "tianq" / "dyl-probe-repro"

# Display names match tianq/.../datasets/gigi/analyses/generalization.py CATEGORIES.
CATEGORIES = {
    "ascii_sandbagging": "ASCII sandbagging",
    "sycophancy_answer": "Sycophancy A",
    "sycophancy_feedback": "Sycophancy B",
    "mask_factual": "MASK provided facts",
    "mask_known_facts": "MASK known facts",
    "mask_statistics": "MASK statistics",
    "mask_continuations": "MASK continuations",
    "mask_disinformation": "MASK disinformation",
    "unanswerable": "Unanswerable",
    "capture_the_secret": "Capture the secret",
    "alibi": "Alibi",
}

# Column order for the train × test matrix. Unanswerable is last and is never trained:
# 7 truthful replies, under the 10-row minimum.
FLAVOR_ORDER = (
    "ascii_sandbagging",
    "sycophancy_answer",
    "sycophancy_feedback",
    "mask_factual",
    "mask_known_facts",
    "mask_statistics",
    "mask_continuations",
    "mask_disinformation",
    "capture_the_secret",
    "alibi",
    "unanswerable",
)
TRAIN_FLAVORS = tuple(name for name in FLAVOR_ORDER if name != "unanswerable")
TEST_ONLY = "unanswerable"

# Tianq training.MIN_ROWS_PER_CLASS. A cell under this in either class is not averaged.
FEW = 10

# Noddybear/lies was rebuilt on 2026-10-05 ("Fix did_lie labels and rebuild configs").
# That revision replaced sample ids with train_0, so Tianq's joins no longer match
# Gigi's uids. This is the parent revision, the one those joins were written against.
LIES_REVISION = "e8fc10f495aabc2d2d95f1e947b10bbcced444d5"


def pin_lies_revision() -> None:
    """Point Hugging Face filesystem reads of Noddybear/lies at LIES_REVISION."""
    import fsspec

    if getattr(fsspec.filesystem, "_lies_revision_pinned", False):
        return
    original = fsspec.filesystem
    revision = LIES_REVISION

    def filesystem(protocol, *args, **kwargs):
        fs = original(protocol, *args, **kwargs)
        if protocol != "hf" or getattr(fs, "_lies_glob_patched", False):
            return fs
        real_glob = fs.glob

        def glob(pattern, *glob_args, **glob_kwargs):
            if isinstance(pattern, str):
                pattern = pattern.replace(
                    "datasets/Noddybear/lies/",
                    f"datasets/Noddybear/lies@{revision}/",
                )
            return real_glob(pattern, *glob_args, **glob_kwargs)

        fs.glob = glob
        fs._lies_glob_patched = True
        return fs

    filesystem._lies_revision_pinned = True
    fsspec.filesystem = filesystem


def resolve_flavors(raw: str | None) -> list[str]:
    """CLI flavor list, always in FLAVOR_ORDER. `all` is the 11 flavors."""
    if raw is None or raw.strip() in {"", "all"}:
        return list(FLAVOR_ORDER)
    asked = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = sorted(asked - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"unknown flavors {unknown}; known: {', '.join(FLAVOR_ORDER)}")
    return [name for name in FLAVOR_ORDER if name in asked]


def allow_dataset_import() -> None:
    """Import Tianq's dataset modules, and pin Noddybear/lies to the revision they were written against."""
    tianq = str(TIANQ_REPRO)
    if tianq not in sys.path:
        sys.path.insert(0, tianq)
    pin_lies_revision()
    import transformers

    if not hasattr(transformers, "Gemma3ForConditionalGeneration"):
        transformers.Gemma3ForConditionalGeneration = type("Gemma3ForConditionalGeneration", (), {})
    try:
        import lie_detectors  # noqa: F401
    except ModuleNotFoundError:
        class _Stub(types.ModuleType):
            def __getattr__(self, name: str):
                if name.startswith("__"):
                    raise AttributeError(name)
                value = type(name, (), {})
                setattr(self, name, value)
                return value

        sys.modules["lie_detectors"] = _Stub("lie_detectors")


def load_category(name: str, max_n: int | None) -> list[tuple[list[dict[str, str]], int, str]]:
    """(messages ending on the reply, lie label, uid). No follow-up is appended."""
    if name not in CATEGORIES:
        raise KeyError(f"unknown flavor {name!r}")
    allow_dataset_import()
    if name == "ascii_sandbagging":
        from dyl_repro.datasets.gigi.ascii_sandbagging import files

        rows = []
        for uid in sorted(files.ascii_results()):
            label = files.lied(uid)
            if label is None:
                continue
            rows.append((files.conversation_before_followup(uid), int(bool(label)), uid))
    else:
        from dyl_repro.datasets import get

        table = get(name).rows()
        rows = [
            (table.prefixes[uid], int(bool(table.turn_lie[uid])), uid)
            for uid in sorted(table.prefixes)
        ]
    if max_n is not None:
        rows = rows[:max_n]
    return rows
