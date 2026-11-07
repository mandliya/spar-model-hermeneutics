"""Pull a target's stored rows from the Hub into scratch, both read positions, every kind of lie.

    DYL_MODEL=qwen3-235b ./run.sh yuqi shared.pull        # from tianq/dyl-probe-repro

Uses compare_diversity's loader, which pulls a run, rebuilds its conversations and stops if any row is
missing. Resumable: rows already in scratch are skipped.
"""

from __future__ import annotations

from dyl_repro.core.model import CURRENT
from experiments.cross_category_transfer.compare_diversity import load
from experiments.cross_category_transfer.diversity import VARIANTS
from experiments.cross_category_transfer.generalization import CATEGORIES


def main() -> None:
    for variant in VARIANTS:
        for name in CATEGORIES:
            category = load(variant, name)
            print(f"{CURRENT.tag} {variant} {name}: {len(category.uids)} rows, complete", flush=True)


if __name__ == "__main__":
    main()
