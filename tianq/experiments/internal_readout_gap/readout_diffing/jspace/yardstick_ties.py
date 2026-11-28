"""Does the yardstick's ranking of the similarity methods depend on how ties are counted? (18 Nov, review check)

    READOUT_DISTRIBUTIONS=<folder of write_jlens_lists.py's files> python yardstick_ties.py

similarity_methods.py's yardstick counts "same restart point scores above a different one" with a strict >, so a tie
counts as a loss. Short lists tie often (two top-10 lists usually share nothing with either), which understates them.
For base gemma against Qwen3.6 through own lenses, same task only, at two-thirds and 0.95 depth, this prints for
every method the strict rate, the share of ties, and the rate with a tie counted as half.
Found: the top of the ranking does not change (rarity-weighted over 5,000 words stays best); "top 10 words" rises
from 0.32 to 0.60 at two-thirds and from 0.66 to 0.79 at 0.95, above the probability-weighted methods there.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # readout_diffing/: method_menu, restart_points
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # internal_readout_gap/: readout_methods, shared
import method_menu  # noqa: E402
from readout_methods.likeness import evaluation  # noqa: E402
from readout_methods.likeness.pair import Pair  # noqa: E402
from readouts import BASE, FINAL, FOLDER, LENSES, QWEN, load_shared  # noqa: E402
from restart_points import tasks  # noqa: E402

DEPTHS = (2 / 3, 0.95)


def main() -> None:
    base, final = ((v, LENSES["own lens"][v]) for v in (BASE, FINAL))
    reads, names = load_shared([(QWEN, "published"), base, final])
    mask = evaluation.others(len(names)) & evaluation.same_task(tasks(names))
    result: dict = {}
    for depth in DEPTHS:
        q, b, f = (reads[key].lists(depth, False) for key in ((QWEN, "published"), base, final))
        sims = method_menu.tables(Pair(q, b, pool=[q, b, f]))   # the final checkpoint only adds to the rare-word weights
        for method, table in sims.items():
            strict, ties = evaluation.same_moment_rates(table, mask)
            print(f"depth {depth:.3f} | {method}: same beats other of the same task {strict:.3f}; ties {ties:.3f}; "
                  f"with a tie as half {strict + ties / 2:.3f}", flush=True)
            result[f"depth {depth:.3f} | {method}"] = {"strict": strict, "ties": ties, "tie_as_half": strict + ties / 2}
    path = FOLDER / "yardstick_ties.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
