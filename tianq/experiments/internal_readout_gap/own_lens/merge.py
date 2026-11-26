"""Save a distilled gemma (base weights with one of Gigi's adapters merged in) as a model folder the lens fitter can load.

    python merge.py --model gemma-classic-step92      # writes /workspace/models/gemma-classic-step92

The merge is shared.models.load_hf, the same one every earlier read of this model used (bf16, merged in memory), so
the fitted lens belongs to exactly the model the restart points were read from.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import models  # noqa: E402

MERGED = pathlib.Path("/workspace/models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*models.DISTILLED, *models.CLASSIC])
    args = ap.parse_args()
    out = MERGED / args.model
    if (out / "config.json").exists():
        print(f"nothing to do: {out} exists", flush=True)
        return
    hf, tok = models.load_hf(args.model)
    tmp = out.with_name(out.name + ".partial")
    hf.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    tmp.rename(out)
    print(f"saved {args.model} ({hf.dtype}) to {out}", flush=True)


if __name__ == "__main__":
    main()
