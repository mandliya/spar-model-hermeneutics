"""How far training moved the writer's weights, and whether the released writer was trained in 16 or 32 bits.

    python weight_change.py --released <released writer folder> --save <run>/actor/iter_0000010

Why: the library's Miles patch loads the models in 16-bit floats (bfloat16) for training. A bfloat16 number has 8
significant bits, so a weight of size w can only change in steps of about w/128. One AdamW update is at most the
learning rate (1.41e-5), so it registers only on weights smaller than about 0.007; on a larger weight it rounds away.
This script measures that on a few of the writer's weight matrices:
  released in 16 bits?   the share of the released writer's values that a bfloat16 can hold exactly. Near 100% means
                         the released writer was itself trained in 16 bits; near 0% means it was trained in 32 bits.
  changed, by size       of the weights in each size band, the share whose value in the save differs from the
                         released value rounded to 16 bits (what training started from).
Reads the save with the library's own loader (tools/convert_fsdp_to_hf.py). CPU only; needs about 60 GB of memory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import torch
from safetensors import safe_open

NLA = pathlib.Path(__file__).resolve().parents[2] / "vendor" / "natural_language_autoencoders"
MATRICES = ["model.layers.0.self_attn.q_proj.weight", "model.layers.20.mlp.down_proj.weight",
            "model.layers.41.self_attn.o_proj.weight", "model.layers.61.mlp.up_proj.weight", "model.embed_tokens.weight"]
BANDS = [(0.0, 1e-3), (1e-3, 7e-3), (7e-3, 3e-2), (3e-2, float("inf"))]


def library_loader():
    spec = importlib.util.spec_from_file_location("convert_fsdp_to_hf", NLA / "tools" / "convert_fsdp_to_hf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--save", required=True)
    args = ap.parse_args()
    convert = library_loader()
    saved = convert._load_fsdp_state_dict(convert._detect_model_dir(args.save))
    saved = {k.removeprefix("model_state.model."): v for k, v in saved.items() if isinstance(v, torch.Tensor)}
    shard_of = json.loads((pathlib.Path(args.released) / "model.safetensors.index.json").read_text())["weight_map"]
    for name in MATRICES:
        with safe_open(str(pathlib.Path(args.released) / shard_of[name]), framework="pt") as f:
            released = f.get_tensor(name)
        start = released.to(torch.bfloat16)
        now = saved[name]
        exact = (start.to(released.dtype) == released).float().mean().item()
        size = start.float().abs()
        changed = start != now
        bands = "; ".join(f"|w| {lo:g} to {hi:g}: {(size >= lo).logical_and(size < hi).float().mean().item():.1%} of weights, "
                          f"{changed[(size >= lo).logical_and(size < hi)].float().mean().item():.2%} changed" for lo, hi in BANDS)
        print(f"{name} ({released.dtype}, saved {now.dtype}): released values exact in 16 bits {exact:.1%}; "
              f"changed overall {changed.float().mean().item():.2%}; {bands}", flush=True)


if __name__ == "__main__":
    main()
