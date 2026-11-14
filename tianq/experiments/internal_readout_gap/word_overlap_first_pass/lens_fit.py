# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# To fit a J-lens in future, call the vendored script behind the published lenses (../vendor/neuronpedia-jlens/fit_lens.py) with a published config.yaml's arguments (../PLAN.md). This file fitted 200 prompts with no convergence stop.
"""Fit a J-lens for a model that has no published one (Qwen3-Next-80B), with the authors' own jlens.fit.

    python lens_fit.py --model qwen3-next-80b --prompts 2 --check     # fit check: does jlens run on it, how fast
    python lens_fit.py --model qwen3-next-80b --prompts 300           # the fit, resumable from its checkpoint

Recipe as in the paper and Neuronpedia's lenses: WikiText-103 prompts of 128 tokens (jlens.examples), bf16, every
layer below the last. Each prompt costs ceil(d_model / dim_batch) backward passes, so the prompt count is a time
budget; the jlens README says quality saturates quickly and ~100 prompts is usable. The count used is logged.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import jlens
from jlens.examples import load_wikitext_prompts

import models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--prompts", type=int, required=True)
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--check", action="store_true", help="fit on --prompts prompts, report time per prompt, save nothing")
    args = ap.parse_args()
    jlens.configure_logging()
    out = pathlib.Path(models.OUT) / "lenses"
    out.mkdir(parents=True, exist_ok=True)
    lm, _ = models.load(args.model)
    print(f"loaded {args.model}: {lm}", flush=True)
    prompts = load_wikitext_prompts(args.prompts)
    start = time.time()
    lens = jlens.fit(lm, prompts, dim_batch=args.dim_batch, max_seq_len=128,
                     checkpoint_path=None if args.check else str(out / f"{args.model}_fit_checkpoint.pt"),
                     checkpoint_every=10)
    per = (time.time() - start) / len(prompts)
    print(f"fit {args.model}: {len(prompts)} prompts, dim_batch {args.dim_batch}, {per:.1f} s per prompt; {lens}", flush=True)
    if not args.check:
        path = out / f"{args.model}_jacobian_lens_n{len(prompts)}.pt"
        lens.save(str(path))
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
