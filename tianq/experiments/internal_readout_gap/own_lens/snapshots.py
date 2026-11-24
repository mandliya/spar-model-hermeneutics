"""Save a running J-lens fit as lens files at fixed prompt counts, while the fit goes on.

    python snapshots.py --fit_dir /workspace/out/internal-readout-gap/own_lens/gemma-3-27b-it

fit_lens.py rewrites <slug>_checkpoint.pt (the running sum of per-prompt Jacobians) after every prompt. Each time the
prompt count first reaches a milestone (20, 50, 100, then every 100), this writes the running mean as a lens file,
<fit_dir>/snapshots/lens_at_<n>.pt, in the layout JacobianLens.load reads. n is the checkpoint's own count, so a file
says exactly how many prompts it holds. Used for the early reproduction check and to compare two fits at an equal
prompt count. Stops once the final lens file exists.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import torch

MILESTONES = [20, 50, *range(100, 1001, 100)]


def due(n: int, have: list[int]) -> bool:
    """True when the fit has passed a milestone that no saved snapshot covers yet."""
    passed = [m for m in MILESTONES if m <= n]
    return bool(passed) and not any(h >= passed[-1] for h in have)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_dir", required=True)
    ap.add_argument("--find", type=int, default=None, help="print the saved snapshot with the fewest prompts >= this; exit 1 if none yet")
    args = ap.parse_args()
    fit_dir = pathlib.Path(args.fit_dir)
    out = fit_dir / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    if args.find is not None:
        counts = sorted(c for c in (int(p.stem.split("_")[-1]) for p in out.glob("lens_at_*.pt")) if c >= args.find)
        if not counts:
            raise SystemExit(1)
        print(out / f"lens_at_{counts[0]}.pt")
        return
    while True:
        finished = any(fit_dir.glob("*_jacobian_lens.pt"))
        csvs = list(fit_dir.glob("*_convergence.csv"))
        n = max(0, sum(1 for _ in csvs[0].open()) - 1) if csvs else 0
        have = sorted(int(p.stem.split("_")[-1]) for p in out.glob("lens_at_*.pt"))
        ckpts = list(fit_dir.glob("*_checkpoint.pt"))
        if due(n, have) and ckpts:
            state = torch.load(ckpts[0], map_location="cpu", weights_only=True)
            done = int(state["n_done"])
            layers = list(state["source_layers"])
            d_model = next(iter(state["jacobian_sum"].values())).shape[0]
            tmp = out / f"lens_at_{done}.partial"
            torch.save({"J": {l: (state["jacobian_sum"][l] / done).to(torch.float16) for l in layers}, "n_prompts": done,
                        "source_layers": layers, "d_model": d_model}, tmp)
            tmp.rename(out / f"lens_at_{done}.pt")
            print(f"{time.strftime('%FT%TZ', time.gmtime())} saved the running lens at {done} prompts", flush=True)
            del state
        if finished:
            print("the fit has written its final lens; stopping", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
