"""GPU: one model's activations for the gate, with the paper's own tokenizer masks and extractor.

    python -m experiments.alignment_faking_hidden_reasoning.extract <model>     # a key of vendor src/models.py, or llama-8b-3.1

For every cell (views.CELLS, both variants) and view, the model is prefilled with the organism's rollouts:
read in the organism itself these are its own activations, read in any other model it is a reader. Plus
the paper's probe training set (synthetic_af__all), on which every model trains its own probe. Kept per
rollout: the last read token and the mean over the read tokens (the paper's two features), at the model's
layers, saved as scratch/af-gate/activations/<model>/<dataset>__<view>.pt and pushed to the Hub under
af-gate/activations/. Resumable: a file already saved is skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

from experiments.alignment_faking_hidden_reasoning import views
from src import models
from src.extract_act import extract_activation_nnsight
from src.tokenized_data import TokenizedDataset

# The 8B reader is not in the paper's registry: added here, as the paper adds its models.
models.ALL_MODEL_PATHS.setdefault("llama-8b-3.1", "meta-llama/Llama-3.1-8B-Instruct")
# The paper's layers (33, 37, 40 of 80); the 8B reads the same relative depths of its 32.
LAYERS = {"llama-8b-3.1": (13, 15, 16)}
PAPER_LAYERS = (33, 37, 40)
OUT = Path(__file__).resolve().parents[3] / "tianq" / "af-probe-repro" / "scratch" / "af-gate" / "activations"
TRAINING = "synthetic_af__all"


def per_rollout(feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The extractor returns the read tokens of every rollout back to back: [tokens, layers, d]. Split by
    each rollout's count into its last token and its mean."""
    counts = mask.sum(1).tolist()
    if sum(counts) != feats.shape[0] or min(counts) == 0:
        raise SystemExit(f"read tokens {feats.shape[0]} vs mask {sum(counts)}, empty rollouts: {counts.count(0)}")
    chunks = torch.split(feats, counts)
    return torch.stack([c[-1] for c in chunks]), torch.stack([c.float().mean(0).to(feats.dtype) for c in chunks])


def main() -> None:
    model_name = sys.argv[1]
    layers = list(LAYERS.get(model_name, PAPER_LAYERS))
    out = OUT / model_name
    out.mkdir(parents=True, exist_ok=True)
    # The paper read of a cell is only needed in its own organism (score.py's tie-in to the paper).
    jobs = [(TRAINING, model_name, "paper")] + [(task, organism, kind) for organism, task in views.cell_tasks()
                                                 for kind in views.VIEWS if kind != "paper" or organism == model_name]
    todo = [j for j in jobs if not (out / f"{j[1]}__{j[0]}__{j[2]}.pt").exists()]
    print(f"{model_name}: layers {layers}; {len(jobs) - len(todo)} of {len(jobs)} datasets already saved", flush=True)
    if not todo:
        return
    gpus = list(range(torch.cuda.device_count()))
    model, tokenizer = models.get_model_and_tokenizer(model_name, os.path.join(os.environ["HF_HOME"], "hub"),
                                                      cut_at_layer=max(layers) + 2,
                                                      gpu_ids=gpus if "8b" not in model_name else [0])
    api = HfApi(token=os.environ["HF_TOKEN"])
    for task, organism, kind in todo:
        dataset, kept = views.view(task, organism, kind)
        tokenized = TokenizedDataset.from_dataset(dataset, tokenizer)
        tokenized.verify_detection_mask()
        feats, _ = extract_activation_nnsight(model, tokenized, batch_size=1 if "bench_af" in task else 8,
                                              layers=layers, verbose=True, detection_mask=tokenized.detection_mask)
        last, mean = per_rollout(feats.cpu(), tokenized.detection_mask.cpu())
        path = out / f"{organism}__{task}__{kind}.pt"
        torch.save({"model": model_name, "organism": organism, "task": task, "view": kind, "layers": layers,
                    "last": last.half(), "mean": mean.half(), "labels": dataset.labels, "kept": kept,
                    "texts": ["\n".join(m.content for m in d) for d in dataset.dialogues]}, path)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"af-gate/activations/{model_name}/{path.name}",
                        repo_id=os.environ["HF_DATASET_REPO"], repo_type="dataset")
        print(f"saved {path.name}: {len(dataset)} rollouts, {sum(dataset.labels)} deceptive", flush=True)


if __name__ == "__main__":
    main()
