"""GPU: what each model holds at the end of each prompt, before it writes anything.

    ./run.sh propensity extract        # from af-probe-repro

For every model in common.MODELS (the organism, its plain parent, the 8B reader), every prompt in the model's
own chat template with the assistant turn opened, read at the last token, at the model's layer (bf16 weights,
kept as float32). Saved as scratch/af-propensity/activations/<model>.pt and pushed to the Hub. Resumable: a
model already saved is skipped.
"""

from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.alignment_faking_propensity.common import MODELS, OUT, model_path, prompts, push, question
from src.extract_act import check_p2p_integrity


@torch.no_grad()
def main() -> None:
    out = OUT / "activations"
    out.mkdir(parents=True, exist_ok=True)
    samples = prompts()
    check_p2p_integrity()
    for name, (_, _, layer) in MODELS.items():
        path = out / f"{name}.pt"
        if path.exists():
            print(f"{name}: already saved", flush=True)
            continue
        tokenizer = AutoTokenizer.from_pretrained(model_path(name))
        model = AutoModelForCausalLM.from_pretrained(model_path(name), dtype=torch.bfloat16, device_map="auto").eval()
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise SystemExit(f"{name}: some weights were offloaded off the GPUs")
        rows = []
        for s in samples:
            ids = tokenizer.apply_chat_template(question(s), add_generation_prompt=True, return_dict=True,
                                                return_tensors="pt")["input_ids"].to(model.device)
            rows.append(model(ids, output_hidden_states=True).hidden_states[layer][0, -1].float().cpu())
        acts = torch.stack(rows)
        torch.save({"model": name, "layer": layer, "ids": [s["id"] for s in samples], "acts": acts}, path)
        push(path)
        print(f"{name}: layer {layer}, {tuple(acts.shape)}, norm mean {acts.norm(dim=1).mean():.1f},"
              f" finite {bool(acts.isfinite().all())}", flush=True)
        del model  # the loader's hooks keep it alive until collected: free it before the next model loads
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
