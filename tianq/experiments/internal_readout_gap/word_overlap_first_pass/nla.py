# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# For new work use ../shared/ (NLA sentences, batched).
"""NLA explanations of a model's activation at each spot, with the authors' round-trip score.

    /workspace/venvs/nla/bin/python nla.py --model gemma-3-27b-it --limit 1   # test on one spot
    /workspace/venvs/nla/bin/python nla.py --model gemma-3-27b-it             # every spot read by readout.py
Runs in the NLA venv (transformers 4.57.1): the vendored client expects transformers 4.x.

Uses the vendored NLA client (../vendor/natural_language_autoencoders/nla_inference.py, see ../vendor/VENDORED.md) for everything
that defines an NLA: the sidecar config and its tokenizer checks (load_nla_config), the injection scale
(normalize_activation), the neighbour-checked injection (inject_at_marked_positions) and the AR score (NLACritic:
direction MSE and cosine). Only the engine differs: the authors serve the AV with SGLang `input_embeds`; their docs say
any engine that accepts input embeddings works, so this runs transformers `generate(inputs_embeds=...)` with their
sampling (temperature 1.0, 200 new tokens). The AV checkpoint is a plain Gemma3ForCausalLM, so its own embedding
layer applies gemma's sqrt(d) scale.

The activation is the output of block K, K = the sidecar's extraction_layer_index (their datagen hooks
model.layers[K]; nla/models.py), i.e. readout.py's acts[K]. Saved per spot: OUT/<model>/nla/<spot>.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import models

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "vendor" / "natural_language_autoencoders"))
import nla_inference as N  # noqa: E402

NLAS = {"gemma-3-27b-it": ("kitft/nla-gemma3-27b-L41-av", "kitft/nla-gemma3-27b-L41-ar")}
SAMPLES = 5
EXPLANATION = re.compile(r"<explanation>\s*(.*?)\s*(?:</explanation>|$)", re.DOTALL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(NLAS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    av_dir, ar_dir = (pathlib.Path(snapshot_download(r)) for r in NLAS[args.model])
    layer = int(yaml.safe_load((av_dir / "nla_meta.yaml").read_text())["extraction_layer_index"])
    tok = AutoTokenizer.from_pretrained(str(av_dir))
    cfg = N.load_nla_config(av_dir, tok)
    av = AutoModelForCausalLM.from_pretrained(str(av_dir), dtype=torch.bfloat16, device_map="cuda:0").eval()
    critic = N.NLACritic(ar_dir, device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    prompt = tok.apply_chat_template([{"role": "user", "content": cfg.actor_prompt_template.format(
        injection_char=cfg.injection_char)}], tokenize=True, add_generation_prompt=True)
    prompt = list(prompt["input_ids"] if hasattr(prompt, "keys") else prompt)
    ids = torch.tensor([prompt], device=av.device)
    base = pathlib.Path(models.OUT) / args.model
    (base / "nla").mkdir(parents=True, exist_ok=True)
    acts = sorted((base / "acts").glob("*.pt"))[: args.limit]
    print(f"NLA {args.model}: layer {layer}, injection scale {cfg.injection_scale}, {len(acts)} spots", flush=True)
    for n, path in enumerate(acts, 1):
        out = base / "nla" / f"{path.stem}.json"
        if out.exists():
            continue
        h = torch.load(path)[layer].float()
        with torch.no_grad():
            embeds = av.get_input_embeddings()(ids)
            vec = N.normalize_activation(h[None].to(av.device, embeds.dtype), cfg.injection_scale)
            embeds = N.inject_at_marked_positions(ids, embeds, vec, cfg.injection_token_id,
                                                  cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
            gen = av.generate(inputs_embeds=embeds.expand(SAMPLES, -1, -1), attention_mask=torch.ones(
                SAMPLES, ids.shape[1], device=av.device, dtype=torch.long), do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, max_new_tokens=200)
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen]
        samples = []
        for text in texts:
            m = EXPLANATION.search(text)
            explanation = (m.group(1) if m else text).strip()
            mse, cos = critic.score(explanation, h.numpy())
            samples.append({"explanation": explanation, "raw": text, "closed": bool(m and "</explanation>" in text),
                            "mse": mse, "cos": cos})
        out.write_text(json.dumps({"spot": path.stem, "layer": layer, "samples": samples}))
        print(f"[{n}/{len(acts)}] {path.stem}: cos {[round(s['cos'], 2) for s in samples]}; "
              f"{samples[0]['explanation'][:160]!r}", flush=True)


if __name__ == "__main__":
    main()
