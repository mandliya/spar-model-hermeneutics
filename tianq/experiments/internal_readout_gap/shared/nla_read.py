"""NLA sentences for stored activations: each model's activation, read through the NLA that models.py names for it.

    <venv>/bin/python nla_read.py --experiment distillation_similarity --model gemma-3-27b-it --limit 2   # test
    <venv>/bin/python nla_read.py --experiment distillation_similarity --model gemma-3-27b-it             # all (resumable)

Reads OUT/<experiment>/<read>/acts/<name>.pt ([n_positions, n_layers, d_model], written by shared.readout; <read> is
the model's tag unless --read names another folder of that model's activations) and writes
OUT/<experiment>/<read>/nla/<name>.json: for each read position, SAMPLES explanations of the activation at the NLA's
own layer (the output of block K, K = the sidecar's extraction layer). Activations are batched: BATCH of them, each
sampled SAMPLES times, in one generate call. A second process with --reverse can share a long read from another GPU. Sampling is the official client's: temperature 1.0, 200 new tokens.

Two kinds of NLA, each run with its authors' code (vendor/, never edited):
  official   kitft/nla-* (gemma). vendor/natural_language_autoencoders/nla_inference.py supplies the sidecar config
             and its checks, the injection scale, the neighbour-checked injection into the input embeddings, and the
             reconstruction score (cos between the activation and what the AR rebuilds from the text). Their docs
             serve the AV with SGLang `input_embeds`; any engine that takes input embeddings works, so this runs
             transformers generate(inputs_embeds=...). Venv: /workspace/venvs/nla (the client expects transformers 4.x).
  easynla    ceselder/qwen3.6-27b-nla-av (unofficial). vendor/EasyNLA supplies the sidecar config and the hook that
             writes the activation into the residual stream at block 1; the prompt is the one shipped with the
             checkpoint (example_activations.parquet). No reconstruction score: its reconstructor is not in a format
             either vendored client loads. Venv: /workspace/venvs/readout (Qwen3.6 needs transformers 5.x).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import models  # noqa: E402

SAMPLES = 5
BATCH = 12
NEW_TOKENS = 200
# the opening "<" is optional: Qwen3.6's NLA sometimes starts its reply at "explanation>"
EXPLANATION = re.compile(r"<?explanation>\s*(.*?)\s*(?:</explanation>|$)", re.DOTALL)


class Official:
    def __init__(self, av_repo: str, ar_repo: str):
        import yaml
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, str(HERE.parent / "vendor" / "natural_language_autoencoders"))
        import nla_inference as N
        self.N = N
        av_dir, ar_dir = (pathlib.Path(snapshot_download(r)) for r in (av_repo, ar_repo))
        self.layer = int(yaml.safe_load((av_dir / "nla_meta.yaml").read_text())["extraction_layer_index"])
        self.tok = AutoTokenizer.from_pretrained(str(av_dir))
        self.cfg = N.load_nla_config(av_dir, self.tok)
        self.av = AutoModelForCausalLM.from_pretrained(str(av_dir), dtype=torch.bfloat16, device_map="cuda:0").eval()
        self.critic = N.NLACritic(ar_dir, device="cuda:0")
        prompt = self.tok.apply_chat_template([{"role": "user", "content": self.cfg.actor_prompt_template.format(
            injection_char=self.cfg.injection_char)}], tokenize=True, add_generation_prompt=True)
        self.ids = torch.tensor([list(prompt["input_ids"] if hasattr(prompt, "keys") else prompt)], device=self.av.device)

    @torch.no_grad()
    def explain(self, acts: torch.Tensor) -> list[list[str]]:
        """acts [B, d_model] -> SAMPLES raw generations for each."""
        n, c = len(acts), self.cfg
        ids = self.ids.expand(n, -1)
        embeds = self.av.get_input_embeddings()(ids)
        vec = self.N.normalize_activation(acts.to(self.av.device, embeds.dtype), c.injection_scale)
        embeds = self.N.inject_at_marked_positions(ids, embeds, vec, c.injection_token_id,
                                                   c.injection_left_neighbor_id, c.injection_right_neighbor_id)
        gen = self.av.generate(inputs_embeds=embeds.repeat_interleave(SAMPLES, 0), attention_mask=torch.ones(
            n * SAMPLES, ids.shape[1], device=self.av.device, dtype=torch.long), do_sample=True, temperature=1.0,
            top_p=1.0, top_k=0, max_new_tokens=NEW_TOKENS)
        texts = [self.tok.decode(g, skip_special_tokens=True) for g in gen]
        return [texts[i * SAMPLES:(i + 1) * SAMPLES] for i in range(n)]

    def cos(self, explanation: str, act: torch.Tensor) -> float | None:
        return self.critic.score(explanation, act.numpy())[1]


class EasyNLA:
    def __init__(self, av_repo: str):
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, str(HERE.parent / "vendor" / "EasyNLA"))
        from nla.config import load_nla_config
        from nla.utils import build_prompt_text, register_karvonen_hook
        av_dir = pathlib.Path(snapshot_download(av_repo))
        self.tok = AutoTokenizer.from_pretrained(str(av_dir))
        self.cfg = load_nla_config(str(av_dir), self.tok)
        self.layer = int(self.cfg.extraction_layer_index)
        self.examples = pq.read_table(av_dir / "example_activations.parquet").to_pylist()
        prompts = [e["prompt"] for e in self.examples]
        assert all(p == prompts[0] for p in prompts), "the shipped examples do not share one prompt"
        text = build_prompt_text(prompts[0], self.cfg.injection_char, self.tok)
        self.av = AutoModelForCausalLM.from_pretrained(str(av_dir), dtype=torch.bfloat16, device_map="cuda:0").eval()
        self.ids = torch.tensor([self.tok.encode(text, add_special_tokens=False)], device=self.av.device)
        self.vref = [None]
        register_karvonen_hook(self.av, self.vref, self.cfg.injection_token_id, self.cfg.injection_left_neighbor_id,
                               self.cfg.injection_right_neighbor_id, layer_idx=1)

    @torch.no_grad()
    def explain(self, acts: torch.Tensor) -> list[list[str]]:
        n = len(acts)
        ids = self.ids.expand(n * SAMPLES, -1)
        self.vref[0] = acts.float().repeat_interleave(SAMPLES, 0).to(self.av.device)
        try:
            gen = self.av.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0,
                                   top_p=1.0, top_k=0, max_new_tokens=NEW_TOKENS, pad_token_id=self.tok.eos_token_id)
        finally:
            self.vref[0] = None
        texts = [self.tok.decode(g[ids.shape[1]:], skip_special_tokens=True) for g in gen]
        return [texts[i * SAMPLES:(i + 1) * SAMPLES] for i in range(n)]

    def cos(self, explanation: str, act: torch.Tensor) -> float | None:
        return None

    def shipped_examples(self, n: int) -> None:
        """The test before any run: explain the checkpoint's own example activations, SAMPLES times each, and print
        each beside its source text with how many samples returned an explanation."""
        rows = self.examples[:n]
        with_text = 0
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            acts = torch.tensor([r["activation_vector"] for r in chunk], dtype=torch.float32)
            for row, texts in zip(chunk, self.explain(acts)):
                found = [m.group(1).strip() for m in map(EXPLANATION.search, texts) if m and m.group(1).strip()]
                with_text += bool(found)
                print(f"shipped example | {len(found)} of {SAMPLES} samples returned an explanation | source text ends: "
                      f"{row['detokenized_text_truncated'][-200:]!r} | first explanation: {(found[0] if found else '')[:300]!r}",
                      flush=True)
        print(f"shipped examples: {with_text} of {len(rows)} got at least one explanation", flush=True)


def backend(model: str):
    kind, *repos = models.MODELS[model].nla
    return {"official": Official, "easynla": EasyNLA}[kind](*repos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--model", required=True, choices=[t for t, m in models.MODELS.items() if m.nla])
    ap.add_argument("--read", default=None, help="the folder of OUT/<experiment> holding the activations (default: the model's tag)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true",
                    help="work through the files from the end: a second process on another GPU meets the first in the middle")
    ap.add_argument("--shipped-examples", type=int, default=0,
                    help="easynla only: first explain this many of the checkpoint's own example activations (the one-row test)")
    args = ap.parse_args()
    base = models.OUT / args.experiment / (args.read or args.model)
    (base / "nla").mkdir(parents=True, exist_ok=True)
    nla = backend(args.model)
    if args.shipped_examples:
        nla.shipped_examples(args.shipped_examples)
    names = [p.stem for p in sorted((base / "acts").glob("*.pt"))][: args.limit]
    todo = [n for n in names if not (base / "nla" / f"{n}.json").exists()]
    if args.reverse:
        todo.reverse()
    print(f"NLA {args.model}: layer {nla.layer}, {len(names)} activations files, {len(todo)} to do, "
          f"{SAMPLES} samples each", flush=True)
    # one work item per (file, position); a file is written once all its positions are explained
    items = []
    for name in todo:
        acts = torch.load(base / "acts" / f"{name}.pt")[:, nla.layer].float()
        items += [(name, p, a) for p, a in enumerate(acts)]
    results: dict[str, dict[int, list[dict]]] = {}
    written = 0
    for start in range(0, len(items), BATCH):
        # files another process finished in the meantime (--reverse) are left alone
        chunk = [it for it in items[start:start + BATCH] if not (base / "nla" / f"{it[0]}.json").exists()]
        if not chunk:
            continue
        for (name, p, act), texts in zip(chunk, nla.explain(torch.stack([a for _, _, a in chunk]))):
            samples = []
            for text in texts:
                m = EXPLANATION.search(text)
                explanation = (m.group(1) if m else text).strip()
                samples.append({"explanation": explanation, "raw": text, "closed": bool(m and "</explanation>" in text),
                                "cos": nla.cos(explanation, act)})
            results.setdefault(name, {})[p] = samples
        for name in sorted({n for n, _, _ in chunk}):
            positions = json.loads((base / "lens" / f"{name}.json").read_text())["positions"]
            if len(results[name]) == len(positions):
                per = results.pop(name)
                (base / "nla" / f"{name}.json").write_text(json.dumps(
                    {"name": name, "layer": nla.layer, "positions": {positions[i]: per[i] for i in sorted(per)}}))
                written += 1
                first = per[0][0]
                print(f"[{written}/{len(todo)}] {name}: cos {[s['cos'] and round(s['cos'], 2) for s in per[0]]}; "
                      f"{first['explanation'][:160]!r}", flush=True)
    print(f"done: {len(list((base / 'nla').glob('*.json')))} files with NLA sentences", flush=True)


if __name__ == "__main__":
    main()
