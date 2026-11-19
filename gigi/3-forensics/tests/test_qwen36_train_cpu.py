"""CPU smoke for Qwen3.6 as the distillation student (phase 2, 19 Nov 2026). No GPU, no real weights.

  python tests/test_qwen36_train_cpu.py --tokenizer <dir with Qwen3.6-27B's config.json + tokenizer files> \
      --examples <jsonl from tests/test_qwen36_render.py build> --work <scratch dir>
  (env: transformers 5.23.0, peft 0.21.3, trl 1.18.1, accelerate 1.16.0, datasets 5.2.0, torch CPU)

Builds a TINY randomly initialised Qwen3_5ForConditionalGeneration from Qwen3.6-27B's own config (same architecture
class and layer pattern: 3 Gated DeltaNet layers + 1 gated attention layer, a 1-block vision tower, the real
248,320-token vocabulary), saves it with the real tokenizer, then runs the real scripts on it:
  1. distill_train.py train --student qwen36 on 2 real rendered runs, 5 optimizer steps (lr 1e-2 so a tiny random model
     visibly learns): asserts LoRA sits on DeltaNet, attention and MLP modules of the language model only, the train
     loss goes down, the adapter and checkpoints are written;
  2. distill_train.py nll --student qwen36 with the checkpoint (adapter switching and the base pass);
  3. merge_lora.py (merge fallback) and a reload of the merged model, which must match the PEFT model's logits;
  4. qwen_lora_gate.py reference (the HF half of the adapter-is-live gate).
"""
import argparse, json, pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"


def tiny_model(tok_dir, out):
    import torch
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    cfg = json.loads((pathlib.Path(tok_dir) / "config.json").read_text())
    tc = cfg["text_config"]
    tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
              num_attention_heads=2, num_key_value_heads=1, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
              linear_key_head_dim=16, linear_value_head_dim=16, mtp_num_hidden_layers=0)
    tc["rope_parameters"] = dict(tc["rope_parameters"], mrope_section=[3, 3, 2])   # 64 x 0.25 = 16 rotary dims = 2 x 8
    cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    config = Qwen3_5Config.from_dict(cfg)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(tok_dir).save_pretrained(out)
    for f in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
        if (pathlib.Path(tok_dir) / f).exists():
            shutil.copy(pathlib.Path(tok_dir) / f, out / f)
    kinds = {}
    for n, m in model.named_modules():
        leaf = n.rsplit(".", 1)[-1]
        if leaf in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "q_proj", "o_proj", "gate_proj"):
            kinds.setdefault(leaf, n)
    print("tiny model:", sum(p.numel() for p in model.parameters()) / 1e6, "M params; module names:", json.dumps(kinds))
    return out


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
    print(tail)
    if r.returncode:
        sys.exit(f"FAILED ({r.returncode}): {cmd[1]}")
    return r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--examples", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = pathlib.Path(a.work); shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    model_dir = tiny_model(a.tokenizer, work / "tiny_qwen36")
    exs = [json.loads(l) for l in open(a.examples) if '"mode": "trajectory"' in l]
    exs = sorted(exs, key=lambda e: len(json.dumps(e["messages"])))
    think = [e for e in exs if any(m.get("reasoning_content") for m in e["messages"])]
    plain = [e for e in exs if e not in think]
    data = work / "data"; data.mkdir()
    with open(data / "train.jsonl", "w") as f:
        for e in [think[0], plain[0]]:
            f.write(json.dumps({k: v for k, v in e.items() if k != "mode"}) + "\n")
    with open(data / "val.jsonl", "w") as f:
        for e in [think[1], plain[1]]:
            f.write(json.dumps({k: v for k, v in e.items() if k != "mode"}) + "\n")
    py = sys.executable; out = work / "out"
    run([py, SCRIPTS / "distill_train.py", "train", "--student", "qwen36", "--model", model_dir, "--data-dir", data,
         "--out", out, "--max-len", 4096, "--max-steps", 5, "--grad-accum", 1, "--lr", 1e-2, "--warmup-ratio", 0,
         "--eval-every", 5, "--save-every", 5, "--keep-checkpoints", "--final-val-sample", 0, "--no-base-eval"])
    log = json.loads((out / "log.json").read_text())
    losses = [h["train_nll_per_target_token"] for h in log["history"] if "train_nll_per_target_token" in h]
    kinds = log.get("lora_modules", {})
    print("lora modules:", kinds, "| train NLL per step:", [round(x, 3) for x in losses], "| eval:",
          [round(h["eval_loss"], 3) for h in log["history"] if "eval_loss" in h])
    assert kinds.get("deltanet", 0) == 3 * 5 and kinds.get("attention", 0) == 4 and kinds.get("mlp", 0) == 4 * 3, kinds
    assert len(losses) == 5 and losses[-1] < losses[0], f"train loss did not go down: {losses}"
    ck = sorted(out.glob("checkpoint-*"))
    assert (out / "adapter" / "adapter_config.json").exists() and ck, "adapter/checkpoint missing"
    run([py, SCRIPTS / "distill_train.py", "nll", "--student", "qwen36", "--model", model_dir, "--data-dir", data,
         "--splits", "val", "--adapters", f"c5={ck[-1]}", "--out", work / "nll", "--max-len", 4096])
    summ = json.loads((work / "nll" / "nll_summary.json").read_text())["summary"]
    b, c = summ["val|base|all"]["nll_per_token"], summ["val|c5|all"]["nll_per_token"]
    print(f"nll on val: base {b:.4f}, checkpoint {c:.4f}")
    assert c < b, "the trained checkpoint should fit the val runs better than the random base"
    run([py, SCRIPTS / "merge_lora.py", "--student", "qwen36", "--model", model_dir, "--adapters", f"c5={ck[-1]}",
         "--out-root", work / "merged", "--min-free-gb", 0.1])
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    ids = torch.tensor([[248045, 846, 198, 9707, 11, 1879, 0]])   # a few real token ids
    ref = PeftModel.from_pretrained(Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32),
                                    ck[-1]).eval()
    mer = Qwen3_5ForConditionalGeneration.from_pretrained(work / "merged" / "c5", dtype=torch.float32).eval()
    with torch.no_grad():
        d = (ref(input_ids=ids).logits - mer(input_ids=ids).logits).abs().max().item()
    print(f"merged vs PEFT logits: max |diff| {d:.4g} (bf16 merge of a lr-1e-2 adapter)")
    assert d < 0.5, "merged model does not reproduce the adapter"
    run([py, SCRIPTS / "qwen_lora_gate.py", "reference", "--model", model_dir, "--data-dir", data,
         "--adapters", f"c5={ck[-1]}", "--out", work / "gate_ref.json", "--n", 1, "--max-len", 4096])
    ref_json = json.loads((work / "gate_ref.json").read_text())
    e0 = ref_json["examples"][0]
    gap = sum(abs(x - y) for x, y in zip(e0["logprobs"]["base"], e0["logprobs"]["c5"])) / len(e0["target_positions"])
    print(f"gate reference: {len(e0['target_positions'])} target tokens; HF adapter vs base differ by {gap:.3f} nats/token")
    print("CPU SMOKE PASSED")


if __name__ == "__main__":
    main()
