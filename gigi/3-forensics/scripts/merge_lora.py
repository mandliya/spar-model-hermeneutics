"""Merge LoRA checkpoints into full bf16 models, for serving without vLLM's LoRA path (phase 2, 19 Nov 2026).

Why: on Qwen3.5/3.6, vLLM has served LoRA adapters as silent no-ops (vllm-project/vllm#49354: identical output to the
base model). scripts/qwen_lora_gate.py detects that; this is the fallback. A merged Qwen3.6 27B is ~54 GB on disk and
in GPU memory, so the four checkpoints do not fit on one H200 together: merge, serve one, re-run its tasks, stop the
server, move to the next (or merge them all first if the disk has ~220 GB free).

  /workspace/venv-train-qwen/bin/python merge_lora.py --student qwen36 --adapters c23=/workspace/out/q1/checkpoint-23,... \
      --out-root /workspace/out/q1/merged [--only c46]

Each merge: base on the CPU in bf16 (~60 GB RAM), PeftModel.merge_and_unload, save_pretrained in 5 GB shards, plus the
processor and tokenizer (with the chat template) so vLLM loads the multimodal class unchanged. Writes merged/<name>/
and skips names already merged (merged/<name>/.done). Checks the merged config keeps the base architecture.
"""
import argparse, json, pathlib, shutil, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="qwen36", choices=["gemma3", "qwen36"])
    ap.add_argument("--model", default=None, help="base model (default: the student's HF id)")
    ap.add_argument("--adapters", required=True, help="comma list name=path")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--only", default=None, help="merge only these names (comma list)")
    ap.add_argument("--min-free-gb", type=float, default=60, help="stop before a merge with less free disk (a merged "
                    "Qwen3.6 27B is ~54 GB)")
    a = ap.parse_args()
    import distill_train as DT
    model_id = a.model or DT.STUDENTS[a.student]
    root = pathlib.Path(a.out_root); root.mkdir(parents=True, exist_ok=True)
    ads = [x.split("=", 1) for x in a.adapters.split(",")]
    only = set(a.only.split(",")) if a.only else None
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from peft import PeftModel
    base_arch = AutoConfig.from_pretrained(model_id).architectures
    for name, path in ads:
        if only and name not in only:
            continue
        out = root / name
        if (out / ".done").exists():
            print(f"{name}: already merged ({out})"); continue
        need = a.min_free_gb * 1e9
        if shutil.disk_usage(root).free < need:
            sys.exit(f"{name}: less than {need / 1e9:.0f} GB free in {root}: stopping before a partial write")
        t0 = time.time()
        model = DT.load_student(a.student, model_id, None, "cpu")
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
        del model
        try:
            AutoProcessor.from_pretrained(model_id).save_pretrained(out)
        except Exception as e:   # noqa: BLE001
            print(f"WARNING {name}: processor not saved ({type(e).__name__}: {e}); tokenizer only")
        AutoTokenizer.from_pretrained(path if (pathlib.Path(path) / "tokenizer_config.json").exists() else model_id).save_pretrained(out)
        arch = json.loads((out / "config.json").read_text()).get("architectures")
        if arch != base_arch:
            sys.exit(f"{name}: merged config architectures {arch} != base {base_arch}")
        (out / ".done").write_text(json.dumps({"adapter": path, "base": model_id, "seconds": round(time.time() - t0)}))
        print(f"{name}: merged into {out} in {time.time() - t0:.0f}s", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
