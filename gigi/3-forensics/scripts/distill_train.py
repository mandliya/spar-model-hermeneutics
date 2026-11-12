"""LoRA SFT of google/gemma-3-27b-it on Qwen3-Next-80B's agentic traces (distill/README.md, distill/PLAN.md; 3 Nov 2026).

Data: train.jsonl / val.jsonl from scripts/distill_data.py build (one JSON per line, `messages` = the history rendered
as gemma sees it in text mode; the LAST message is the target, an assistant turn). Each example is rendered with
gemma's own chat template, exactly as vLLM renders the prompt at inference (apply_chat_template with
add_generation_prompt=True), and the loss is on the target turn only: the target's tokens through <end_of_turn>.
Examples longer than --max-len tokens are dropped, never truncated (a truncated history is not what gemma sees at
inference); the counts per environment and split go to the log.

Versions (pin them; the API surface used here was checked against these):
  pip install torch==2.8.0 transformers==4.57.1 peft==0.17.1 trl==0.24.0 accelerate==1.10.1 datasets==4.2.0 pillow==11.3.0
  optional: flash-attn==2.8.3 (--attn flash_attention_2)

Memory on one H100 80 GB (79.6 GiB usable), bf16, batch 1, gradient checkpointing:
  - base weights, bf16: gemma-3-27b-it is 27.4 B parameters (27.0 B text + 0.4 B SigLIP vision tower) = 51.1 GiB, frozen.
  - LoRA r=16 on q,k,v,o,gate,up,down of the 62 text layers (hidden 5376, MLP 21504, 32 q heads / 16 kv heads x 128):
    1.83 M parameters per layer, 113 M in all; fp32 weights + grads + two Adam moments = 16 B each = 1.7 GiB.
  - checkpointed activations: one bf16 hidden state per layer per token, 62 x 5376 x 2 B = 0.65 MiB per token:
    10.2 GiB at 16k tokens, 20.4 GiB at 32k. --offload-activations keeps them in pinned CPU memory instead
    (torch.autograd.graph.save_on_cpu around the forward, with REENTRANT checkpointing: its autograd.Function saves
    each layer's input through save_for_backward, which the hook moves to the CPU; non-reentrant checkpointing holds
    the inputs in a closure the hook never sees. ~20 GiB over PCIe each way per 32k step, ~1-2 s against ~15 s of compute).
  - recompute of one layer in backward (MLP gate/up/act 3 x 21504 x 2 B, attention projections, norms; their grads):
    roughly 0.35 MiB per token at peak, 5.6 GiB at 16k, 11 GiB at 32k. SDPA's memory-efficient kernel (or flash-attn)
    keeps attention linear in length; the sliding-window mask is materialised as bool, 0.25 GiB at 16k, 1 GiB at 32k.
  - logits: the full sequence x 262,208 vocabulary would be 8 GiB in bf16 at 16k plus a 16 GiB fp32 upcast, so they are
    never materialised: the loss reads the backbone's last hidden state at the target positions only and applies the
    tied lm_head in chunks of --logit-chunk positions under checkpoint (1024 x 262,208 x 6 B = 1.5 GiB per chunk).
  - CUDA context, allocator fragmentation: ~2-3 GiB (set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True).
  Sum: 51.1 + 1.7 + 10.2 + 5.6 + 0.5 + 3 + 3 = ~75 GiB at --max-len 16384 without offload (fits, not by much);
  51.1 + 1.7 + ~0.5 (offloaded) + 11 + 1 + 3 + 3 = ~71 GiB at --max-len 32768 with --offload-activations (needs ~25 GiB
  of free host RAM for the pinned stash). 32k without offload (~91 GiB) does not fit. Run --smoke first: it trains on
  the longest kept examples, so it fails on memory in minutes rather than hours in.

Usage (PLAN.md has the full box sequence):
  python distill_train.py train --data-dir distill/stratified --out /workspace/out/v1 --max-len 32768 --offload-activations --smoke
  python distill_train.py train --data-dir distill/stratified --out /workspace/out/v1 --max-len 32768 --offload-activations
  python distill_train.py merge --adapter /workspace/out/v1/adapter --out /workspace/out/v1/merged
Outputs in --out: adapter/ (PEFT adapter + tokenizer), checkpoints (checkpoint-N/, resumable with --resume),
log.json (args, versions, data counts, every train/eval log line, final per-split losses), val_per_example.jsonl
(target NLL per val example for the trained model and, unless --no-base-eval, for base gemma: same examples, adapter off).
"""
import argparse, collections, json, os, pathlib, random, sys, time


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data-dir", required=True, help="folder with train.jsonl and val.jsonl")
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="google/gemma-3-27b-it")
    t.add_argument("--max-len", type=int, default=16384, help="max tokens per example (prompt + target); longer are dropped")
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lora-dropout", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--max-steps", type=int, default=-1, help="optimizer steps; overrides --epochs when > 0")
    t.add_argument("--grad-accum", type=int, default=16, help="examples per optimizer step (batch size is 1)")
    t.add_argument("--warmup-ratio", type=float, default=0.03)
    t.add_argument("--eval-every", type=int, default=50, help="optimizer steps between evals on the val sample")
    t.add_argument("--eval-sample", type=int, default=150, help="val examples in each periodic eval (seeded sample)")
    t.add_argument("--save-every", type=int, default=50, help="optimizer steps between checkpoints (2 kept)")
    t.add_argument("--keep-checkpoints", action="store_true", help="keep every checkpoint, not the last 2 (12 Nov call: "
                   "re-infer each one to plot the Qwen3.6-gemma gap over training)")
    t.add_argument("--logit-chunk", type=int, default=1024)
    t.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    t.add_argument("--offload-activations", action="store_true", help="keep checkpointed activations in pinned CPU memory")
    t.add_argument("--envs", default=None, help="comma-separated environments to keep (default all)")
    t.add_argument("--limit-train", type=int, default=0, help="keep at most N train examples (seeded sample; 0 = all)")
    t.add_argument("--smoke", action="store_true", help="memory/plumbing gate: the 4 longest kept train examples, "
                   "4 val examples, 2 optimizer steps of 2, one eval, adapter save, peak memory printed")
    t.add_argument("--no-base-eval", action="store_true", help="skip base gemma's per-example val NLL at the end")
    t.add_argument("--final-val-sample", type=int, default=0, help="per-example final eval on a seeded sample of N val "
                   "examples (0 = all; every example from a hacked run is always kept)")
    t.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --out")
    t.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--model", default="google/gemma-3-27b-it")
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    return ap.parse_args()


# ---------------------------------------------------------------- data

def encode(tok, msgs, max_len):
    """(input_ids, labels) for one example, or (None, reason). The prompt is rendered as vLLM renders it at inference
    (generation prompt appended); the completion is what the template puts after it for the target turn, i.e. the
    target with surrounding whitespace trimmed (gemma's template trims) and <end_of_turn>; the trailing newline the
    template adds after <end_of_turn> is not trained (generation stops at <end_of_turn>). Prompt and completion are
    tokenized separately, as at inference, where the server tokenizes the prompt and the model continues it."""
    if not msgs or msgs[-1]["role"] != "assistant":
        return None, "last message is not an assistant turn"
    if sum(len(m["content"]) for m in msgs) > 16 * max_len:   # cheap pre-filter: far beyond any chars/token ratio
        return None, "too long"
    try:
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
    except Exception as e:   # e.g. the template's "roles must alternate" check
        return None, f"template error: {type(e).__name__}"
    if not full.startswith(prompt):
        return None, "template: prompt is not a prefix of the full rendering"
    completion = full[len(prompt):].rstrip("\n")
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        return None, "too long"
    return (p + c, [-100] * len(p) + c), None


def encode_turns(tok, msgs, train_turns, max_len):
    """(input_ids, labels) for a whole run with the loss on the assistant turns in train_turns (0-based among the
    assistant turns; distill_data.py build --trajectories), or (None, reason). Built turn by turn exactly as encode()
    builds one turn: the context up to turn i is the template's rendering with the generation prompt, the turn is what
    the template adds after it, without the trailing newline. Context pieces and turns are tokenized separately, as in
    encode(); each piece starts at a special token (<start_of_turn>, <end_of_turn>), so the boundaries are the
    inference ones. A run longer than max_len is cut before the first turn that does not fit; its later turns are
    dropped, as the per-turn build drops examples longer than max_len."""
    want = set(train_turns); ids, labels, text = [], [], ""
    idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    for t, i in enumerate(idx):
        try:
            prompt = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(msgs[:i + 1], tokenize=False)
        except Exception as e:
            return None, f"template error: {type(e).__name__}"
        if not prompt.startswith(text) or not full.startswith(prompt):
            return None, "template: renderings are not prefixes of each other"
        ctx = tok(prompt[len(text):], add_special_tokens=False)["input_ids"]
        turn = tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
        if len(ids) + len(ctx) + len(turn) > max_len:
            break
        ids += ctx + turn; labels += [-100] * len(ctx) + (turn if t in want else [-100] * len(turn))
        text = full.rstrip("\n")
    if not any(x != -100 for x in labels):
        return None, "too long"
    return (ids, labels), None


def load_split(path, tok, max_len, envs, counts, split):
    rows = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if envs and e["env"] not in envs:
                continue
            enc, why = (encode_turns(tok, e["messages"], e["train_turns"], max_len) if "train_turns" in e
                        else encode(tok, e["messages"], max_len))
            c = counts[split][e["env"]]; c["seen"] += 1
            if enc is None:
                c["dropped: " + why] += 1; continue
            ids, labels = enc
            c["kept"] += 1; c["tokens"] += len(ids); c["target_tokens"] += sum(1 for x in labels if x != -100)
            rows.append({"input_ids": ids, "labels": labels,
                         "meta": {k: e.get(k) for k in ("env", "run", "group", "step", "kind", "hacked")}})
    return rows


def collate(batch):
    import torch
    n = max(len(b["input_ids"]) for b in batch)   # batch size 1 in practice; padding kept correct anyway
    ids = torch.zeros(len(batch), n, dtype=torch.long); lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros(len(batch), n, dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"]); lab[i, :k] = torch.tensor(b["labels"]); att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ---------------------------------------------------------------- loss

def _ce_sum(h, y, w):
    import torch.nn.functional as F
    return F.cross_entropy((h @ w.t()).float(), y, reduction="sum")


def target_nll(model, inputs, chunk, offload=False):
    """(summed NLL over target tokens, number of target tokens). Runs the backbone (Gemma3Model: text layers + final
    norm) and applies the tied lm_head only at positions whose next token is a target token, in checkpointed chunks,
    so the full-sequence logits are never materialised. gemma-3 has no final logit soft-capping (config: null), so
    this equals the model's own logits."""
    import contextlib, torch
    from torch.utils.checkpoint import checkpoint
    core = model.get_base_model() if hasattr(model, "get_base_model") else model   # the HF model, LoRA layers injected
    ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload and torch.is_grad_enabled() else contextlib.nullcontext()
    with ctx:
        out = core.model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), use_cache=False)
        h = out.last_hidden_state
        tgt = inputs["labels"][:, 1:]; mask = tgt != -100
        hs, ys = h[:, :-1][mask], tgt[mask]
        w = core.get_output_embeddings().weight
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, hs.shape[0], chunk):
            if torch.is_grad_enabled():
                total = total + checkpoint(_ce_sum, hs[i:i + chunk], ys[i:i + chunk], w, use_reentrant=False)
            else:
                total = total + _ce_sum(hs[i:i + chunk], ys[i:i + chunk], w)
    return total, int(mask.sum())


# ---------------------------------------------------------------- train

def git_commit(path):
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip() or None


def train(args):
    import torch, transformers, peft, trl, datasets
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, TrainerCallback
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    data = pathlib.Path(args.data_dir); envs = set(args.envs.split(",")) if args.envs else None
    tok = AutoTokenizer.from_pretrained(args.model)
    counts = {s: collections.defaultdict(collections.Counter) for s in ("train", "val")}
    t0 = time.time()
    tr = load_split(data / "train.jsonl", tok, args.max_len, envs, counts, "train")
    va = load_split(data / "val.jsonl", tok, args.max_len, envs, counts, "val")
    print(f"tokenized in {time.time() - t0:.0f}s: train {len(tr)} kept, val {len(va)} kept (max_len {args.max_len})", flush=True)
    rng = random.Random(args.seed)
    if args.smoke:
        tr = sorted(tr, key=lambda r: -len(r["input_ids"]))[:4]; va = sorted(va, key=lambda r: -len(r["input_ids"]))[:4]
        args.max_steps, args.grad_accum, args.eval_every, args.save_every = 2, 2, 2, 2
        print("SMOKE: longest train lengths", [len(r["input_ids"]) for r in tr], flush=True)
    elif args.limit_train and len(tr) > args.limit_train:
        tr = rng.sample(tr, args.limit_train)
    if not tr or not va:
        sys.exit("no train or val example left after filtering")
    eval_sample = va if len(va) <= args.eval_sample else rng.sample(va, args.eval_sample)
    ds = lambda rows: datasets.Dataset.from_list([{"input_ids": r["input_ids"], "labels": r["labels"]} for r in rows])

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
        device_map={"": 0} if torch.cuda.is_available() else None)
    model.config.use_cache = False
    if args.offload_activations:   # reentrant checkpointing needs a grad-requiring input to propagate LoRA gradients
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      # text layers only: the SigLIP tower also has q_proj/k_proj/v_proj, so a bare name list would hit it
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(out), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=args.max_steps, lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio, weight_decay=0.0,
        max_grad_norm=1.0, optim="adamw_torch", bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": bool(args.offload_activations)},
        logging_steps=1, eval_strategy="steps", eval_steps=args.eval_every, save_strategy="steps",
        save_steps=args.save_every, save_total_limit=None if args.keep_checkpoints else 2, report_to="none", seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True}, packing=False,
        prediction_loss_only=True, dataloader_num_workers=0, eval_on_start=True)

    class TargetOnlySFT(SFTTrainer):
        """SFTTrainer whose loss is target_nll: summed NLL over the target tokens, divided by the number of target
        tokens in the whole accumulated batch when the Trainer passes it (token-weighted, as Trainer expects with
        model_accepts_loss_kwargs), else by the example's own count (eval: eval_loss = mean of per-example means)."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.model_accepts_loss_kwargs = True
            self.acc = {"nll": 0.0, "n": 0, "tokens": 0}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            nll, n = target_nll(model, inputs, args.logit_chunk, args.offload_activations and model.training)
            if model.training:
                self.acc["nll"] += float(nll.detach()); self.acc["n"] += n; self.acc["tokens"] += int(inputs["input_ids"].numel())
            denom = num_items_in_batch if (model.training and num_items_in_batch is not None) else max(n, 1)
            loss = nll / denom
            return (loss, {}) if return_outputs else loss

    log = {"args": vars(args), "versions": {m.__name__: m.__version__ for m in (torch, transformers, peft, trl, datasets)},
           "git_commit": git_commit(pathlib.Path(__file__).parent), "data_dir": str(data),
           "data": {s: {e: dict(c) for e, c in sorted(counts[s].items())} for s in counts},
           "n_train": len(tr), "n_val": len(va), "n_eval_sample": len(eval_sample), "history": []}
    if (data / "manifest.json").exists():
        log["data_manifest"] = {k: v for k, v in json.loads((data / "manifest.json").read_text()).items()
                                if k not in ("counts", "excluded_runs", "runs_not_in_sample", "cheating_families")}

    if args.resume and (out / "log.json").exists():   # keep the history of the interrupted run
        log["history"] = json.loads((out / "log.json").read_text()).get("history", [])

    def write_log():
        tmp = out / "log.json.tmp"; tmp.write_text(json.dumps(log, indent=1)); tmp.replace(out / "log.json")

    class JsonLog(TrainerCallback):
        def __init__(self):
            self.t = time.time(); self.tokens = 0

        def on_log(self, a, state, control, logs=None, **kw):
            rec = {"step": state.global_step, "epoch": state.epoch, "time": round(time.time(), 1), **(logs or {})}
            acc = trainer.acc
            if acc["n"]:   # exact per-token train NLL over the examples since the last log line
                rec["train_nll_per_target_token"] = acc["nll"] / acc["n"]; rec["target_tokens"] = acc["n"]
                dt = time.time() - self.t; rec["tokens_per_s"] = acc["tokens"] / dt if dt > 0 else None
                self.tokens += acc["tokens"]; rec["tokens_seen"] = self.tokens
                acc.update(nll=0.0, n=0, tokens=0)
            self.t = time.time()
            if torch.cuda.is_available():
                rec["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            log["history"].append(rec); write_log()
            print(json.dumps(rec), flush=True)

    trainer = TargetOnlySFT(model=model, args=cfg, train_dataset=ds(tr), eval_dataset=ds(eval_sample),
                            processing_class=tok, data_collator=collate, callbacks=[JsonLog()])
    last = None
    if args.resume:
        cks = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        last = str(cks[-1]) if cks else None
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(out / "adapter")); tok.save_pretrained(str(out / "adapter"))
    print("adapter saved:", out / "adapter", flush=True)

    # per-example target NLL on the whole val split: trained model, then base gemma (adapter off) on the same examples
    model = trainer.model; model.eval(); dev = next(model.parameters()).device
    per = []; fin = va
    if args.final_val_sample and len(va) > args.final_val_sample:
        hacked = [r for r in va if r["meta"]["hacked"]]; rest = [r for r in va if not r["meta"]["hacked"]]
        fin = hacked + rng.sample(rest, max(0, min(len(rest), args.final_val_sample - len(hacked))))
    log["n_final_val"] = len(fin)
    with torch.no_grad():
        for i, r in enumerate(fin):
            b = {k: v.to(dev) for k, v in collate([r]).items()}
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                nll, n = target_nll(model, b, args.logit_chunk)
                rec = {**r["meta"], "tokens": len(r["input_ids"]), "target_tokens": n, "nll_distilled": float(nll) / max(n, 1)}
                if not args.no_base_eval:
                    with model.disable_adapter():
                        nb, _ = target_nll(model, b, args.logit_chunk)
                    rec["nll_base"] = float(nb) / max(n, 1)
            per.append(rec)
            if i % 100 == 0:
                print(f"final val {i}/{len(fin)}", flush=True)
    with open(out / "val_per_example.jsonl", "w") as f:
        for rec in per:
            f.write(json.dumps(rec) + "\n")
    summ = collections.defaultdict(lambda: collections.Counter())
    for rec in per:
        for key in ("all", f"env={rec['env']}", f"hacked={rec['hacked']}"):
            s = summ[key]; s["n"] += 1; s["tok"] += rec["target_tokens"]
            s["distilled"] += rec["nll_distilled"] * rec["target_tokens"]
            if "nll_base" in rec:
                s["base"] += rec["nll_base"] * rec["target_tokens"]
    log["final_val"] = {k: {"examples": s["n"], "target_tokens": s["tok"],
                            "nll_per_token_distilled": s["distilled"] / s["tok"] if s["tok"] else None,
                            "nll_per_token_base": (s["base"] / s["tok"] if s["tok"] else None) if not args.no_base_eval else None}
                        for k, s in sorted(summ.items())}
    if torch.cuda.is_available():
        log["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    write_log()
    print(json.dumps(log["final_val"].get("all")), flush=True)
    if args.smoke:
        import math
        norms = [h["grad_norm"] for h in log["history"] if "grad_norm" in h]
        if not norms or not all(isinstance(g, (int, float)) and math.isfinite(g) and g > 0 for g in norms):
            sys.exit(f"SMOKE FAILED: grad norms {norms} (NaN/zero: no usable gradient; with --offload-activations, "
                     f"retry without it at a shorter --max-len)")
        print(f"SMOKE OK: grad norms {norms}; peak {log.get('peak_mem_gib')} GiB at {max(len(r['input_ids']) for r in tr)} tokens", flush=True)


def merge(args):
    """Base + adapter folded into one bf16 checkpoint (~55 GB) that vLLM serves like the original model; the
    processor files are copied so the multimodal architecture loads unchanged."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
    from peft import PeftModel
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    try:   # vLLM loads Gemma3ForConditionalGeneration with its processor (needs pillow)
        AutoProcessor.from_pretrained(args.model).save_pretrained(args.out)
    except Exception as e:
        print(f"WARNING: processor not saved ({type(e).__name__}: {e}); tokenizer only. Copy preprocessor_config.json "
              f"and processor_config.json from {args.model} before serving.")
        AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print("merged:", args.out)


if __name__ == "__main__":
    a = parse_args()
    train(a) if a.cmd == "train" else merge(a)
