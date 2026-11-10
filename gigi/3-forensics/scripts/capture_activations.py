"""Replay the hero transcripts through the model itself and save activations (prepared 10 Nov 2026 for the NLA / J-space
work on the heroes gemma-3 27B and Qwen3.6 27B; runs on a rented Vast box with >= 80 GB, never on the laptop).

For every run folder (a messages.json as uploaded to the HF transcripts dataset, or a results/raw run with step-*/messages.json):
  1. rebuild the exact token stream the served model saw: the stored conversation plus the environment's tool schemas,
     through the model's own chat template (Qwen3.6: native tools; gemma-3: the harness's text-mode tool preamble and
     ```tool_code``` blocks, the same function the provider used), no generation;
  2. one forward pass in bf16 with hooks on the chosen layers (one conversation at a time);
  3. save float32 activations per layer: every token of every assistant reply (concatenated, with the reply boundaries in
     index.json), the per-reply mean and the per-reply last token, as one safetensors file per run, plus the input ids;
  4. upload the run folder to the HF dataset right away (pipeline stage), skip runs already there (resumable).
Runs longer than --max-tokens are skipped and listed (the 82k-context Qwen3.6 runs can reach 190k tokens of text).
  python3 capture_activations.py --model-tag qwen3.6-27b --runs-root /data/transcripts --stages hero9_qwen36 hero9_qwen36_key hero9_qwen36_all \
      --layers 16,32,48 --out /data/activations --repo model-hermeneutics/gigi-forensics-transcripts
Needs HF_TOKEN in a .env next to this script (or in the environment). CPU dry run: --model-id Qwen/Qwen2.5-0.5B-Instruct --device cpu --no-upload --limit 2.
"""
import argparse, importlib.util, json, os, pathlib, re, sys, time, traceback

HERE = pathlib.Path(__file__).resolve().parent
MODELS = {"gemma-3-27b": dict(hub="google/gemma-3-27b-it", text_tools=True, env_dir="google-gemma-3-27b-it"),
          "qwen3.6-27b": dict(hub="Qwen/Qwen3.6-27B", text_tools=False, env_dir="qwen-qwen3.6-27b")}
FINISH_TOOLS = ("end_task", "done", "end_session")


def load_env():
    for p in (HERE / ".env", HERE.parents[1] / ".env", HERE.parents[2] / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


# ---------- tool schemas, exactly as the harness built them ----------
def env_registry(envs_root):
    src = (envs_root / "scripts" / "env_registry.py").read_text()
    return dict(re.findall(r'"([a-z_0-9]+)":\s*"(environments/[^"]+)/Dockerfile"', src))


def code_dir(envs_root, environment, registry):
    d = envs_root / registry.get(environment, f"environments/{environment}")
    while not (d / "agent.py").exists() and d != envs_root:
        d = d.parent
    return d


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def resolve_tools(envs_root, environment, config, registry):
    """The tool list the provider was created with (then end_task appended when the hard rule had no finish tool)."""
    from agent_interp_envs.tool_calling import EXECUTE_COMMAND_TOOL
    cdir = code_dir(envs_root, environment, registry); note = ""
    tools = None
    tp = cdir / "tools.py"
    mod = _import(tp, f"envtools_{cdir.name}") if tp.exists() else None
    try:
        if mod is not None and hasattr(mod, "get_tools"):
            tools, note = list(mod.get_tools(config)), "tools.get_tools(config)"
        elif mod is not None and hasattr(mod, "ALL_TOOLS"):
            tools, note = list(mod.ALL_TOOLS), "tools.ALL_TOOLS"
        elif mod is not None and hasattr(mod, "build_tools"):
            names = (config.get("task") or {}).get("tools") or getattr(mod, "DEFAULT_TOOLS", ["execute_command"])
            tools, note = list(mod.build_tools(list(names))), f"tools.build_tools({list(names)})"
        else:
            agent_src = (cdir / "agent.py").read_text()
            m = re.search(r"tools=\[([^\]]*)\]", agent_src)
            if m:
                names = [n.strip() for n in m.group(1).split(",") if n.strip()]
                tools = [EXECUTE_COMMAND_TOOL if n == "EXECUTE_COMMAND_TOOL" else getattr(mod, n) for n in names]
                note = f"agent.py literal {names}"
            elif "def get_tools" in agent_src:
                amod = _import(cdir / "agent.py", f"envagent_{cdir.name}")
                tools, note = list(amod.get_tools(config)), "agent.get_tools(config)"
    except Exception as e:   # the box has the package but an environment module may need its container: say so, keep going
        tools, note = None, f"tool resolution failed: {type(e).__name__}: {e}"
    if tools is None:
        tools, note = [EXECUTE_COMMAND_TOOL], note or "fallback: execute_command only (CHECK)"
    agent = config.get("agent") or {}
    if agent.get("end_task_required"):
        names = {(t.get("function") or t).get("name") for t in tools}
        if not any(n in names for n in FINISH_TOOLS):
            from agent_interp_envs.end_task_required import END_TASK_TOOL
            tools = tools + [END_TASK_TOOL]; note += " + synthetic end_task"
    return tools, note


# ---------- the token stream ----------
def clean_messages(messages):
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        c = m.get("content"); c = c if isinstance(c, str) else ("" if c is None else json.dumps(c))
        n = {"role": role, "content": c}
        if role == "assistant" and m.get("tool_calls"):
            n["tool_calls"] = [{"id": tc.get("id"), "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}}
                               for tc in m["tool_calls"]]
        if role == "tool":
            n["tool_call_id"] = m.get("tool_call_id")
        out.append(n)
    return out


def render(tok, messages, tools, text_tools, add_gen):
    if text_tools:
        from agent_interp_envs.providers.openrouter_provider import _text_tools_messages
        msgs = _text_tools_messages(messages, tools); kw = {}
    else:
        msgs = messages; kw = {"tools": tools or None}
    ids = tok.apply_chat_template(msgs, add_generation_prompt=add_gen, tokenize=True, **kw)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def reply_spans(tok, messages, tools, text_tools):
    """[(message index, start, end, header_included)] over the full render; prefix-checked."""
    full = render(tok, messages, tools, text_tools, add_gen=False)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        post = render(tok, messages[:i + 1], tools, text_tools, add_gen=False)
        pre = render(tok, messages[:i], tools, text_tools, add_gen=True)
        header = False
        if full[:len(pre)] != pre or full[:len(post)] != post:
            pre = render(tok, messages[:i], tools, text_tools, add_gen=False); header = True   # the generation prompt differs from the real header: span from the previous turn's end
            if full[:len(post)] != post:
                raise ValueError(f"message {i}: the truncated render is not a prefix of the full render")
        spans.append((i, len(pre), len(post), header))
    return full, spans


# ---------- the model ----------
def decoder_layers(model, n_layers):
    import torch
    cands = [m for _, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n_layers]
    if not cands:
        raise RuntimeError(f"no ModuleList with {n_layers} layers found")
    return cands[0]


def capture(model, layers, ids, layer_idx, device):
    import torch
    store = {}
    hooks = []
    for li in layer_idx:
        def hook(mod, inp, out, li=li):
            h = out[0] if isinstance(out, tuple) else out
            store[li] = h[0].detach()
        hooks.append(layers[li].register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True, choices=list(MODELS))
    ap.add_argument("--model-id", default=None, help="override the hub id (CPU dry run with a tiny model)")
    ap.add_argument("--runs-root", required=True, help="folder with <stage>/.../run-N (HF transcripts snapshot or results/raw)")
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--envs-root", default=str(HERE.parent / "agent-interp-envs"))
    ap.add_argument("--configs-root", default=str(HERE.parent / "configs"), help="fallback configs/<environment>__<tag>.yaml")
    ap.add_argument("--layers", default="16,32,48")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="model-hermeneutics/gigi-forensics-transcripts")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=0, help="stop after N runs (smoke)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--keep-local", action="store_true")
    ap.add_argument("--no-token-level", action="store_true", help="save only the per-reply mean and last token (token-level float32 at 3 layers is ~360 GB for Qwen3.6's 6M reply tokens)")
    a = ap.parse_args()
    load_env()
    import torch, yaml
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(pathlib.Path(a.envs_root) / "src"))
    spec = MODELS[a.model_tag]; hub = a.model_id or spec["hub"]; text_tools = spec["text_tools"]
    envs_root = pathlib.Path(a.envs_root); registry = env_registry(envs_root)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"capture_{a.model_tag}.log", "a")
    say = lambda *x: (print(*x, flush=True), log.write(time.strftime("%F %T ") + " ".join(map(str, x)) + "\n"), log.flush())
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    say(f"loading {hub} on {device} ({a.dtype})")
    tok = AutoTokenizer.from_pretrained(hub)
    model = AutoModelForCausalLM.from_pretrained(hub, dtype=dtype, device_map=device if device != "cpu" else None, low_cpu_mem_usage=True)
    model.eval()
    cfg = AutoConfig.from_pretrained(hub); tc = getattr(cfg, "text_config", None) or cfg
    n_layers = tc.num_hidden_layers
    layer_idx = [int(x) for x in a.layers.split(",") if x.strip()]
    bad = [l for l in layer_idx if not 0 <= l < n_layers]
    if bad:
        sys.exit(f"layers {bad} out of range (model has {n_layers})")
    layers = decoder_layers(model, n_layers)
    api = None; done_remote = set()
    if not a.no_upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        done_remote = {f for f in api.list_repo_files(a.repo, repo_type="dataset") if f.startswith(f"activations/{a.model_tag}/") and f.endswith("index.json")}
    n_done = n_skip = 0
    for stage in a.stages:
        root = pathlib.Path(a.runs_root) / stage
        runs = sorted(p for p in root.rglob("run-*") if p.is_dir() and re.fullmatch(r"run-\d+", p.name) and spec["env_dir"] in p.parts
                      and ((p / "messages.json").exists() or any(p.glob("step-*/messages.json"))))
        say(f"{stage}: {len(runs)} runs")
        for run in runs:
            rel = run.relative_to(a.runs_root).as_posix()
            dest = out / a.model_tag / rel; remote = f"activations/{a.model_tag}/{rel}/index.json"
            if (dest / "index.json").exists() or remote in done_remote:
                continue
            try:
                mp = run / "messages.json"
                if not mp.exists():
                    mp = sorted(run.glob("step-*/messages.json"), key=lambda q: int(q.parent.name.split("-")[1]))[-1]
                messages = clean_messages(json.load(open(mp)))
                cp = next((p for p in (run / "config.yaml", run.parent / "config.yaml") if p.exists()), None)
                environment = None
                if cp is not None:
                    config = yaml.safe_load(open(cp)); environment = config.get("environment")
                if environment is None:   # HF layout without a config: configs/<environment>__<tag>.yaml by the folder name
                    parts = run.relative_to(root).parts
                    env_name = parts[0] if len(parts) >= 5 else next((k for k, v in registry.items() if parts[0] in v), parts[0])
                    cands = sorted(pathlib.Path(a.configs_root).glob(f"{env_name}__{a.model_tag}.yaml"))
                    if not cands:
                        say(f"  SKIP {rel}: no config found"); n_skip += 1; continue
                    config = yaml.safe_load(open(cands[0])); environment = config.get("environment")
                tools, tool_note = resolve_tools(envs_root, environment, config, registry)
                ids, spans = reply_spans(tok, messages, tools, text_tools)
                if len(ids) > a.max_tokens:
                    say(f"  SKIP {rel}: {len(ids)} tokens > {a.max_tokens}"); n_skip += 1; continue
                t0 = time.time()
                store = capture(model, layers, ids, layer_idx, device)
                tensors = {"input_ids": torch.tensor(ids, dtype=torch.int32)}
                for li, h in store.items():
                    toks = torch.cat([h[s:e] for _, s, e, _ in spans]).float().cpu() if spans else torch.zeros(0, h.shape[-1])
                    if not a.no_token_level:
                        tensors[f"L{li}.reply_tokens"] = toks
                    tensors[f"L{li}.reply_mean"] = torch.stack([h[s:e].float().mean(0) for _, s, e, _ in spans]).cpu() if spans else torch.zeros(0, h.shape[-1])
                    tensors[f"L{li}.reply_last"] = torch.stack([h[e - 1].float() for _, s, e, _ in spans]).cpu() if spans else torch.zeros(0, h.shape[-1])
                dest.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(dest / "activations.safetensors"))
                off = 0; replies = []
                for mi, s, e, header in spans:
                    replies.append({"message_index": mi, "start": s, "end": e, "offset_in_concat": off, "n_tokens": e - s, "header_included": header}); off += e - s
                json.dump({"run": rel, "model": hub, "model_tag": a.model_tag, "dtype": a.dtype, "layers": layer_idx, "n_layers": n_layers,
                           "hidden_size": tc.hidden_size, "n_tokens": len(ids), "n_reply_tokens": off, "token_level": not a.no_token_level, "replies": replies, "text_tools": text_tools,
                           "tools": [(t.get("function") or t).get("name") for t in tools], "tools_source": tool_note, "environment": environment,
                           "seconds": round(time.time() - t0, 1), "written": time.strftime("%F %T")}, open(dest / "index.json", "w"), indent=1)
                say(f"  {rel}: {len(ids)} tokens, {len(spans)} replies, {off} reply tokens, {time.time() - t0:.0f}s [{tool_note}]")
                if api is not None:
                    for attempt in range(4):
                        try:
                            api.upload_folder(folder_path=str(dest), path_in_repo=f"activations/{a.model_tag}/{rel}", repo_id=a.repo, repo_type="dataset",
                                              commit_message=f"activations {a.model_tag} {rel}")
                            break
                        except Exception as e:
                            say(f"  upload attempt {attempt + 1} failed: {e!r}"[:200]); time.sleep(30 * (attempt + 1))
                    else:
                        say(f"  UPLOAD FAILED {rel}: kept at {dest}")
                        a.keep_local = True
                    if not a.keep_local:
                        (dest / "activations.safetensors").unlink()
                n_done += 1
                if a.limit and n_done >= a.limit:
                    say(f"limit {a.limit} reached"); return
            except Exception as e:
                say(f"  FAILED {rel}: {type(e).__name__}: {e}"); log.write(traceback.format_exc()); n_skip += 1
    say(f"done: {n_done} runs captured, {n_skip} skipped or failed")


if __name__ == "__main__":
    main()
