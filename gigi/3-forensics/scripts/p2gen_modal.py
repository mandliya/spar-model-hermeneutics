"""p2gen_modal.py (24 Nov 2026): train Qwen3.6 27B on a general training set, and serve the saved copies, on Modal GPUs.
Design note: phase2/general_set/PIPELINE.md. NOTHING HERE HAS RUN ON MODAL YET (written and checked without spending).

It is the DeepSeek round (p2ds) moved from one rented Vast box to Modal, so three trainings can run at the same time:
  train  = scripts/p2q_train.sh       the same distill_train.py, called with the same arguments, on one H200
  serve  = scripts/p2q_serve.sh + p2q_copies.sh + p2q_load_copy.py     the same vLLM 0.32.0 flags, the same load check
Each training set has a name, DISTILL_V style: p2gen_toolscale, p2gen_wildchat, p2gen_tbench. Everything of a set
lives under that name on the Modal volume <MODAL_VOLUME_PREFIX>p2gen (gigi-forensics-p2gen):
  data/<dv>/train.jsonl, val.jsonl     uploaded from the launch box (modal volume put); never in git, never on HF
  out/<dv>/checkpoint-<step>/          a saved copy every half pass (trainer's own resumable checkpoints)
  out/<dv>/copies.json, copies_ready.txt, DONE
  logs/<dv>/stages.txt, train_<try>.log, ...
  serve/<dv>/<name>.json               a running server's address and what it serves; <name>.stop asks it to end
The model weights sit on a second volume (<prefix>p2gen-hf) so they are downloaded once (Qwen/Qwen3.6-27B is public:
no Hugging Face token is needed, and none is used).

Access, reusing what exists (no new Modal secret, no credential moved): the server's password is VLLM_API_KEY from the
Modal secret vllm-secret<MODAL_SECRET_SUFFIX> (vllm-secret-gigi), the one the test runs already mount. vLLM reads it
from the environment, so it never appears in a command line or a log.

Launch from the launch box (scripts/p2gen_launch.sh does all of this; `modal run --detach` keeps a single function
alive after the launcher disconnects):
  modal run p2gen_modal.py::check                                  # a few cents: can this workspace start a function?
  modal run --detach p2gen_modal.py::train --dv p2gen_toolscale
  modal run --detach p2gen_modal.py::serve --dv p2gen_toolscale --copies c8 --hours 8
Environment read when launching (and baked into the image so the container sees the same values): MODAL_VOLUME_PREFIX,
MODAL_SECRET_SUFFIX, P2GEN_GPU (default H200), P2GEN_SERVE_GPU (default H200), P2GEN_TRAIN_MEM_MB, P2GEN_CUDA_IMAGE.
"""
import json, os, pathlib, re, subprocess, sys, time

import modal

HERE = pathlib.Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.6-27B"
REV = "8f33b98937c6031bb49e713dee3e8e185aafd26f"        # the revision of every Qwen3.6 run so far (p2q_setup.sh)
PREFIX = os.environ.get("MODAL_VOLUME_PREFIX", "gigi-forensics-")
SUFFIX = os.environ.get("MODAL_SECRET_SUFFIX", "-gigi")
GPU = os.environ.get("P2GEN_GPU", "H200")                # p2ds trained on an H200: peak 75.9 GiB at 32,765 tokens
SERVE_GPU = os.environ.get("P2GEN_SERVE_GPU", "H200")    # p2ds served on an H200 (KV cache ~1.06M tokens)
TRAIN_MEM = int(os.environ.get("P2GEN_TRAIN_MEM_MB", "65536"))   # --offload-activations pins ~20 GiB of host memory
CUDA_IMAGE = os.environ.get("P2GEN_CUDA_IMAGE", "nvidia/cuda:12.9.0-devel-ubuntu22.04")   # devel: nvcc for the kernels
BAKED = {"MODAL_VOLUME_PREFIX": PREFIX, "MODAL_SECRET_SUFFIX": SUFFIX, "P2GEN_GPU": GPU, "P2GEN_SERVE_GPU": SERVE_GPU,
         "P2GEN_TRAIN_MEM_MB": str(TRAIN_MEM), "P2GEN_CUDA_IMAGE": CUDA_IMAGE,
         "HF_HOME": "/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
MAX_LEN = 32768
# p2q_train.sh's training arguments, word for word (2 passes, a saved copy every half pass, the 32,768-token cut)
TRAIN_ARGS = ["--max-len", str(MAX_LEN), "--offload-activations", "--epochs", "2", "--keep-checkpoints",
              "--save-every", "0.25", "--eval-every", "0.25", "--final-val-sample", "0", "--resume"]
# p2q_serve.sh's server arguments, word for word, except the password (read from the environment, see above)
VLLM_ARGS = ["--revision", REV, "--served-model-name", "qwen36-base", "--language-model-only", "--dtype", "bfloat16",
             "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
             "--reasoning-parser", "qwen3",
             "--default-chat-template-kwargs", '{"enable_thinking": true, "preserve_thinking": true}',
             "--enable-lora", "--max-lora-rank", "16", "--max-loras", "4", "--max-model-len", "65536",
             "--gpu-memory-utilization", "0.92", "--host", "0.0.0.0", "--port", "8000"]

app = modal.App("p2gen")
vol = modal.Volume.from_name(f"{PREFIX}p2gen", create_if_missing=True, version=2)
hf_vol = modal.Volume.from_name(f"{PREFIX}p2gen-hf", create_if_missing=True, version=2)
VOLS = {"/vol": vol, "/hf": hf_vol}

# scripts/venv_train_qwen.sh's pins (checked on the p2ds box: results/logs/p2ds/gpu_box/freeze_train_qwen.txt)
train_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12").entrypoint([])
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.13.0", "transformers==5.23.0", "peft==0.21.3", "trl==1.18.1", "accelerate==1.16.0",
                    "datasets==5.2.0", "pillow==11.3.0", "flash-linear-attention==0.6.0", "fla-core==0.6.0",
                    "huggingface_hub[hf_xet]", "jinja2", "sentencepiece", "pyyaml", "ninja", "packaging",
                    extra_index_url="https://download.pytorch.org/whl/cu129")
    # compiles against the torch above. CC/CXX: Modal's add_python is a clang-built Python, so without them the build asks
    # for clang++, which the image lacks (24 Nov: the first check's image build failed on that); the Vast box used gcc.
    .pip_install("causal-conv1d==1.7.0", extra_options="--no-build-isolation", env={"CC": "gcc", "CXX": "g++"})
    .env(BAKED)
    .add_local_file(HERE / "distill_train.py", "/root/distill_train.py")
)
# scripts/p2q_setup.sh's serving venv: vllm 0.32.0 (it compiles a kernel at start-up: needs ninja and nvcc)
vllm_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12").entrypoint([])
    .uv_pip_install("vllm==0.32.0", "huggingface_hub[hf_xet]", "ninja")
    .env({**BAKED, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "CUDA_HOME": "/usr/local/cuda"})
)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _say(dv, msg):
    """One line per event in logs/<dv>/stages.txt on the volume (the launch box polls it) and in the app's own log."""
    d = pathlib.Path("/vol/logs") / dv; d.mkdir(parents=True, exist_ok=True)
    with open(d / "stages.txt", "a") as f:
        f.write(f"{_now()} {msg}\n")
    print(f"{_now()} {msg}", flush=True)
    vol.commit()


def _dv_ok(dv):
    if not re.fullmatch(r"p2gen_[a-z0-9]+", dv):
        sys.exit(f"dv {dv!r}: expected p2gen_<name> (letters and digits), e.g. p2gen_toolscale")


def _model_dir():
    """Qwen3.6 27B at the pinned revision, from the weights volume (downloaded on first use, ~54 GB)."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL, revision=REV)
    hf_vol.commit()
    return p


def _checkpoints(out):
    """Complete saved copies, oldest first. The trainer writes trainer_state.json last, so a folder without it was cut
    off while being written (a preemption) and is not a copy."""
    cks = [c for c in out.glob("checkpoint-*") if c.name.split("-")[1].isdigit() and (c / "trainer_state.json").exists()
           and (c / "adapter_model.safetensors").exists()]
    return sorted(cks, key=lambda c: int(c.name.split("-")[1]))


@app.function(image=modal.Image.debian_slim(python_version="3.12").env(BAKED), volumes=VOLS, timeout=600)
def check(dv: str = ""):
    """A CPU-only function that lists what is on the volume. It answers one question for a few cents before any GPU is
    asked for: does this workspace start our functions at all? (A spawned function once stayed pending in spar2026.)"""
    for top in ("data", "out", "serve", "logs"):
        d = pathlib.Path("/vol") / top / dv
        print(top, sorted(p.name for p in d.iterdir()) if d.exists() else "absent")
    print("weights volume:", sorted(p.name for p in pathlib.Path("/hf").glob("hub/*")))
    print("P2GEN CHECK OK")


@app.function(image=train_image, gpu=GPU, cpu=8, memory=TRAIN_MEM, volumes=VOLS, timeout=24 * 3600,
              retries=modal.Retries(max_retries=4, backoff_coefficient=1.0, initial_delay=30.0))
def train(dv: str, skip_smoke: bool = False, smoke_only: bool = False, extra_nll: str = ""):
    """Smoke test (the 4 longest examples), 2 passes with a saved copy every half pass, then the loss of the untrained
    model and of every copy on the 20% check set: scripts/p2q_train.sh, step for step. Safe to start again at any point:
    a GPU function can be preempted, and Modal then calls it again with the same input. Finished steps are skipped,
    training resumes from the last complete saved copy, and the loss file is appended to.
    extra_nll: another data folder on the volume (e.g. data/p2ds) whose val/test files are scored too."""
    _dv_ok(dv)
    data = pathlib.Path("/vol/data") / dv; out = pathlib.Path("/vol/out") / dv; logs = pathlib.Path("/vol/logs") / dv
    for f in ("train.jsonl", "val.jsonl"):
        if not (data / f).exists():
            sys.exit(f"{data / f} is not on the volume: upload the set first (p2gen_launch.sh data)")
    out.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        _say(dv, "train: already done (out/DONE): nothing to do"); return
    if (out / "FAILED").exists():   # Modal calls a failed function again (retries cover preemptions): a real failure
        print("out/FAILED is there:", (out / "FAILED").read_text()); return   # is not repeated. Remove the file to retry.
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    _say(dv, f"START p2gen train {dv} on {gpu} (asked for {GPU})")
    model = _model_dir()
    py = [sys.executable, "/root/distill_train.py"]
    base = ["--student", "qwen36", "--model", model, "--data-dir", str(data)]

    def run(args, log):
        with open(logs / log, "a") as f:
            return subprocess.run(py + args, stdout=f, stderr=subprocess.STDOUT).returncode

    if not skip_smoke and not (out / "SMOKE_OK").exists():
        _say(dv, "START train_smoke (32,768 tokens)")
        rc = run(["train", *base, "--out", "/tmp/p2gen_smoke", "--max-len", str(MAX_LEN), "--offload-activations", "--smoke"], "train_smoke.log")
        tail = (logs / "train_smoke.log").read_text(errors="replace")
        ok = re.search(r"SMOKE OK.*", tail)
        if rc != 0 or not ok:
            (out / "FAILED").write_text(f"{_now()} smoke\n")
            _say(dv, "TRAINING SMOKE FAILED (logs/train_smoke.log). Nothing else was run."); return
        (out / "SMOKE_OK").write_text(ok.group(0) + "\n"); _say(dv, f"train_smoke OK: {ok.group(0)}")
    if smoke_only:
        return

    import threading
    stop = threading.Event(); ready = out / "copies_ready.txt"

    def note_copies():   # each complete saved copy is named once in copies_ready.txt, and the volume is committed
        seen = set(ready.read_text().split()) if ready.exists() else set()
        for c in _checkpoints(out):
            n = "c" + c.name.split("-")[1]
            if n not in seen and time.time() - (c / "trainer_state.json").stat().st_mtime > 60:
                ep = json.loads((c / "trainer_state.json").read_text()).get("epoch")
                with open(ready, "a") as f:
                    f.write(n + "\n")
                _say(dv, f"CHECKPOINT READY {c.name} (epoch {round(ep, 2) if ep is not None else '?'})")

    def committer():
        while not stop.wait(60):
            try:
                note_copies(); vol.commit()
            except Exception as e:   # noqa: BLE001  a failed commit must never stop the training
                print("commit:", type(e).__name__, e, flush=True)

    for c in out.glob("checkpoint-*"):   # cut off while being written: set aside, never resumed from, never deleted
        if c.is_dir() and not (c / "trainer_state.json").exists():
            c.rename(out / f"broken-{c.name}-{int(time.time())}"); _say(dv, f"set aside an incomplete {c.name}")
    th = threading.Thread(target=committer, daemon=True); th.start()
    ok = False
    for attempt in (1, 2, 3):
        n = len([f for f in logs.glob("train_*.log") if f.name != "train_smoke.log"]) + 1   # one log per try, kept
        _say(dv, f"START train (2 passes, a saved copy every half pass; resumes from {([c.name for c in _checkpoints(out)] or ['the start'])[-1]})")
        if run(["train", *base, "--out", str(out), *TRAIN_ARGS], f"train_{n}.log") == 0:
            ok = True; break
        _say(dv, f"train ended with an error (logs/train_{n}.log); resuming")
    stop.set(); th.join(timeout=120)
    if not ok:
        (out / "FAILED").write_text(f"{_now()} training\n")
        _say(dv, "TRAINING FAILED three times (logs/train_*.log)."); return

    # the copy after the last pass: the trainer saves every ceil(steps / 4) steps, so when the number of steps is not a
    # multiple of 4 the last saved copy is a few steps short of the end and the finished model is only in adapter/.
    # It is added as its own checkpoint folder so the tests can load it like the others.
    hist = json.loads((out / "log.json").read_text())["history"]
    last = max(h["step"] for h in hist); cks = _checkpoints(out)
    if not cks or int(cks[-1].name.split("-")[1]) < last:
        import shutil
        d = out / f"checkpoint-{last}"; d.mkdir(exist_ok=True)
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(out / "adapter" / f, d / f)
        (d / "trainer_state.json").write_text(json.dumps({"global_step": last, "epoch": max(h.get("epoch") or 0 for h in hist),
                                                          "note": "final adapter copied from adapter/ by p2gen_modal.py"}))
    copies = {"c" + c.name.split("-")[1]: {"step": int(c.name.split("-")[1]),
                                          "epoch": json.loads((c / "trainer_state.json").read_text()).get("epoch")}
              for c in _checkpoints(out)}
    (out / "copies.json").write_text(json.dumps(copies, indent=1)); note_copies()
    _say(dv, "TRAINING DONE: copies " + " ".join("%s (epoch %s)" % (k, round(v["epoch"], 2) if v["epoch"] is not None else "?")
                                                 for k, v in copies.items()))

    ads = ",".join(f"c{c.name.split('-')[1]}={c}" for c in _checkpoints(out))
    for name, d in [("nll", data)] + ([("nll_" + pathlib.Path(extra_nll).name, pathlib.Path("/vol") / extra_nll)] if extra_nll else []):
        _say(dv, f"START checkpoint loss on {d.name} (untrained model + {len(copies)} copies)")
        rc = run(["nll", "--student", "qwen36", "--model", model, "--data-dir", str(d), "--splits", "val,test", "--adapters", ads,
                  "--out", str(out / name), "--max-len", str(MAX_LEN)], f"{name}.log")
        tail = (logs / f"{name}.log").read_text(errors="replace").strip().splitlines()[-2:]
        _say(dv, f"checkpoint loss on {d.name}: {'; '.join(tail)}" if rc == 0 else f"CHECKPOINT LOSS FAILED (logs/{name}.log)")
    (out / "DONE").write_text(_now() + "\n")
    _say(dv, "P2GEN TRAIN ALL DONE")


def _wire(msgs):
    """scripts/p2q_load_copy.py's wire(): the training file's messages in the API's format."""
    out, pending = [], []
    for i, m in enumerate(msgs):
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for j, tc in enumerate(m["tool_calls"]):
                f = dict(tc["function"]); cid = tc.get("id") or f"call_{i}_{j}"
                if not isinstance(f.get("arguments"), str):
                    f["arguments"] = json.dumps(f.get("arguments") or {})
                tcs.append({"id": cid, "type": "function", "function": f}); pending.append(cid)
            m["tool_calls"] = tcs
        if m["role"] == "tool" and not m.get("tool_call_id"):
            m["tool_call_id"] = pending.pop(0) if pending else f"call_{i}"
        out.append(m)
    return out


def _post(url, body, timeout=900):
    import urllib.error, urllib.request
    req = urllib.request.Request("http://localhost:8000" + url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            t = r.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}: {e.read().decode()[:600]}")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {"text": t}


def _gate(name, data):
    """scripts/p2q_load_copy.py's check that a loaded copy really changes the model (vLLM has served LoRA copies of this
    model as the untrained model before, vLLM #49354): on 3 training examples the server scores the teacher's own reply
    under the untrained model and under the copy; the copy must give it a clearly higher chance. One change from p2ds:
    the reply scored is the example's third assistant turn or, when it has fewer (single-question sets), its last one.
    Returns (ok, one line)."""
    ex = []
    for line in open(data / "train.jsonl"):
        e = json.loads(line); msgs = e["messages"]
        idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if not idx:
            continue
        i = idx[min(2, len(idx) - 1)]
        tgt = (msgs[i].get("reasoning_content") or "") + (msgs[i].get("content") or "")
        if len(tgt) >= 40 and sum(len(str(m.get("content") or "")) for m in msgs[:i]) < 20000:
            ex.append((e, i))
        if len(ex) == 3:
            break
    gaps = []
    for e, i in ex:
        lp = {}
        for model in ("qwen36-base", name):
            body = {"model": model, "messages": _wire(e["messages"][:i + 1]), "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
                    "add_generation_prompt": False, "continue_final_message": True,
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
            if e.get("tools"):
                body["tools"] = e["tools"]
            r = _post("/v1/chat/completions", body)
            lp[model] = [next(iter(d.values()))["logprob"] for d in (r.get("prompt_logprobs") or []) if d]
        a, b = lp["qwen36-base"], lp[name]
        n = min(len(a), len(b)); k = max(1, n // 4)      # the last quarter of the prompt: mostly the teacher's reply
        gaps.append(-sum(a[-k:]) / k + sum(b[-k:]) / k)
    if not gaps:
        return False, "GATE: no usable example"
    good = sum(g > 0.02 for g in gaps); ok = good >= max(1, len(gaps) - 1)
    return ok, f"GATE {'OK' if ok else 'FAILED'}: {name} lowers the loss on {good} of {len(gaps)} examples (mean gap {sum(gaps) / len(gaps):.3f})"


def _metrics():
    """The server's own counters (classic_fleet3.sh's health line): cache queries and hits, cache in use, running, waiting."""
    import urllib.request
    try:
        t = urllib.request.urlopen(urllib.request.Request("http://localhost:8000/metrics", headers={
            "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}), timeout=10).read().decode()
    except Exception:   # noqa: BLE001
        return None
    g = lambda name: sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#"))
    return {"queries": g("vllm:prefix_cache_queries_total"), "hits": g("vllm:prefix_cache_hits_total"),
            "generated": g("vllm:generation_tokens_total"), "preemptions": g("vllm:num_preemptions_total"),
            "cache_pct": round(100 * g("vllm:kv_cache_usage_perc")), "running": g("vllm:num_requests_running"),
            "waiting": g("vllm:num_requests_waiting"), "t": time.time()}


@app.function(image=vllm_image, gpu=SERVE_GPU, cpu=8, memory=49152, volumes=VOLS, timeout=24 * 3600,
              secrets=[modal.Secret.from_name(f"vllm-secret{SUFFIX}")])
def serve(dv: str, copies: str = "all", hours: float = 10.0, name: str = ""):
    """Qwen3.6 27B, thinking on, with the saved copies of one training set loaded as LoRA adapters c<step>
    (p2q_serve.sh + p2q_copies.sh). copies: "all" (every copy as it appears, at most 4) or a comma list such as "c8";
    "none" serves the untrained model only. The address goes to serve/<dv>/<name>.json; the server ends when
    serve/<dv>/<name>.stop appears or after `hours`. If Modal preempts the GPU the function starts again with a NEW
    address and "restarts" goes up by one: runs that were in flight are cut, and the launcher stops and says so."""
    _dv_ok(dv)
    if not os.environ.get("VLLM_API_KEY"):
        sys.exit("VLLM_API_KEY is not in the environment: the Modal secret vllm-secret<suffix> must hold it")
    name = name or re.sub(r"[^a-z0-9]+", "-", copies.lower()).strip("-")
    sd = pathlib.Path("/vol/serve") / dv; sd.mkdir(parents=True, exist_ok=True)
    (pathlib.Path("/vol/logs") / dv).mkdir(parents=True, exist_ok=True)
    key = os.environ["VLLM_API_KEY"]
    clean = lambda t: t.replace(key, "[password]")   # the password never reaches a saved log
    state_f, stop_f = sd / f"{name}.json", sd / f"{name}.stop"
    out = pathlib.Path("/vol/out") / dv; data = pathlib.Path("/vol/data") / dv
    prev = json.loads(state_f.read_text()) if state_f.exists() else {}
    if stop_f.exists():
        _say(dv, f"serve {name}: a stop file is already there, not starting"); return
    _model_dir()
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    log = open("/tmp/vllm.log", "w")
    proc = subprocess.Popen(["vllm", "serve", MODEL, *VLLM_ARGS], stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while True:   # ready when the model list answers
        if proc.poll() is not None or time.time() - t0 > 2400:
            (pathlib.Path("/vol/logs") / dv / f"vllm_{name}_failed.log").write_text(clean(open("/tmp/vllm.log", errors="replace").read())[-200000:])
            _say(dv, f"SERVER FAILED TO START {name} (logs/vllm_{name}_failed.log)"); proc.kill(); sys.exit(1)
        try:
            _post("/v1/models", None, timeout=5); break
        except Exception:   # noqa: BLE001
            time.sleep(10)
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", open("/tmp/vllm.log", errors="replace").read())
    state = {"dv": dv, "name": name, "gpu": SERVE_GPU, "started": _now(), "restarts": prev.get("restarts", -1) + 1,
             "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None, "window_tokens": 65536,
             "served": ["qwen36-base"], "gate": {}, "status": "up"}
    want = None if copies == "all" else [] if copies == "none" else copies.split(",")
    done = set(); prev_m = None; last_beat = 0

    def save():
        tmp = state_f.with_suffix(".tmp"); tmp.write_text(json.dumps(state, indent=1)); tmp.replace(state_f); vol.commit()

    with modal.forward(8000) as tunnel:   # a direct TLS tunnel to this container: no web-endpoint request time limit
        state["url"] = tunnel.url + "/v1"; state["host"] = tunnel.url.split("://", 1)[1].split("/")[0].split(":")[0]
        save(); _say(dv, f"SERVER UP {name} (start {state['restarts'] + 1}, KV cache {state['kv_cache_tokens']} tokens, ready in {int(time.time() - t0)} s)")
        try:
            while time.time() - t0 < hours * 3600 and proc.poll() is None:
                try:
                    vol.reload()
                except Exception as e:   # noqa: BLE001
                    print("reload:", type(e).__name__, e, flush=True)
                if stop_f.exists():
                    break
                for c in _checkpoints(out):
                    n = "c" + c.name.split("-")[1]
                    if n in done or (want is not None and n not in want) or len(done) >= 4:
                        continue
                    if time.time() - (c / "trainer_state.json").stat().st_mtime < 90:
                        continue
                    done.add(n)
                    local = pathlib.Path("/tmp/loras") / n; local.mkdir(parents=True, exist_ok=True)   # a local copy: no
                    for f in ("adapter_model.safetensors", "adapter_config.json"):                    # file stays open on
                        (local / f).write_bytes((c / f).read_bytes())                                # the volume
                    try:
                        _post("/v1/load_lora_adapter", {"lora_name": n, "lora_path": str(local)})
                        ok, line = _gate(n, data)
                    except Exception as e:   # noqa: BLE001
                        ok, line = False, f"GATE ERROR: {type(e).__name__}: {str(e)[:300]}"
                    state["gate"][n] = line
                    if ok:
                        state["served"].append(n); _say(dv, f"COPY SERVED {n} on {name}: {line}")
                    else:   # unloaded, so no run can ever be made on a copy that did not pass
                        try:
                            _post("/v1/unload_lora_adapter", {"lora_name": n}, timeout=60)
                        except Exception:   # noqa: BLE001
                            pass
                        _say(dv, f"COPY GATE FAILED {n} on {name}: {line}")
                    save()
                if time.time() - last_beat >= 300:
                    m = _metrics(); last_beat = time.time()
                    if m and prev_m:
                        dq = m["queries"] - prev_m["queries"]; dt = max(1, m["t"] - prev_m["t"])
                        hit = round(100 * (m["hits"] - prev_m["hits"]) / dq) if dq > 0 else 100
                        state["health"] = {"at": _now(), "cache_hits_pct": hit, "cache_in_use_pct": m["cache_pct"],
                                           "running": m["running"], "waiting": m["waiting"],
                                           "tokens_per_s": round((m["generated"] - prev_m["generated"]) / dt),
                                           "preemptions": m["preemptions"] - prev_m["preemptions"],
                                           "healthy": hit >= 70 and m["waiting"] == 0}
                        save()
                        with open(pathlib.Path("/vol/logs") / dv / f"serve_{name}.txt", "a") as f:
                            f.write(json.dumps(state["health"]) + "\n")
                    prev_m = m or prev_m
                time.sleep(30)
        finally:
            state["status"] = "stopped"; state["stopped"] = _now(); save()
            proc.terminate()
            (pathlib.Path("/vol/logs") / dv / f"vllm_{name}.log").write_text(clean(open("/tmp/vllm.log", errors="replace").read())[-400000:])
            _say(dv, f"SERVER STOPPED {name} after {round((time.time() - t0) / 3600, 2)} h")
