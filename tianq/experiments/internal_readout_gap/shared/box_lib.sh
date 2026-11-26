# Sourced by a box's setup script: the Python environments and the download helper every box shares, so each
# pinned version is written once. Expects /workspace/.env (HF_TOKEN) and /workspace/repo.
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1   # plain HTTP: the xet backend stalled with a few files left
set -a; . /workspace/.env; set +a
REPO=/workspace/repo
VENV=/workspace/venvs/readout     # transformers 5.x: every model, the J-lens (vendor/jacobian-lens), Qwen3.6's NLA
NV=/workspace/venvs/nla           # transformers 4.57.1, which the vendored official NLA client expects (Gemma NLAs)
FITV=/workspace/venvs/fit         # the published gemma-3-27b-it lens's own environment, for fitting a lens
VENDOR=$REPO/tianq/experiments/internal_readout_gap/vendor
mkdir -p /workspace/logs "$HF_HOME" /workspace/data

readout_venv() {
  [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
  # transformers and peft are pinned to the versions of the 13 Nov reads: unpinned, a box on 17 Nov got transformers
  # 5.23.0, under which Qwen3.6's forward pass gives NaN from the first layer on.
  uv pip install -q -p "$VENV/bin/python" torch==2.8.0 transformers==5.22.2 accelerate==1.10.1 peft==0.21.2 safetensors huggingface_hub datasets \
    pyyaml pandas pyarrow numpy scipy scikit-learn matplotlib openai \
    "anthropic>=0.121.0" "omegaconf>=2.3" "pydantic>=2.0" "python-dotenv>=1.2.1" "sentences>=0.1.1" "tenacity>=9.1.2"
  uv pip install -q -p "$VENV/bin/python" -e "$VENDOR/jacobian-lens"
}

qwen_kernels() {   # fast kernels for the Qwen3.6 / Qwen3-Next layer type; fla 0.4.0 is the newest that works with torch 2.8's Triton
  uv pip install -q -p "$VENV/bin/python" flash-linear-attention==0.4.0 fla-core==0.4.0
  uv pip install -q -p "$VENV/bin/python" --no-build-isolation causal-conv1d==1.7.0
}

nla_venv() {
  [ -x "$NV/bin/python" ] || uv venv -q -p 3.12 "$NV"
  uv pip install -q -p "$NV/bin/python" torch==2.8.0 transformers==4.57.1 accelerate==1.10.1 safetensors huggingface_hub pyyaml numpy httpx orjson
}

fit_venv() {   # versions from the published lens's config.yaml on HF ("environment": torch 2.11.0+cu128, transformers 5.11.0, ...)
  [ -x "$FITV/bin/python" ] || uv venv -q -p 3.12 "$FITV"
  uv pip install -q -p "$FITV/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
  uv pip install -q -p "$FITV/bin/python" transformers==5.11.0 accelerate==1.14.0 datasets==5.0.0 huggingface_hub numpy sentencepiece tqdm safetensors
}

# Downloads: snapshot_download with explicit patterns (the `hf download` CLI ignores --include once a second
# pattern is given). dl <repo> <revision or -> <repo type> [allow patterns...]; an "!pattern" is ignored instead.
# A dataset lands in /workspace/data/<its name>, a model in the HF cache.
dl() {
  "$VENV/bin/python" - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, kind, *patterns = sys.argv[1:]
allow = [p for p in patterns if not p.startswith("!")] or None
ignore = [p[1:] for p in patterns if p.startswith("!")] or None
local = "/workspace/data/" + repo.split("/")[-1] if kind == "dataset" else None
snapshot_download(repo, revision=None if revision == "-" else revision, repo_type=kind, allow_patterns=allow,
                  ignore_patterns=ignore, local_dir=local, max_workers=16)
print("downloaded", repo, revision, patterns, flush=True)
PY
}

fetch_distilled() {   # fetch_distilled <tag>: base gemma and that checkpoint's adapter, as shared/models.py pins them
  local base revision adapter
  read -r base revision adapter < <("$VENV/bin/python" -c "
import sys
sys.path.insert(0, '$REPO/tianq/experiments/internal_readout_gap')
from shared import models
m = models.MODELS['$1']
print(m.hf, m.revision, m.adapter)")
  dl model-hermeneutics/gigi-forensics-transcripts - dataset "$adapter/adapter_*"
  dl "$base" "$revision" model
}
