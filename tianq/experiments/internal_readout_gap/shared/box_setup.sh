#!/usr/bin/env bash
# One-time setup of the rented box for PLAN.md (13 Nov 2026 run). Idempotent; safe to rerun.
#   (nohup setsid bash /workspace/repo/tianq/experiments/internal_readout_gap/shared/box_setup.sh > /dev/null 2>&1 &)
# Expects /workspace/repo (tianq/ and gigi/ synced) and /workspace/.env (HF_TOKEN). Makes two venvs:
#   /workspace/venvs/readout   transformers 5.x: every model, the J-lens (vendor/jacobian-lens), Qwen3.6's NLA
#   /workspace/venvs/nla       transformers 4.57.1, which the vendored official NLA client expects (Gemma NLAs)
# then Gigi's harness (the text-mode renderer our histories go through) and every model, lens,
# NLA, transcript and adapter the two stages read. Logs to /workspace/logs/setup.log; the last line is "setup done"
# or the failing line.
set -euo pipefail
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1   # plain HTTP: the xet backend stalled with a few files left
set -a; . /workspace/.env; set +a
REPO=/workspace/repo; VENV=/workspace/venvs/readout; NV=/workspace/venvs/nla
VENDOR=$REPO/tianq/experiments/internal_readout_gap/vendor
mkdir -p /workspace/logs "$HF_HOME" /workspace/data
exec >> /workspace/logs/setup.log 2>&1
trap 'echo "setup failed at line $LINENO"' ERR
echo "### $(date -u +%FT%TZ) setup"

[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
uv pip install -q -p "$VENV/bin/python" torch==2.8.0 accelerate==1.10.1 peft safetensors huggingface_hub datasets \
  pyyaml pandas pyarrow numpy scipy scikit-learn matplotlib openai \
  "anthropic>=0.121.0" "omegaconf>=2.3" "pydantic>=2.0" "python-dotenv>=1.2.1" "sentences>=0.1.1" "tenacity>=9.1.2"
uv pip install -q -p "$VENV/bin/python" -e "$VENDOR/jacobian-lens"
# Fast kernels for the Qwen3.6 / Qwen3-Next layer type; fla 0.4.0 is the newest that works with torch 2.8's Triton.
uv pip install -q -p "$VENV/bin/python" flash-linear-attention==0.4.0 fla-core==0.4.0
uv pip install -q -p "$VENV/bin/python" --no-build-isolation causal-conv1d==1.7.0
[ -x "$NV/bin/python" ] || uv venv -q -p 3.12 "$NV"
uv pip install -q -p "$NV/bin/python" torch==2.8.0 transformers==4.57.1 accelerate==1.10.1 safetensors huggingface_hub pyyaml numpy httpx orjson
"$VENV/bin/python" -c "import torch, transformers, jlens, peft; print('torch', torch.__version__, 'cuda', torch.cuda.device_count(), 'transformers', transformers.__version__, 'peft', peft.__version__)"

# Gigi's harness, restored the way gigi/3-forensics/harness_patch/README.md gives for a fresh machine (his setup.sh
# applies patches/ in an order that only works on a clone patched step by step). We only import its code, with our
# venv's packages.
F=$REPO/gigi/3-forensics; H=$F/agent-interp-envs
if ! grep -qs _text_tools_messages "$H/src/agent_interp_envs/providers/openrouter_provider.py"; then
  rm -rf "$H"; git clone -q https://github.com/gkroiz/agent-interp-envs.git "$H"
  git -C "$H" checkout -q "$(cat "$F/harness_patch/base_commit.txt")"
  git -C "$H" apply "$F/harness_patch/changes.patch" && cp -R "$F/harness_patch/new_files/." "$H/"
fi
echo "harness: agent-interp-envs@$(git -C "$H" rev-parse --short HEAD) with harness_patch"

# Downloads: snapshot_download with explicit patterns (the `hf download` CLI ignores --include once a second
# pattern is given). dl <repo> <revision or -> <repo type> [allow patterns...]; an "!pattern" is ignored instead.
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
# Gigi's transcripts. The swap continuations are ~26,000 small files: fetched one by one the Hub answers 429, so
# the repo comes as one shallow git clone (plain git blobs; large files stay LFS pointers; the Hub's git server
# refuses partial clones). The distilled adapters are LFS files, fetched through the Hub API (no optimizer states).
T=/workspace/data/gigi-forensics-transcripts
# The token reaches git (and the fetches it starts itself) through a credential helper in the environment, never a URL.
export GIT_LFS_SKIP_SMUDGE=1 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper
export GIT_CONFIG_VALUE_0='!f() { echo username=user; echo "password=$HF_TOKEN"; }; f'
transcripts() {
  [ -f "$T/manifest.csv" ] || { rm -rf "$T"; git clone -q --depth 1 \
    https://huggingface.co/datasets/model-hermeneutics/gigi-forensics-transcripts "$T"; }
  echo "$(date -u +%T) transcripts: $(find "$T/hero9_swap" -name messages.json | wc -l) continuations with messages.json"
  # the adapters, over their LFS pointers
  dl model-hermeneutics/gigi-forensics-transcripts - dataset "distill/hero9_v1/runs/v1/checkpoint-*/adapter_*"
}
transcripts &
LENS=neuronpedia/jacobian-lens
dl $LENS - model "gemma-3-27b-it/jlens/Salesforce-wikitext/*" "qwen3.6-27b/jlens/Salesforce-wikitext/*"    "gemma-3-12b-it/jlens/Salesforce-wikitext/*" "qwen3-32b/jlens/Salesforce-wikitext/*"
# Stage 1 first, so it can start while stage 2's models are still coming down.
dl google/gemma-3-27b-it 005ad3404e59d6023443cb575daa05336842228a model &
dl Qwen/Qwen3.6-27B 8f33b98937c6031bb49e713dee3e8e185aafd26f model &
wait
dl kitft/nla-gemma3-27b-L41-av - model &
dl kitft/nla-gemma3-27b-L41-ar - model &
dl ceselder/qwen3.6-27b-nla-av - model &
wait
echo "stage 1 downloads done"
dl google/gemma-3-12b-it 96b6f1eccf38110c56df3a15bffe176da04bfd80 model &
dl kitft/nla-gemma3-12b-L32-av - model &
dl kitft/nla-gemma3-12b-L32-ar - model &
dl Qwen/Qwen3-32B 9216db5781bf21249d130ec9da846c4624c16137 model &
wait
echo "setup done"
