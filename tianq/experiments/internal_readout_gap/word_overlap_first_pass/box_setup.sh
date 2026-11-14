#!/usr/bin/env bash
# ARCHIVED (8 Nov 2026 first pass). Kept only so that experiment can be reproduced; do not import or extend.
# New box setup is described in ../PLAN.md.
# One-time setup on a rented box for the 8 Nov first pass (PLAN.md). Idempotent; safe to rerun.
#   bash tianq/experiments/internal_readout_gap/word_overlap_first_pass/box_setup.sh
# Expects /workspace/repo (tianq/ and gigi/3-forensics synced) and /workspace/.env (HF_TOKEN, OPENROUTER_API_KEY).
# Makes the project venv /workspace/venvs/internal-readout-gap (jlens from ../vendor/), the NLA venv
# /workspace/venvs/nla (transformers 4.57.1, which the vendored NLA client expects), Gigi's environments (his
# setup.sh) and raw runs, and every model and lens the first pass reads.
# Logs to /workspace/logs/setup_readouts.log; the last line is "setup done" or the failing step.
set -euo pipefail
export HF_HOME=/workspace/hf
set -a; . /workspace/.env; set +a
REPO=/workspace/repo; VENV=/workspace/venvs/internal-readout-gap
VENDOR=$REPO/tianq/experiments/internal_readout_gap/vendor
mkdir -p /workspace/logs "$HF_HOME"
exec >> /workspace/logs/setup_readouts.log 2>&1
trap 'echo "setup failed at line $LINENO"' ERR
echo "### $(date -u +%FT%TZ) setup readouts"

# The authors' code: verbatim copies at pinned commits in our repo (vendor/VENDORED.md), never edited.
# Project venv: torch, the deps agent-interp-envs declares (Gigi's renderer imports), and jlens editable from the
# vendored copy (it brings transformers>=5.5, which covers Qwen3-Next and gemma-3).
[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
uv pip install -q -p "$VENV/bin/python" torch==2.8.0 accelerate==1.10.1 safetensors huggingface_hub datasets \
  pyyaml pandas pyarrow numpy scipy matplotlib openai httpx orjson \
  "anthropic>=0.121.0" "omegaconf>=2.3" "pydantic>=2.0" "python-dotenv>=1.2.1" "sentences>=0.1.1" "tenacity>=9.1.2"
uv pip install -q -p "$VENV/bin/python" -e "$VENDOR/jacobian-lens"
# Qwen3-Next's fast kernels (transformers otherwise falls back to a correct but much slower PyTorch path);
# fla 0.4.0 is the newest release that works with the Triton 3.4 that torch 2.8 ships.
uv pip install -q -p "$VENV/bin/python" flash-linear-attention==0.4.0 fla-core==0.4.0
uv pip install -q -p "$VENV/bin/python" --no-build-isolation causal-conv1d==1.7.0
NV=/workspace/venvs/nla
[ -x "$NV/bin/python" ] || uv venv -q -p 3.12 "$NV"
uv pip install -q -p "$NV/bin/python" torch==2.8.0 transformers==4.57.1 accelerate==1.10.1 safetensors huggingface_hub pyyaml numpy httpx orjson
"$VENV/bin/python" -c "import torch, transformers, jlens; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

# Gigi's environments (rendering imports) and raw runs (spot histories), as published on HF.
bash "$REPO/gigi/3-forensics/setup.sh"
RAW=$REPO/gigi/3-forensics/results/raw; mkdir -p "$RAW" /workspace/dl
B=https://huggingface.co/datasets/model-hermeneutics/gigi-lie-detection-activations/resolve/main/forensics/raw
for s in swap_all full_laptop modal; do
  [ -d "$RAW/$s" ] && continue
  [ -s /workspace/dl/$s.tar.gz ] || curl -sfL -o /workspace/dl/$s.tar.gz "$B/$s.tar.gz"
  tar xzf /workspace/dl/$s.tar.gz -C "$RAW"
done

# Models and lenses (revisions pinned in tianq/dyl-probe-repro/dyl_repro/core/model.py).
dl() { "$VENV/bin/hf" download "$@" > /dev/null; echo "downloaded $*"; }
dl google/gemma-3-27b-it --revision 005ad3404e59d6023443cb575daa05336842228a
dl Qwen/Qwen3-Next-80B-A3B-Instruct --revision 9c7f2fbe84465e40164a94cc16cd30b6999b0cc7
dl neuronpedia/jacobian-lens gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt
dl kitft/nla-gemma3-27b-L41-av
dl kitft/nla-gemma3-27b-L41-ar
dl Salesforce/wikitext --repo-type dataset --include "wikitext-103-raw-v1/train-*"
echo "setup done"
