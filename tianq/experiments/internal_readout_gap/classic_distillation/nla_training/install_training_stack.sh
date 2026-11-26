#!/usr/bin/env bash
# The official NLA training stack on the NLA box, installed the way the library's docs/setup.md says. Idempotent.
#   bash install_training_stack.sh
# In order:
#   Miles at the library's pin (nla/miles_patches/UPSTREAM_PIN), built by Miles' own build_conda.sh, unmodified:
#     a micromamba env "miles" with CUDA 12.9, torch 2.9.1, SGLang at Miles' pinned commit with Miles' patch, flash-attn
#   the library's patches for Miles, then Miles installed from the pinned checkout
#   the library's patches for SGLang (its apply_sglang_patches.sh)
#   the library itself (the vendored copy)
# Every patch applied is one their repos ship; no line of their code is changed by us.
set -euo pipefail
STACK=/workspace/stack      # build_conda.sh's BASE_DIR: it puts sglang/ and Megatron-LM/ beside miles/
NLA=/workspace/repo/tianq/experiments/internal_readout_gap/vendor/natural_language_autoencoders
PIN=$(cut -d@ -f2 "$NLA/nla/miles_patches/UPSTREAM_PIN")
export MAMBA_ROOT_PREFIX=/root/micromamba
mkdir -p "$STACK"; cd "$STACK"

[ -d miles/.git ] || git clone -q https://github.com/radixark/miles.git
git -C miles checkout -q "$PIN"
echo "miles at $(git -C miles rev-parse HEAD)"

if [ ! -e "$STACK/build_conda.done" ]; then
  # a half-finished earlier build is cleared first: the script clones and creates, and stops if they exist
  rm -rf "$STACK/sglang" "$STACK/Megatron-LM"
  [ -x /root/.local/bin/micromamba ] && /root/.local/bin/micromamba env remove -n miles -y || true
  # With miles already in BASE_DIR the script looks for Miles' own patches under $BASE_DIR/docker: link that folder.
  ln -sfn "$STACK/miles/docker" "$STACK/docker"
  # The script sources ~/.bashrc to pick up micromamba. On a Vast box that file also tries to attach tmux, which
  # fails without a terminal; ~/.no_auto_tmux is Vast's own switch for turning that off.
  touch /root/.no_auto_tmux
  ( cd "$STACK/miles" && BASE_DIR=$STACK SHELL=/bin/bash bash build_conda.sh )
  touch "$STACK/build_conda.done"
fi

set +u   # the env's activation scripts read variables that may be unset
eval "$(/root/.local/bin/micromamba shell hook -s bash)"
micromamba activate miles
set -u

cd "$STACK/miles"
if git apply --reverse --check "$NLA"/nla/miles_patches/*.patch 2> /dev/null; then
  echo "the library's Miles patches are already applied"
else
  git apply "$NLA"/nla/miles_patches/*.patch
  echo "applied the library's Miles patches"
fi
pip install -q -e .
bash "$NLA/patches/apply_sglang_patches.sh" "$STACK/sglang"
pip install -q -e "$STACK/sglang/python[all]"
pip install -q -e "$NLA"

python -c "
import torch, miles, sglang, nla, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))
print('sglang', sglang.__version__, 'transformers', transformers.__version__)
x = torch.ones(4, device='cuda') * 2
print('a GPU op works:', x.sum().item())"
git -C "$STACK/sglang" rev-parse HEAD
pip freeze > /workspace/logs/classic_nla_training_stack_freeze.txt
echo "training stack ready"
