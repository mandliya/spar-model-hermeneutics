# Sourced by every script that runs the library's code on the NLA box: the environment install_training_stack.sh
# built, the keys, and where things are.
set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1
export MAMBA_ROOT_PREFIX=/root/micromamba
set +u   # the env's activation scripts read variables that may be unset
eval "$(/root/.local/bin/micromamba shell hook -s bash)"
micromamba activate miles
set -u
STACK=/workspace/stack
NLA=/workspace/repo/tianq/experiments/internal_readout_gap/vendor/natural_language_autoencoders

released() {   # released av|ar: the folder of the released Gemma-27B writer (av) or reconstructor (ar), as downloaded
  ls -d "$HF_HOME"/hub/models--kitft--nla-gemma3-27b-L41-"$1"/snapshots/*
}
