#!/usr/bin/env bash
# p2q_setup.sh (23 Nov 2026): set up the phase 2 GPU box for DeepSeek V4 Pro -> Qwen3.6 27B (Vast, 2x H200), detached:
#   setsid nohup bash /workspace/forensics/gigi/3-forensics/scripts/p2q_setup.sh > /workspace/logs/setup.out 2>&1 < /dev/null &
# Tools, then three things in parallel: the vLLM venv (serving, GPU 1), the training venv (venv_train_qwen.sh, GPU 0) and
# Qwen/Qwen3.6-27B at the revision of every Qwen3.6 run so far. Never stops the box.
set -uo pipefail
W=/workspace; LOG=$W/logs; mkdir -p "$LOG"; S=$LOG/stages.txt; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
export DEBIAN_FRONTEND=noninteractive PATH="$HOME/.local/bin:$PATH" HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1
say "START setup"
apt-get update -qq > "$LOG/apt.log" 2>&1 && apt-get install -y -qq git rsync curl python3 python3-venv tmux htop >> "$LOG/apt.log" 2>&1 \
  || say "APT FAILED (apt.log)"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > "$LOG/uv.log" 2>&1
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >> "$S"; df -h / | tail -1 >> "$S"; free -g | sed -n 2p >> "$S"
set -a; . "$W/.env"; set +a   # HF_TOKEN, VLLM_API_KEY: never printed
( uv venv -q "$W/venv-vllm" --python 3.12 && VIRTUAL_ENV="$W/venv-vllm" uv pip install vllm==0.32.0 "huggingface_hub[hf_transfer]" \
    > "$LOG/venv_vllm.log" 2>&1 \
  && say "venv-vllm ok: $("$W/venv-vllm/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)' 2>&1 | tail -1)" \
  || say "VENV-VLLM FAILED (venv_vllm.log)" ) &
( bash "$W/forensics/gigi/3-forensics/scripts/venv_train_qwen.sh" ) &
( uvx --from "huggingface_hub[hf_transfer]" hf download Qwen/Qwen3.6-27B --revision 8f33b98937c6031bb49e713dee3e8e185aafd26f \
    > "$LOG/download_qwen.log" 2>&1 && say "Qwen3.6-27B downloaded ($(du -sh "$HF_HOME/hub/models--Qwen--Qwen3.6-27B" | cut -f1))" \
  || say "QWEN DOWNLOAD FAILED (download_qwen.log)" ) &
wait
say "END setup"
