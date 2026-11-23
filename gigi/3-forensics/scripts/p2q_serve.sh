#!/usr/bin/env bash
# p2q_serve.sh (23 Nov 2026): Qwen3.6 27B on GPU 1 of the phase 2 box, thinking ON, for the "before" runs and, as LoRA
# adapters loaded at run time (POST /v1/load_lora_adapter), every saved copy. GPU 0 trains at the same time.
# Flags as the classic round's Qwen server (distill/PLAN_classic.md section 1) except: thinking on by default, LoRA
# enabled, 65,536-token window (the training cut is 32,768; the window must not be smaller).
#   setsid nohup bash scripts/p2q_serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &
set -a; . /workspace/.env; set +a   # VLLM_API_KEY, HF_TOKEN: never printed (do not print this process's arguments)
export HF_HOME=/workspace/hf CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# vLLM compiles a kernel at start-up: it needs ninja and nvcc on PATH (23 Nov: "No such file or directory: ninja")
export PATH="/workspace/venv-vllm/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda
exec /workspace/venv-vllm/bin/vllm serve Qwen/Qwen3.6-27B --revision 8f33b98937c6031bb49e713dee3e8e185aafd26f \
  --served-model-name qwen36-base --language-model-only --dtype bfloat16 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true}' \
  --enable-lora --max-lora-rank 16 --max-loras 4 --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 8000
