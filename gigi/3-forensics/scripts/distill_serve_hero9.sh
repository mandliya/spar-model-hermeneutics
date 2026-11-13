#!/usr/bin/env bash
# distill_serve_hero9.sh: serve base gemma and every hero9 distillation checkpoint from ONE vLLM server on the box
# (distill/PLAN_hero9.md step 3; 12 Nov 2026). Base gemma-3-27b-it in bf16 with the checkpoints as LoRA adapters:
# model "gemma-base" = adapter off (the control), "c24", "c48", "c72" = checkpoint-N, "final" = the trained adapter.
# Started detached by scripts/distill_eval_supervisor.sh after TRAINING DONE:
#   setsid nohup bash /workspace/distill_serve_hero9.sh > /workspace/logs/serve.log 2>&1 < /dev/null &
set -uo pipefail
# Prefix caching OFF: with it, gemma-3 (sliding-window layers) + LoRA in vLLM 0.11 returns garbage on every cache hit
# (12 Nov gate: step 1 fine, steps 2+ gibberish; the same request sent twice: sensible, then ' outlier?').
W=/workspace; LOG=$W/logs; OUT=$W/out/hero9_v1
set -a; . "$W/.env"; set +a                      # VLLM_API_KEY, HF_TOKEN: in a file on the box, never in chat or git
echo "$(date -u +%FT%TZ) START serve" >> "$LOG/stages.txt"
pip install -q vllm==0.11.0 > "$LOG/vllm_install.log" 2>&1 || { echo "$(date -u +%FT%TZ) VLLM INSTALL FAILED" >> "$LOG/stages.txt"; exit 1; }
mods=()
for c in "$OUT"/checkpoint-*; do
  n=${c##*-}; [ -f "$c/adapter_config.json" ] && mods+=("c$n=$c")
done
[ -f "$OUT/adapter/adapter_config.json" ] && mods+=("final=$OUT/adapter")
echo "$(date -u +%FT%TZ) serving gemma-base + ${mods[*]}" >> "$LOG/stages.txt"
exec vllm serve google/gemma-3-27b-it --served-model-name gemma-base --dtype bfloat16 \
  --enable-lora --max-lora-rank 16 --max-loras ${#mods[@]} --lora-modules "${mods[@]}" \
  --max-model-len 65536 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0}' \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 8000
