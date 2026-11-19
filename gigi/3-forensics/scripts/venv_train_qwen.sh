#!/usr/bin/env bash
# venv_train_qwen.sh: training environment for Qwen3.6 27B as the distillation student (phase 2, 19 Nov 2026), ON THE
# BOX. A separate venv: /workspace/venv-train (gemma, transformers 4.57.1) stays as it is so the classic round
# reproduces. Qwen3.6 (Qwen3_5ForConditionalGeneration, Gated DeltaNet) needs transformers 5.x, and its fast path needs
# flash-linear-attention (Triton GDN kernels) and causal-conv1d (CUDA build), else ~10x slower.
# Pins, as of 19 Nov 2026 on PyPI:
#   verified on CPU (tests/test_qwen36_train_cpu.py: tiny Qwen3_5 model, LoRA incl. DeltaNet, 5 steps, loss down):
#     transformers 5.23.0, peft 0.21.3, trl 1.18.1, accelerate 1.16.0, datasets 5.2.0 (with torch 2.13.0 CPU)
#   NOT verified (need the GPU box): torch 2.13.0 CUDA 12.9 build (the version our vLLM 0.32 venv used on the H200),
#     flash-linear-attention 0.6.0 (+ fla-core 0.6.0; requires transformers >=5.6.2,<6), causal-conv1d 1.7.0 (builds
#     against the installed torch; needs nvcc, ~10-20 min). If causal-conv1d fails to build, the run still works through
#     fla's kernels for the DeltaNet core and a PyTorch conv1d (slower); distill_train.py reports which path it found.
#   bash /workspace/forensics/gigi/3-forensics/scripts/venv_train_qwen.sh
set -uo pipefail
V=/workspace/venv-train-qwen; S=/workspace/logs/stages.txt; L=/workspace/logs/venv_train_qwen.log
say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
say "START venv train-qwen (torch 2.13.0 cu129, transformers 5.23.0, peft 0.21.3, trl 1.18.1, fla 0.6.0, causal-conv1d 1.7.0)"
uv venv -q "$V" --python 3.12 && VIRTUAL_ENV="$V" uv pip install --torch-backend cu129 torch==2.13.0 \
  transformers==5.23.0 peft==0.21.3 trl==1.18.1 accelerate==1.16.0 datasets==5.2.0 pillow==11.3.0 \
  flash-linear-attention==0.6.0 fla-core==0.6.0 "huggingface_hub[hf_transfer]" jinja2 sentencepiece pyyaml ninja packaging \
  > "$L" 2>&1
rc=$?; say "END venv train-qwen core rc=$rc"
# causal-conv1d compiles against the torch above: no build isolation, so it sees it
VIRTUAL_ENV="$V" uv pip install --no-build-isolation causal-conv1d==1.7.0 >> "$L" 2>&1 \
  && say "causal-conv1d 1.7.0 built" || say "causal-conv1d BUILD FAILED (venv_train_qwen.log): DeltaNet conv falls back to PyTorch"
"$V/bin/python" - >> "$S" 2>&1 <<'PY'
import importlib, torch, transformers, peft, trl
ok = {}
for m in ("fla", "causal_conv1d"):
    try:
        importlib.import_module(m); ok[m] = "ok"
    except Exception as e:   # noqa: BLE001
        ok[m] = f"{type(e).__name__}"
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401  the class distill_train.py loads
print("train-qwen venv: torch", torch.__version__, "cuda ok:", torch.cuda.is_available(), "transformers",
      transformers.__version__, "peft", peft.__version__, "trl", trl.__version__, "kernels", ok)
PY
VIRTUAL_ENV="$V" uv pip freeze > /workspace/logs/freeze_train_qwen.txt 2>/dev/null
