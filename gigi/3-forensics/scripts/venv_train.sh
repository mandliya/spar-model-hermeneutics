#!/usr/bin/env bash
# venv_train.sh: the training environment for the classic round (round 1 pins; distill_train.py header). 17 Nov 2026.
set -uo pipefail
echo "$(date -u +%FT%TZ) START venv train (torch 2.8.0, transformers 4.57.1, peft 0.17.1, trl 0.24.0)" >> /workspace/logs/stages.txt
uv venv -q /workspace/venv-train --python 3.12 && VIRTUAL_ENV=/workspace/venv-train uv pip install torch==2.8.0 transformers==4.57.1 \
  peft==0.17.1 trl==0.24.0 accelerate==1.10.1 datasets==4.2.0 pillow==11.3.0 "huggingface_hub[hf_transfer]" jinja2 sentencepiece pyyaml \
  > /workspace/logs/venv_train.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) END venv train rc=$rc" >> /workspace/logs/stages.txt
/workspace/venv-train/bin/python -c "import torch,transformers,peft,trl;print(\"train venv: torch\",torch.__version__,\"cuda ok:\",torch.cuda.is_available(),\"transformers\",transformers.__version__,\"peft\",peft.__version__,\"trl\",trl.__version__)" >> /workspace/logs/stages.txt 2>&1
VIRTUAL_ENV=/workspace/venv-train uv pip freeze > /workspace/logs/freeze_train.txt 2>/dev/null
