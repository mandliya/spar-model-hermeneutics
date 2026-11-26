#!/usr/bin/env bash
# One-time setup of a one-GPU box that fits a J-lens for one distilled checkpoint. Idempotent; safe to rerun.
#   bash setup_lens_box.sh gemma-classic-step92
# Expects /workspace/repo (tianq/experiments synced) and /workspace/.env (HF_TOKEN). Makes the readout and fit
# environments, then downloads base gemma and that checkpoint's adapter (the folder shared/models.py names).
# Log: /workspace/logs/setup.log; its last line is "setup done" or the failing line. Then: run.sh gate fit_final upload_final.
set -euo pipefail
TAG=$1
. "$(dirname "$0")/../shared/box_lib.sh"
exec >> /workspace/logs/setup.log 2>&1
trap 'echo "setup failed at line $LINENO"' ERR
echo "### $(date -u +%FT%TZ) lens box setup for $TAG"

readout_venv
fit_venv
"$VENV/bin/python" -c "import torch, transformers, jlens, peft; print('readout: torch', torch.__version__, 'cuda', torch.cuda.device_count(), 'transformers', transformers.__version__, 'peft', peft.__version__)"
"$FITV/bin/python" -c "import torch, transformers, accelerate, datasets; print('fit: torch', torch.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'datasets', datasets.__version__)"

fetch_distilled "$TAG"
echo "setup done"
