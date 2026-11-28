#!/usr/bin/env bash
# One-time setup of the box of the gender-secret run. Idempotent; safe to rerun.
#   (nohup setsid bash /workspace/repo/tianq/experiments/internal_readout_gap/gender_secret_rerun/setup_box.sh > /dev/null 2>&1 &)
# Expects /workspace/repo (tianq/experiments synced) and /workspace/.env (HF_TOKEN). Makes the readout environment
# with the fast kernels for Qwen3.6's layer type, then downloads what shared/models.py pins for the two readers:
# Qwen3.6-27B, AISI's organism adapter, the published J-lens and the community NLA.
# Log: /workspace/logs/setup.log; its last line is "setup done" or the failing line.
set -euo pipefail
. "$(dirname "$0")/../shared/box_lib.sh"   # the environments, the download helper dl, HF_HOME, the keys
exec >> /workspace/logs/setup.log 2>&1
trap 'echo "setup failed at line $LINENO"' ERR
echo "### $(date -u +%FT%TZ) gender-secret box setup"

readout_venv
qwen_kernels
"$VENV/bin/python" -c "import torch, transformers, jlens, peft; print('torch', torch.__version__, 'cuda', torch.cuda.device_count(), torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'peft', peft.__version__)"

read -r BASE REVISION ADAPTER ADAPTER_REVISION NLA LENS < <("$VENV/bin/python" -c "
import sys
sys.path.insert(0, '$REPO/tianq/experiments/internal_readout_gap')
from shared import models
m = models.MODELS['qwen3.6-gender-secret-female']
print(m.hf, m.revision, *m.adapter_hub, m.nla[1], m.lens)")
echo "pinned: $BASE@$REVISION, adapter $ADAPTER@$ADAPTER_REVISION, NLA $NLA, lens $LENS"
dl neuronpedia/jacobian-lens - model "$LENS/*"
dl "$ADAPTER" "$ADAPTER_REVISION" model
dl "$BASE" "$REVISION" model &
dl "$NLA" - model &
wait
echo "setup done"
