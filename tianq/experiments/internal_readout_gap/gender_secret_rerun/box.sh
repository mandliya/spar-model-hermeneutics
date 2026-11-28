#!/usr/bin/env bash
# Talk to the box of the gender-secret run (Vast 62651459: 4 RTX PRO 6000, rented 26 Nov), every call recorded.
#   box.sh run '<command>'    box.sh sync    box.sh put <local folder> <box folder>    box.sh get <box folder> <local folder>
#   box.sh rows               # the row files rows.py wrote on this machine, to where the box's readers look for them
#   box.sh bundles            # the small files gap.py reads (each read's two bundle files, the word lists), straight from
#                             # the box into scratch/.../gender_secret_rerun/bundles, without waiting for the upload to HF
# The work is done by ../shared/box.sh; this file only names the box.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(cd "$HERE/../../.." && pwd)/dyl-probe-repro/scratch/internal-readout-gap/gender_secret_rerun"
export BOX_HOST=106.185.159.136 BOX_PORT=22468 BOX_EXPERIMENT=gender_secret_rerun
export DYL_SESSION_LOG="$SCRATCH/logs/local_orchestration.log"

if [ "${1:-}" = rows ]; then
  exec "$HERE/../shared/box.sh" put "$SCRATCH/rows" /workspace/out/internal-readout-gap/gender_secret_rerun/rows
fi
if [ "${1:-}" = bundles ]; then
  SSH="ssh -i $HOME/.ssh/id_ed25519_vast -o IdentitiesOnly=yes -o BatchMode=yes -o ServerAliveInterval=30 -p $BOX_PORT"
  exec "$HERE/../../../dyl-probe-repro/logged.sh" "mkdir -p $SCRATCH/bundles && rsync -a -m --info=stats1 -e '$SSH' --include='*/' --include='words.json.gz' --include='probe_layers.npz' --include='distributions/*' --exclude='*' root@$BOX_HOST:/workspace/out/internal-readout-gap/gender_secret_rerun/ $SCRATCH/bundles/ && echo copied"
fi
exec "$HERE/../shared/box.sh" "$@"
