#!/usr/bin/env bash
# Talk to one of the two rented boxes of the 25 Nov run, with every call recorded on both sides.
#   box.sh run  lens|nla '<command>'    # run it on the box through logged.sh
#   box.sh sync lens|nla                # copy the code, the keys file and logged.sh to the box
# The work is done by ../shared/box.sh (what is recorded and where: its header); this file only names the boxes.
# The box's own session log is /workspace/logs/<experiment>_session.log: own_lens for the lens box, classic_nla for
# the NLA box, so shared/upload.py sends it with that experiment's logs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DYL_SESSION_LOG="$(cd "$HERE/../../.." && pwd)/dyl-probe-repro/scratch/internal-readout-gap/classic_distillation/logs/local_orchestration.log"

action=${1:?usage: box.sh run|sync lens|nla ['<command>']}
case ${2:?which box: lens or nla} in
  lens) export BOX_PORT=41466 BOX_HOST=54.68.157.60 BOX_EXPERIMENT=own_lens ;;        # Vast 62475580, 1 B200
  nla)  export BOX_PORT=35096 BOX_HOST=213.181.108.220 BOX_EXPERIMENT=classic_nla ;;  # Vast 62481216, 8 H200, destroyed 25 Nov 15:57 UTC
        # when its run was over (the first NLA box, Vast 62475563 with 8 RTX PRO 6000, was destroyed at 07:17 UTC)
  *) echo "unknown box $2"; exit 1 ;;
esac
shift 2
exec "$HERE/../shared/box.sh" "$action" "$@"
