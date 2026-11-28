#!/usr/bin/env bash
# Talk to a rented box, with every call recorded on both sides. Called through an experiment's own box.sh, which
# names the box in the environment: BOX_HOST, BOX_PORT, BOX_EXPERIMENT, and DYL_SESSION_LOG (this machine's log).
#   box.sh run '<command>'                     # run it on the box through logged.sh
#   box.sh sync                                # copy the code, the keys file and logged.sh to the box
#   box.sh put <local folder> <box folder>     # copy a folder's files into a folder on the box
#   box.sh get <box folder> <local folder>     # copy a folder's files from the box
# On the box, logged.sh appends the command's text, time, full output and exit code to
# /workspace/logs/<experiment>_session.log, so shared/upload.py sends it with that experiment's logs. On this machine
# the same call (the ssh or rsync line and what came back) is appended to DYL_SESSION_LOG. Both reach HF.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIANQ="$(cd "$HERE/../../.." && pwd)"
LOGGED="$TIANQ/dyl-probe-repro/logged.sh"
: "${BOX_HOST:?}" "${BOX_PORT:?}" "${BOX_EXPERIMENT:?}" "${DYL_SESSION_LOG:?}"
SSH="ssh -i $HOME/.ssh/id_ed25519_vast -o IdentitiesOnly=yes -o BatchMode=yes -o ServerAliveInterval=30 -p $BOX_PORT"

case ${1:?usage: box.sh run|sync|put|get ...} in
  run)
    on_box="DYL_SESSION_LOG=/workspace/logs/${BOX_EXPERIMENT}_session.log /workspace/logged.sh $(printf %q "${2:?the command}")"
    "$LOGGED" "$SSH root@$BOX_HOST $(printf %q "$on_box")" ;;
  sync)
    "$LOGGED" "$SSH root@$BOX_HOST 'mkdir -p /workspace/repo/tianq/experiments /workspace/logs' && \
rsync -a --delete --exclude __pycache__ --exclude audit_app --exclude '*.egg-info' --exclude handoff_from_john -e '$SSH' $TIANQ/experiments/internal_readout_gap $TIANQ/experiments/common root@$BOX_HOST:/workspace/repo/tianq/experiments/ && \
rsync -a -e '$SSH' $LOGGED root@$BOX_HOST:/workspace/logged.sh && \
rsync -a --chmod=600 -e '$SSH' $TIANQ/dyl-probe-repro/.env root@$BOX_HOST:/workspace/.env && echo synced" ;;
  put)
    "$LOGGED" "$SSH root@$BOX_HOST 'mkdir -p ${3:?the folder on the box}' && rsync -a -e '$SSH' ${2:?the local folder}/ root@$BOX_HOST:$3/ && echo copied" ;;
  get)
    "$LOGGED" "mkdir -p ${3:?the local folder} && rsync -a -e '$SSH' root@$BOX_HOST:${2:?the folder on the box}/ $3/ && echo copied" ;;
  *) echo "unknown action $1"; exit 1 ;;
esac
