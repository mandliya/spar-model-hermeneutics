#!/usr/bin/env bash
# Talk to one of the two rented boxes of the 25 Nov run, with every call recorded on both sides.
#   box.sh run  lens|nla '<command>'    # run it on the box through logged.sh
#   box.sh sync lens|nla                # copy the code, the keys file and logged.sh to the box
# On the box, logged.sh appends the command's text, time, full output and exit code to
# /workspace/logs/<experiment>_session.log (own_lens for the lens box, classic_nla for the NLA box, so shared/upload.py
# sends it with that experiment's logs). On this machine the same call (the ssh or rsync line and what came back) is
# appended to the orchestration log under dyl-probe-repro/scratch. Both reach HF with the run's logs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIANQ="$(cd "$HERE/../../.." && pwd)"
LOGGED="$TIANQ/dyl-probe-repro/logged.sh"
export DYL_SESSION_LOG="$TIANQ/dyl-probe-repro/scratch/internal-readout-gap/classic_distillation/logs/local_orchestration.log"
KEY=~/.ssh/id_ed25519_vast

action=${1:?usage: box.sh run|sync lens|nla ['<command>']}
case ${2:?which box: lens or nla} in
  lens) PORT=41466; HOST=54.68.157.60; EXPERIMENT=own_lens ;;      # Vast 62475580, 1 B200
  nla)  PORT=35096; HOST=213.181.108.220; EXPERIMENT=classic_nla ;;   # Vast 62481216, 8 H200, destroyed 25 Nov 15:57 UTC
        # when its run was over (the first NLA box, Vast 62475563 with 8 RTX PRO 6000, was destroyed at 07:17 UTC)
  *) echo "unknown box $2"; exit 1 ;;
esac
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ServerAliveInterval=30 -p $PORT"

case $action in
  run)
    on_box="DYL_SESSION_LOG=/workspace/logs/${EXPERIMENT}_session.log /workspace/logged.sh $(printf %q "${3:?the command}")"
    "$LOGGED" "$SSH root@$HOST $(printf %q "$on_box")" ;;
  sync)
    "$LOGGED" "$SSH root@$HOST 'mkdir -p /workspace/repo/tianq/experiments /workspace/logs' && \
rsync -a --delete --exclude __pycache__ --exclude audit_app --exclude '*.egg-info' -e '$SSH' $TIANQ/experiments/internal_readout_gap $TIANQ/experiments/common root@$HOST:/workspace/repo/tianq/experiments/ && \
rsync -a -e '$SSH' $LOGGED root@$HOST:/workspace/logged.sh && \
rsync -a --chmod=600 -e '$SSH' $TIANQ/dyl-probe-repro/.env root@$HOST:/workspace/.env && echo synced" ;;
  *) echo "unknown action $action"; exit 1 ;;
esac
