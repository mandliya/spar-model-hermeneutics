#!/usr/bin/env bash
# Destroy this box from inside, with the box-scoped key (never printed). The CLI's reply goes to the status file.
#   destroy_box.sh now "<reason>"              # destroy at once
#   destroy_box.sh at <ISO time, UTC> "<reason>"   # the dead-man switch: sleep until then, then destroy
#   destroy_box.sh grace <minutes> "<reason>"  # wait, then destroy unless /workspace/keep_box exists
#   destroy_box.sh check                       # only ask the API about this instance (is the key still accepted?)
# The box-scoped key has stopped working hours after renting before (12 Oct); then the box must be destroyed from
# TQ's machine. The reply in the status file says which happened.
set -uo pipefail
STATUS=/workspace/logs/own_lens_status.txt
env_of_init() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p"; }
ID=$(env_of_init CONTAINER_ID); KEY=$(env_of_init CONTAINER_API_KEY)
CLI=/opt/instance-tools/bin/vastai
say() { echo "$(date -u +%FT%TZ) destroy_box: $*" >> "$STATUS"; }
case $1 in
  check) $CLI show instance "$ID" --api-key "$KEY" 2>&1 | tail -3 | cut -c1-200; exit ;;
  now) REASON=$2 ;;
  at) REASON=$3; say "dead-man switch armed for $2 ($REASON)"; sleep $(( $(date -u -d "$2" +%s) - $(date -u +%s) )) ;;
  grace) REASON=$3; say "destroying in $2 minutes unless /workspace/keep_box exists ($REASON)"; sleep $(( $2 * 60 ))
         [ -e /workspace/keep_box ] && { say "kept: /workspace/keep_box exists"; exit 0; } ;;
  *) echo "usage: see the header"; exit 1 ;;
esac
say "destroying instance $ID: $REASON"
say "reply: $($CLI destroy instance "$ID" --api-key "$KEY" 2>&1 | tr '\n' ' ' | cut -c1-300)"
