#!/usr/bin/env bash
# Local watcher (Git Bash on the user's PC). Every 10 minutes: read each box's status file over SSH. Once a box
# has reported "VERIFIED complete on the Hub", destroy exactly that instance with the local Vast CLI if it is
# still listed. Never acts on any other instance id. Exits when both are gone.
VAST=~/.venvs/dyl-transfer/Scripts/vastai.exe
LOG=/c/Users/tqmen/AppData/Local/Temp/claude/--wsl-localhost-Ubuntu-home-user-projects-spar-model-hermeneutics/8f57f645-ac89-4088-8ef3-6fe1ca508eb8/scratchpad/watcher.log
declare -A PORT=([58397412]=35177 [58355405]=30350) HOST=([58397412]=213.181.108.222 [58355405]=82.98.239.123)
declare -A VERIFIED=() GONE=()
log () { echo "$(date -u +%H:%M:%SZ) $*" >> "$LOG"; }
# Prints 1 if listed, 0 if not, "?" if the CLI gave no answer (a timeout must never count as gone).
listed () { local out; out=$(timeout 120 "$VAST" show instances --raw 2>/dev/null); [ -z "$out" ] && { echo "?"; return; }
            echo "$out" | grep -c "\"id\": $1\b"; }
log "watcher started"
while [ ${#GONE[@]} -lt 2 ]; do
  for id in 58397412 58355405; do
    [ -n "${GONE[$id]:-}" ] && continue
    if [ "$(listed $id)" = 0 ]; then GONE[$id]=1; log "$id is gone (listed 0)"; continue; fi
    tail=$(MSYS_NO_PATHCONV=1 /c/Windows/System32/wsl.exe bash -c "timeout 40 ssh -i ~/.ssh/id_ed25519_vast -o ConnectTimeout=20 -p ${PORT[$id]} root@${HOST[$id]} 'tail -3 /workspace/status.log' 2>/dev/null")
    log "$id: $(echo "$tail" | tail -1)"
    echo "$tail" | grep -q "VERIFIED complete on the Hub" && VERIFIED[$id]=1
    if [ -n "${VERIFIED[$id]:-}" ] && echo "$tail" | grep -q "driver finished"; then
      log "$id verified and finished: destroying with the local CLI"
      timeout 180 "$VAST" destroy instance "$id" -y >> "$LOG" 2>&1
      sleep 30
      [ "$(listed $id)" = 0 ] && { GONE[$id]=1; log "$id destroyed (no longer listed)"; } || log "$id still listed after destroy"
    fi
  done
  [ ${#GONE[@]} -lt 2 ] && sleep 600
done
log "watcher done: both boxes gone"
