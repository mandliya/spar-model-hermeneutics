#!/bin/bash
# distill_autostop.sh: stop (not destroy) this Vast box when the distill redo is over or broken (GG, 13 Nov 2026:
# "I don't want any boxes sitting idle if there are any mistakes"). Stopped, it bills only its disk (~$0.19/h) and
# keeps every file; GG decides about destroying it. Checks every 5 min:
#   - the gate failed, or the loop wrote DONE (results scored and uploaded to HF)
#   - the redo script is gone without either (crashed)
#   - vLLM served no request for 90 min while the redo still runs (hung)
# Uses the box's own Vast key from the container environment; never prints it.
#   setsid nohup bash /workspace/forensics/scripts/distill_autostop.sh > /workspace/autostop.out 2>&1 < /dev/null &
L=/workspace/forensics/results/logs/distill_v2; LOG=/workspace/logs/serve.log
say () { echo "$(date '+%F %T') AUTOSTOP: $*" >> $L/redo.txt; }
stop_box () {
  say "$1; stopping the box (disk kept, ~\$0.19/h)"
  K=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^CONTAINER_API_KEY=//p"); C=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^CONTAINER_ID=//p")
  vastai stop instance "$C" --api-key "$K" >> /workspace/autostop.out 2>&1; exit 0
}
sleep 600   # let the redo get going
last=-1; quiet=0
while true; do
  grep -q "GATE FAILED" $L/redo.txt 2>/dev/null && stop_box "the gate failed"
  grep -q "^.*DONE" $L/topup.txt 2>/dev/null && stop_box "the redo is done"
  pgrep -f distill_redo_v2.sh > /dev/null || stop_box "the redo script is no longer running"
  busy=$(grep -cE "Running: [1-9]" $LOG 2>/dev/null)
  if [ "$busy" = "$last" ]; then quiet=$((quiet + 5)); else quiet=0; last=$busy; fi
  [ $quiet -ge 90 ] && stop_box "vLLM served nothing for 90 min"
  sleep 300
done
