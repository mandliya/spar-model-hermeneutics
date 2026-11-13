#!/bin/bash
# distill_redo_v2.sh: redo every distillation re-run with vLLM's CUDA graphs off (13 Nov 2026, GG's go).
# v1 (results/raw/distill_v1_*) had ~15% of runs with gibberish replies, base gemma included: vLLM 0.11's small-batch
# CUDA graphs broke gemma-3 + LoRA (11/20 one-at-a-time replies garbled, 0/30 in a big batch). v1 is kept as it is.
# Steps: pause the v1 loop and stop its Modal apps, restart vLLM with --enforce-eager, gate on 0 garbled one-at-a-time
# replies (base and c96), then the whole 5 models x 6 tasks x 20 runs as distill_v2_* through distill_topup.sh.
#   setsid nohup bash /workspace/forensics/scripts/distill_redo_v2.sh > /workspace/redo_v2.out 2>&1 < /dev/null &
W=/workspace; F=$W/forensics; L=$F/results/logs/distill_v2; mkdir -p $L; say () { echo "$(date '+%F %T') $*" >> $L/redo.txt; }
if ! ps -eo args | grep "[v]llm serve" | grep -q enforce-eager; then   # a rerun after a gate bug skips the restart
say "start: pausing the v1 loop"
  pkill -f distill_topup.sh; pkill -f start_topup.sh; pkill -f "modal run --detach"
  cd $F/agent-interp-envs
  for a in $(MODAL_PROFILE=spar .venv/bin/modal app list --json 2>/dev/null | .venv/bin/python -c "import json,sys; [print(x['app_id']) for x in json.load(sys.stdin) if x['state'] != 'stopped' and 'distill_v1_' in x['description']]"); do
    MODAL_PROFILE=spar .venv/bin/modal app stop --yes "$a" >> $L/redo.txt 2>&1 && say "stopped Modal app $a"
  done
  say "restarting vLLM with --enforce-eager"
  pkill -f "vllm serve"; sleep 30; pkill -9 -f "vllm serve"; sleep 5
  grep -q enforce-eager $W/distill_serve_hero9.sh || sed -i 's/--no-enable-prefix-caching/--no-enable-prefix-caching --enforce-eager/' $W/distill_serve_hero9.sh
  mv $W/logs/serve.log $W/logs/serve.v1_cudagraph_garble.log
  setsid nohup bash $W/distill_serve_hero9.sh > $W/logs/serve.log 2>&1 < /dev/null &
fi
for i in $(seq 1 90); do curl -s -m 5 -H "Authorization: Bearer $(grep ^VLLM_API_KEY= $W/.env | cut -d= -f2- | tr -d '"')" localhost:8000/v1/models | grep -q c96 && break; sleep 10; done
grep -q enforce-eager <(ps -eo args | grep "[v]llm serve") || { say "GATE FAILED: vLLM did not come back with --enforce-eager (see $W/logs/serve.log). Box idle, still billing"; exit 1; }
say "vLLM up; garble gate"
python3 $F/scripts/distill_garble_test.py 20 4 gemma-base,c96 > $L/gate.txt 2>&1
cat $L/gate.txt >> $L/redo.txt
if grep -E "garbled [1-9]" $L/gate.txt || ! grep -q "one at a time c96 garbled 0" $L/gate.txt; then
  say "GATE FAILED: still garbled with CUDA graphs off (see gate.txt). Box idle, still billing"; exit 1
fi
say "gate passed: 0 garbled; launching distill_v2 (5 models x 6 tasks x 20 runs, 4 at a time per task)"
DISTILL_V=v2 MAXC=4 bash $F/scripts/distill_topup.sh
say "redo finished (see $L/topup.txt)"
