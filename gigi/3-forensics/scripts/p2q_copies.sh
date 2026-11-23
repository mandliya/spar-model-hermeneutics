#!/usr/bin/env bash
# p2q_copies.sh (23 Nov 2026), ON THE GPU BOX: as each saved copy appears (checkpoint-<step>, every 0.5 epoch), load it
# into the Qwen3.6 server as c<step> and check it changes the model (p2q_load_copy.py). One line per copy in
# stages.txt: "COPY SERVED c<step>" (the launch box then runs it) or "COPY GATE FAILED c<step>".
#   setsid nohup bash scripts/p2q_copies.sh > /workspace/logs/p2q_copies.out 2>&1 < /dev/null &
H=/workspace/forensics/gigi/3-forensics; OUT=/workspace/out/p2_deepseekv4_v1; S=/workspace/logs/stages.txt
say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
while true; do
  for c in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    n="c${c##*-}"; [ -f "$c/.served" ] || [ -f "$c/.gate_failed" ] && continue
    [ -f "$c/adapter_model.safetensors" ] && [ -f "$c/trainer_state.json" ] && [ $(( $(date +%s) - $(stat -c %Y "$c/trainer_state.json") )) -gt 90 ] || continue
    if python3 "$H/scripts/p2q_load_copy.py" "$n" "$c" > "/workspace/logs/p2q_gate_$n.txt" 2>&1; then
      touch "$c/.served"; say "COPY SERVED $n: $(tail -n 1 /workspace/logs/p2q_gate_$n.txt)"
    else
      touch "$c/.gate_failed"   # unload it, so the launch box (which watches the server's model list) never runs it
      ( set -a; . /workspace/.env; set +a; curl -s -m 60 -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
          localhost:8000/v1/unload_lora_adapter -d "{\"lora_name\":\"$n\"}" > /dev/null )
      say "COPY GATE FAILED $n (/workspace/logs/p2q_gate_$n.txt): $(tail -n 1 /workspace/logs/p2q_gate_$n.txt | cut -c1-200)"
    fi
  done
  grep -q "P2Q TRAIN ALL DONE\|TRAINING FAILED three times" "$S" && [ -z "$(for c in "$OUT"/checkpoint-*; do [ -f "$c/.served" ] || [ -f "$c/.gate_failed" ] || echo x; done)" ] && { say "p2q_copies: every copy handled"; exit 0; }
  sleep 120
done
