#!/usr/bin/env bash
# Cloud side of the hero9 swap handover (11 Nov 2026). Run in a Claude Code cloud session (claude.ai/code) whose
# environment has these secrets set (Settings of the environment, never in chat or git):
#   OPENROUTER_API_KEY, HF_TOKEN, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET   (the Modal pair = the [spar] profile of ~/.modal.toml)
# From the repo root:   nohup bash gigi/3-forensics/scripts/hero9_swap_cloud.sh > /tmp/hero9_cloud.out 2>&1 &
# Steps: 1 check secrets; 2 write the local .env and ~/.modal.toml (both outside git); 3 install the harness at the
# pinned upstream commit + our overlay; 4 push a CLOUD_READY marker and wait for the laptop's handoff commit
# (scripts/hero9_swap_laptop_handover.sh stops the laptop launcher and pushes handoff/hero9_swap/); 5 unpack the
# checkpoints, rebuild stub launch logs for what the laptop launched, start the spend watchdog, and run the paced
# launcher on what is left, which starts the fold-in (fetch, redact, grade, score, chart) when everything is launched.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; H="$ROOT/gigi/3-forensics"; HO="$H/handoff/hero9_swap"
L="$H/results/logs/hero9_swap"; mkdir -p "$L"
say() { echo "$(date '+%F %T') [cloud] $*" | tee -a "$L/cloud.txt"; }
cd "$ROOT" || exit 1

# 1. secrets present (values never printed)
miss=""; for v in OPENROUTER_API_KEY HF_TOKEN MODAL_TOKEN_ID MODAL_TOKEN_SECRET; do [ -n "${!v:-}" ] || miss="$miss $v"; done
[ -z "$miss" ] || { say "missing secrets:$miss (set them in the cloud environment's settings); nothing done"; exit 2; }

# 2. local credential files, outside git
git check-ignore -q .env || { say ".env is not gitignored here: refusing to write it"; exit 3; }
printf 'OPENROUTER_API_KEY=%s\nHF_TOKEN=%s\n' "$OPENROUTER_API_KEY" "$HF_TOKEN" > .env; chmod 600 .env
printf '[spar]\ntoken_id = "%s"\ntoken_secret = "%s"\nactive = true\n' "$MODAL_TOKEN_ID" "$MODAL_TOKEN_SECRET" > ~/.modal.toml; chmod 600 ~/.modal.toml
export MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics-
command -v uv >/dev/null || pip install -q uv || { say "cannot install uv"; exit 4; }

# 3. harness = upstream at the pinned commit + our overlay (needs the handoff for BASE_COMMIT; the laptop pushes an
#    early copy of the overlay with the scripts, the final one with the handoff)
wait_for () { until git pull -q --rebase 2>/dev/null; [ -f "$1" ]; do say "waiting for $1 in git"; sleep 120; done; }
wait_for "$HO/BASE_COMMIT"
if [ ! -d "$H/agent-interp-envs/.git" ]; then
  git clone -q https://github.com/gkroiz/agent-interp-envs.git "$H/agent-interp-envs" || { say "harness clone failed"; exit 5; }
fi
(cd "$H/agent-interp-envs" && git checkout -q "$(cat "$HO/BASE_COMMIT")" && tar xzf "$HO/harness_overlay.tar.gz" && env -u VIRTUAL_ENV uv sync -q) \
  || { say "harness setup failed"; exit 5; }
(cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal volume list >/dev/null 2>&1) || { say "Modal login failed with the given token"; exit 6; }
say "harness ready at $(cat "$HO/BASE_COMMIT" | cut -c1-7) + overlay; Modal reachable"

# 4. handshake: tell the laptop we are ready, then wait for its final handoff
if [ ! -f "$HO/HANDOFF_DONE" ]; then
  date '+%F %T' > "$H/handoff/CLOUD_READY"
  git add "$H/handoff/CLOUD_READY" && git commit -q -m "gigi: cloud session ready to take over the hero9 swap" && git push -q || { say "could not push CLOUD_READY"; exit 7; }
  say "CLOUD_READY pushed; waiting for the laptop to stop its launcher and push the handoff"
  wait_for "$HO/HANDOFF_DONE"
fi
say "handoff received: $(cat "$HO/HANDOFF_DONE")"

# 5. unpack and run
(cd "$H" && tar xzf "$HO/checkpoints.tar.gz") || { say "checkpoint unpack failed"; exit 8; }
python3 - "$HO/log_index.tsv" "$L" <<'PY'
import csv, pathlib, sys
L = pathlib.Path(sys.argv[2]); n = 0
for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    p = L / r["log"]
    if p.exists(): continue
    lines = ["Checkpoint: (launched from the laptop; stub rebuilt from handoff/hero9_swap/log_index.tsv)"]
    if r["volume"]: lines.append(f"Volume: {r['volume']}")
    lines += r["fleet_runs"].split(); lines.append("App completed (stub)")
    p.write_text("\n".join(lines) + "\n"); n += 1
print(f"{n} stub launch logs written")
PY
sed "s#<repo>#$ROOT#g" "$HO/commands_remaining.txt" > "$L/commands_cloud.txt"
say "$(wc -l < "$L/commands_cloud.txt" | tr -d ' ') resumes left to launch"
CAP=$(cat "$L/../hero9/SWAP_APPROVED" 2>/dev/null | head -1); CAP=$(( ${CAP:-1026} * 3 ))
FORENSICS_ENV_FILE="$ROOT/.env" CAP_USD="$CAP" APP_MATCH="/results/raw/hero9_swap/" nohup bash "$H/scripts/spend_watchdog.sh" hero9_swap > "$L/watchdog_cloud.out" 2>&1 &
say "watchdog started (cap \$$CAP, counts the whole key)"
cd "$H" && bash scripts/hero9_swap_paced.sh "$L/commands_cloud.txt" "$CAP"
say "paced launcher returned; the fold-in runs detached (results/logs/hero9_swap/foldin.txt)"
