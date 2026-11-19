#!/usr/bin/env bash
# setup_launch_box.sh: make an always-on Vast box the launcher for phase 2 fleets (19 Nov 2026). Fleets of Modal apps must
# not be launched from the laptop: `modal run --detach` needs the launcher alive until each app completes, and a sleeping
# Mac or a Wi-Fi drop kills the fleet (memories modal-fleets-die-when-mac-sleeps, launch-modal-from-vast-box).
# Run ON THE LAPTOP, after GG has rented the box (renting is GG's yes; this script rents nothing):
#   bash gigi/3-forensics/scripts/setup_launch_box.sh <ssh host or ip> <ssh port>
# A CPU box is enough for launching (the teacher runs through OpenRouter, the sandboxes on Modal); a cheap one costs cents an
# hour. Suggested search (GG runs it and picks):  vastai search offers 'cpu_cores>=8 cpu_ram>=16 reliability>0.98' -o dph
# What it does:
# 1. rsync gigi/3-forensics to /workspace/forensics/gigi/3-forensics: scripts, configs, the harness clone WITH our
#    uncommitted changes (incl. the usage ledger), harness_patch, phase2 docs. Never results/raw, .venv, or any .env.
# 2. On the box: uv, the harness's uv environment (uv sync), the hf CLI with hf_xet (binary files need Xet), PyYAML for the
#    helper scripts, ownership root + git safe.directory (Vast copies arrive with other owners).
# 3. Checks, printing names and yes/no only, never a value: ~/.modal.toml present, `modal profile current` works, the
#    Modal secrets openrouter-secret-gigi / openrouter-agent-gigi exist, /workspace/.env holds HF_TOKEN (and
#    OPENROUTER_API_KEY when GG wants the money checks to run on the box).
# GG copies the two credential files himself (I may not move Modal credentials; keys never go through chat):
#   scp -i ~/.ssh/vastai_lies_experiment -P <port> ~/.modal.toml root@<host>:/root/.modal.toml
#   scp -i ~/.ssh/vastai_lies_experiment -P <port> <a local file holding HF_TOKEN=... (and OPENROUTER_API_KEY=...)> root@<host>:/workspace/.env
set -euo pipefail
HOST=${1:?ssh host}; PORT=${2:?ssh port}
KEY=${KEY:-$HOME/.ssh/vastai_lies_experiment}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -p "$PORT" "root@$HOST")
DEST=/workspace/forensics/gigi/3-forensics

"${SSH[@]}" "mkdir -p $DEST /workspace/logs"
rsync -az -e "ssh -i $KEY -p $PORT -o BatchMode=yes" \
  --exclude '.venv/' --exclude '.pytest_cache/' --exclude '__pycache__/' --exclude '.env' --exclude 'results/raw/' \
  --exclude 'archive/' --exclude 'charts/' --exclude 'dashboard/' --exclude '*.png' \
  "$HERE/scripts" "$HERE/configs" "$HERE/agent-interp-envs" "$HERE/harness_patch" "$HERE/phase2" "root@$HOST:$DEST/"
echo "synced scripts, configs, agent-interp-envs (with our changes), harness_patch, phase2"

"${SSH[@]}" bash -s <<'REMOTE'
set -uo pipefail
DEST=/workspace/forensics/gigi/3-forensics
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
python3 -m pip install -q -U "huggingface_hub[cli]" hf_xet pyyaml >/dev/null 2>&1 || echo "pip install of hf / pyyaml FAILED"
chown -R root:root /workspace/forensics; git config --global --add safe.directory '*'
cd "$DEST/agent-interp-envs" && uv sync -q && echo "harness environment ready ($(uv run python -c 'import agent_interp_envs, sys; print(sys.version.split()[0])'))"
grep -q USAGE_KEY src/agent_interp_envs/providers/openrouter_provider.py && echo "usage ledger: present" || echo "usage ledger: MISSING"
[ -f /root/.modal.toml ] && echo "~/.modal.toml: present" || echo "~/.modal.toml: MISSING (GG copies it)"
if [ -f /root/.modal.toml ]; then
  uv run modal profile current >/dev/null 2>&1 && echo "modal profile: works" || echo "modal profile: NOT working"
  secrets=$(MODAL_PROFILE=spar uv run modal secret list --json 2>/dev/null)   # --json: the table truncates names
  for s in openrouter-secret-gigi openrouter-agent-gigi; do echo "$secrets" | grep -q "$s" && echo "Modal secret $s: present" || echo "Modal secret $s: MISSING"; done
fi
for k in HF_TOKEN OPENROUTER_API_KEY; do grep -q "^$k=" /workspace/.env 2>/dev/null && echo "/workspace/.env $k: present" || echo "/workspace/.env $k: absent"; done
command -v hf >/dev/null && echo "hf CLI: $(hf --version 2>&1 | head -n 1)" || echo "hf CLI: MISSING"
REMOTE
echo "Watch the box from the laptop with a polling heartbeat over this ssh port (memory box-watch-heartbeat)."
