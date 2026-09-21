#!/usr/bin/env bash
# Find and rent a GPU box suitable for Gemma-3-27B activation extraction.
#
#   ./setup/gpu_rent.sh                  # search and show candidates only
#   ./setup/gpu_rent.sh --create <id>    # actually rent that offer
#   ./setup/gpu_rent.sh --status         # what is running right now
#
# Searching and renting are separate commands on purpose: the first costs
# nothing, the second starts a meter. Nothing here rents anything unless
# --create is passed with an explicit offer id.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$HOME/.vast_active_instance"
SSH_KEY="$HOME/.ssh/id_ed25519_vast"
LABEL="tianq-dyl-transfer"

VENV_BIN="$HOME/.venvs/dyl-transfer/Scripts"
[[ -d "$VENV_BIN" ]] || VENV_BIN="$HOME/.venvs/dyl-transfer/bin"
VASTAI="$VENV_BIN/vastai"
[[ -x "$VASTAI" ]] || VASTAI="$VENV_BIN/vastai.exe"

# Gemma-3-27B bf16 is ~54GB of weights, so 80GB is the floor once hidden states
# for every layer are materialised.
#
# The gpu_name filter is not incidental. The cheapest cards on the marketplace
# right now are RTX PRO 6000 (Blackwell, sm_120), and the pinned cu124 torch
# image has no sm_120 kernels — that failure only surfaces after a 54GB
# download. Restricting to Hopper and Ampere keeps us on the architecture
# Ravi's setup is known to work on.
QUERY='gpu_ram>=80 num_gpus=1 gpu_name in [H100_SXM,H100_NVL,H100_PCIE,H200,H200_NVL,A100_SXM4,A100_PCIE] cuda_vers>=12.1 reliability>0.98 verified=true rentable=true disk_space>=120 inet_down>=400'

IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"

case "${1:-}" in
  --status)
    echo "=== instances ==="
    "$VASTAI" show instances
    if [[ -f "$STATE" ]]; then
      echo
      echo "recorded active instance: $(cat "$STATE")"
    fi
    exit 0
    ;;

  --create)
    OFFER_ID="${2:-}"
    [[ -n "$OFFER_ID" ]] || { echo "usage: $0 --create <offer_id>" >&2; exit 1; }

    if [[ -f "$STATE" ]]; then
      echo "An instance is already recorded: $(cat "$STATE")" >&2
      echo "Destroy it first with ./setup/gpu_destroy.sh, or clear $STATE." >&2
      exit 1
    fi

    echo "Renting offer $OFFER_ID ..."
    OUT="$("$VASTAI" create instance "$OFFER_ID" \
      --image "$IMAGE" \
      --disk 160 \
      --ssh --direct \
      --label "$LABEL" \
      --onstart-cmd 'nvidia-smi && echo ready' \
      --raw)"
    echo "$OUT"

    # Record the id immediately, before anything else can fail. An instance
    # that exists but was never written down is the expensive case.
    NEW_ID="$(echo "$OUT" | python -c "
import json, sys
try:
    print(json.load(sys.stdin).get('new_contract', ''))
except Exception:
    print('')
")"
    if [[ -z "$NEW_ID" ]]; then
      echo
      echo "WARNING: could not parse the instance id from that response." >&2
      echo "Check https://cloud.vast.ai/instances/ and destroy manually if needed." >&2
      exit 1
    fi

    echo "$NEW_ID" > "$STATE"
    echo
    echo "=============================================="
    echo " INSTANCE $NEW_ID IS NOW BILLING"
    echo " stop it with:  ./setup/gpu_destroy.sh"
    echo " or in the web console: https://cloud.vast.ai/instances/"
    echo " id also recorded at: $STATE"
    echo "=============================================="
    echo
    echo "Waiting for it to come up..."
    for _ in $(seq 1 60); do
      STATUS="$("$VASTAI" show instance "$NEW_ID" --raw 2>/dev/null | python -c "
import json, sys
try:
    print(json.load(sys.stdin).get('actual_status', ''))
except Exception:
    print('')
" || true)"
      [[ "$STATUS" == "running" ]] && break
      sleep 10
    done
    echo "status: ${STATUS:-unknown}"
    echo
    echo "SSH in with:"
    "$VASTAI" ssh-url "$NEW_ID" 2>/dev/null || true
    echo "  (key: $SSH_KEY)"
    exit 0
    ;;

  ""|--search)
    echo "=== candidate offers (nothing is rented by this command) ==="
    echo "filter: 80GB+, single GPU, Hopper/Ampere only, verified, 120GB+ disk"
    echo
    "$VASTAI" search offers "$QUERY" -o 'dph_total+' | head -15
    echo
    echo "Then rent the one you want:"
    echo "  $0 --create <offer_id>"
    exit 0
    ;;

  *)
    echo "usage: $0 [--search | --create <offer_id> | --status]" >&2
    exit 1
    ;;
esac
