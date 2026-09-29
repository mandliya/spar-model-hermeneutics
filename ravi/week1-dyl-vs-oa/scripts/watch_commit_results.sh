#!/usr/bin/env bash
# Commit + push oa_transfer_eval.json whenever it changes, so a lost box
# does not take the scores with it. Does not touch probe weights or npz caches.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
FILE="ravi/week1-dyl-vs-oa/results/oa_transfer_eval.json"
LAST=""
echo "watching $FILE"
while true; do
  if [[ -f "$FILE" ]]; then
    SUM="$(cksum "$FILE" | awk '{print $1" "$2}')"
    if [[ "$SUM" != "$LAST" ]]; then
      git add -- "$FILE"
      if git diff --cached --quiet -- "$FILE"; then
        LAST="$SUM"
      else
        DATASETS="$(python3 -c "import json; print(','.join(json.load(open('$FILE'))['datasets']))" 2>/dev/null || echo unknown)"
        git commit -m "$(cat <<EOF
ravi: checkpoint OA transfer scores ($DATASETS)

EOF
)"
        if git push origin HEAD; then
          echo "pushed $DATASETS"
        else
          echo "commit ok, push failed; will retry next change"
        fi
        LAST="$SUM"
      fi
    fi
  fi
  sleep 20
done
