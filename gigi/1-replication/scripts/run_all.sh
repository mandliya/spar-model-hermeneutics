#!/usr/bin/env bash
# Everything left for this session, in one unattended script. Nothing here
# waits on a person. Order:
#   0. keep the box up for this job (expiring hold + watcher), backstop armed
#   1. upload the recovered capture-the-secret float32 tarball (no GPU)
#   2. named_cot smoke test on 5 rows - if it fails, stop the box and exit
#   3. named_cot on all seven categories, each backed up to HF as it finishes
#   4. float32 recapture for alibi, backed up to HF
#   5. stop the box
# No `set -e`: the final stop always runs. Stopping preserves /workspace.
cd /workspace/lies-experiment
source /venv/main/bin/activate
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
REPO=model-hermeneutics/gigi-lie-detection-activations

# --- 0. hold the box for this job only, with a hard ceiling ------------------
date -u -d '+12 hours' +%s > HOLD_STOP_UNTIL
pgrep -f '^bash hold_watch\.sh$' >/dev/null || setsid bash hold_watch.sh > hold_watch.log 2>&1 < /dev/null &
say "hold until $(date -u -d @$(cat HOLD_STOP_UNTIL) +%H:%M)Z; hold_watch stops the box when it lapses"
say "disk: $(df -h /workspace | tail -1 | awk '{print $4}') free"

upload () {   # upload <tarball-or-dir> <hf-name> <results-file> <hf-folder>
  local SRC=$1 NAME=$2 RES=$3 DEST=$4 TAR
  if [ -d "$SRC" ]; then
    [ "$(ls "$SRC" | wc -l)" -ge 1 ] || { say "!!! $NAME: no tensors"; return 1; }
    TAR=/workspace/${NAME}.tar.gz; tar czf "$TAR" -C "$SRC" . || { say "!!! $NAME: tar failed"; return 1; }
  else TAR=$SRC; fi
  say "$NAME: uploading $(du -h "$TAR" | cut -f1)"
  if python - "$TAR" "$NAME" "$RES" "$DEST" <<'PY'
import os, sys
from dotenv import load_dotenv
load_dotenv("/workspace/lies-experiment/.env")
from huggingface_hub import HfApi
tar, name, res, folder = sys.argv[1:5]
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "model-hermeneutics/gigi-lie-detection-activations"
dest = f"tarballs/fp32/{name}.tar.gz"
api.upload_file(path_or_fileobj=tar, path_in_repo=dest, repo_id=repo, repo_type="dataset")
if res and os.path.exists(res):
    api.upload_file(path_or_fileobj=res, path_in_repo=f"{folder}/{os.path.basename(res)}",
                    repo_id=repo, repo_type="dataset")
    print(f"VERIFIED {folder}/{os.path.basename(res)}")
size = next(s.size for s in api.dataset_info(repo, files_metadata=True).siblings if s.rfilename == dest)
assert size == os.path.getsize(tar), f"size mismatch {size} vs {os.path.getsize(tar)}"
print(f"VERIFIED {dest} {size/1e9:.2f} GB")
PY
  then
    [ -d "$SRC" ] && rm -rf "$SRC" "$TAR"
    say "$NAME: verified on HF"
  else say "!!! $NAME: upload NOT verified - local files kept"; return 1; fi
}

# --- setup: packages + model (fresh box) -------------------------------------
say "=== setup: packages ==="
pip install -q "transformers==5.17.0" accelerate pandas statsmodels python-dotenv openai tqdm pyarrow fsspec \
  && say "--- packages OK ---" || say "!!! pip failed"
python -c 'import transformers,torch;print("transformers",transformers.__version__,"torch",torch.__version__)'
say "=== setup: gemma-3-27b-it ==="
TOKEN=$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2-)
GOT=0
for attempt in 1 2 3; do
  if HF_TOKEN="$TOKEN" python -c "from huggingface_hub import snapshot_download; print('DOWNLOADED', snapshot_download('google/gemma-3-27b-it'))"; then GOT=1; break; fi
  say "download attempt $attempt failed, retrying"; sleep 20
done
if [ "$GOT" -ne 1 ]; then say "!!! model download failed 3 times - stopping"; FORCE_STOP=1 bash stop_self.sh; exit 1; fi
say "--- model cached ---"

# --- 1. recovered capture-the-secret float32 --------------------------------
if [ -f /workspace/fp32_capture_secret.tar.gz ]; then
  say "=== 1. capture-the-secret float32 (recovered tarball) ==="
  tar -tzf /workspace/fp32_capture_secret.tar.gz > /workspace/capture_tar_list.txt 2>/dev/null
  python - <<'PY'
import json, collections
names = [l.strip().lstrip('./') for l in open('/workspace/capture_tar_list.txt') if l.strip().endswith('.pt')]
rows = collections.OrderedDict()
for f in names:
    base, kind = f.rsplit('__', 1)
    r = rows.setdefault(base, {'name': base, 'predecision_file': None, 'answer_token_file': None})
    r['predecision_file' if kind == 'predecision.pt' else 'answer_token_file'] = f
with open('results/fp32_map_capture_secret.jsonl', 'w') as fh:
    for r in rows.values():
        fh.write(json.dumps(r) + '\n')
print(f"map rebuilt: {len(rows)} rows")
PY
  upload /workspace/fp32_capture_secret.tar.gz capture_secret_fp32 results/fp32_map_capture_secret.jsonl maps/fp32
fi

# --- 2. smoke test: named_cot on 5 rows --------------------------------------
say "=== 2. named_cot smoke test ==="
python 05_named_cot.py --category mask_factual --limit 5
OK=$(python -c "import json;r=[json.loads(l) for l in open('results/named_cot_mask_factual.jsonl')];print(sum(1 for x in r if 'error' not in x and x.get('admitted') is not None))" 2>/dev/null || echo 0)
if [ "${OK:-0}" -lt 4 ]; then
  say "!!! SMOKE TEST FAILED ($OK/5 parsed) - stopping the box, not running the rest"
  FORCE_STOP=1 bash stop_self.sh; exit 1
fi
say "--- smoke test OK: $OK/5 parsed ---"

# --- 3. named_cot, all seven categories --------------------------------------
for CAT in mask_factual alibi capture_secret ascii sycophancy_b sycophancy_a unanswerable; do
  say "=== 3. named_cot: $CAT ==="
  python 05_named_cot.py --category $CAT || say "!!! named_cot $CAT failed (resumable)"
  upload /workspace/fp32_staging/named_cot_$CAT named_cot_${CAT}_fp32 results/named_cot_$CAT.jsonl results/named_cot
done

# --- 4. float32 recapture: alibi, plus capture-the-secret if no tarball -----
FP32="alibi"
[ -f /workspace/fp32_capture_secret.tar.gz ] || FP32="capture_secret alibi"
for CAT in $FP32; do
  say "=== 4. float32 recapture: $CAT ==="
  python 04_recapture_fp32.py --category $CAT || say "!!! fp32 $CAT failed (resumable)"
  upload /workspace/fp32_staging/$CAT ${CAT}_fp32 results/fp32_map_$CAT.jsonl maps/fp32
done

say "=== ALL DONE - stopping instance ==="
FORCE_STOP=1 bash stop_self.sh
