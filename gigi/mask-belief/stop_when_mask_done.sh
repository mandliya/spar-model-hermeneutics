#!/usr/bin/env bash
# On the user's explicit instruction (2 Oct 2026): "put an automatic stop after
# the job is done".
# 1. wait for run_mask_belief.sh to exit (or treat it as stuck at 17:00 UTC)
# 2. make sure every result is on HF (re-upload anything missing; back up the
#    staged tensors if the upload stage didn't verify)
# 3. stop the box only if step 2 verified; otherwise leave it running
cd /workspace/lies-experiment
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
DEADLINE=$(date -u -d "today 17:00" +%s)

say "waiting for run_mask_belief.sh to finish (deadline 17:00 UTC)"
while pgrep -f "bash run_mask_belief.sh" >/dev/null; do
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then
    say "!!! still running at the deadline - treating it as stuck, stopping it"
    pkill -f "bash run_mask_belief.sh"; pkill -f "python 1[2-5]_mask_belief"
    sleep 30
    break
  fi
  sleep 60
done
say "pipeline finished; checking every result is on HF"

python - <<'PY'
import glob, os, pathlib, subprocess, sys
from huggingface_hub import HfApi
from src.hf_sync import upload, REPO
api = HfApi(token=os.environ["HF_TOKEN"])

def remote_size(path):
    info = api.get_paths_info(REPO, [path], repo_type="dataset")
    return info[0].size if info else None

ok = True
for f in sorted(glob.glob("results/mask_belief_*.jsonl")):
    remote = "results/mask_belief/" + os.path.basename(f)
    if remote_size(remote) != os.path.getsize(f):
        upload(f, remote)
    good = remote_size(remote) == os.path.getsize(f)
    print(("OK  " if good else "BAD ") + remote, flush=True)
    ok &= good

log = pathlib.Path("run_mask_belief.log").read_text(errors="replace")
if "MASK BELIEF UPLOAD VERIFIED" in log:
    print("OK  activations and first_run (verified by the upload stage)", flush=True)
else:
    st = pathlib.Path("/workspace/mask_belief_staging")
    if st.exists() and any(st.rglob("*.pt")):
        print("upload stage did not verify - backing up the staged tensors", flush=True)
        tb = pathlib.Path("/workspace/mask_belief_tensors_backup.tar.gz")
        subprocess.run(["tar", "-czf", str(tb), "-C", "/workspace", "mask_belief_staging"], check=False)
        upload(tb, "activations/backup/mask_belief_tensors_backup.tar.gz")
        good = remote_size("activations/backup/mask_belief_tensors_backup.tar.gz") == tb.stat().st_size
        print(("OK  " if good else "BAD ") + "activations/backup/mask_belief_tensors_backup.tar.gz", flush=True)
        ok &= good
sys.exit(0 if ok else 1)
PY

if [ $? -eq 0 ]; then
  say "everything verified on HF - stopping the box as the user asked"
  touch STOP_APPROVED_BY_USER
  bash stop_self.sh
else
  say "!!! some results could not be verified on HF - box LEFT RUNNING so nothing is lost"
fi
