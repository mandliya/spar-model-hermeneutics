#!/usr/bin/env bash
# On the user's explicit instruction (1 Oct 2026, going to bed): "stop the
# instance yourself after all of this is done".
#
# 1. wait for run_alibi_rest.sh to exit (or treat it as stuck at 19:00 UTC)
# 2. make sure every result is on HF (re-upload anything missing; back up the
#    new tensors if the activation step didn't finish)
# 3. stop the box only if step 2 verified; otherwise leave it running
cd /workspace/lies-experiment
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
DEADLINE=$(date -u -d "today 19:00" +%s)

say "waiting for run_alibi_rest.sh to finish (deadline 19:00 UTC)"
while pgrep -f "bash run_alibi_rest.sh" >/dev/null; do
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then
    say "!!! still running at the deadline - treating it as stuck, stopping it"
    pkill -f "bash run_alibi_rest.sh"; pkill -f "python 0[0-9]_"; pkill -f "python 10_alibi"
    sleep 30
    break
  fi
  sleep 60
done
say "pipeline finished; checking every result is on HF"

python - <<'PY'
import os, pathlib, subprocess, sys
from huggingface_hub import HfApi
from src.hf_sync import upload, REPO
api = HfApi(token=os.environ["HF_TOKEN"])
FILES = [(f"results/{f}", f"results/alibi/{f}") for f in
         ["alibi_guilt_labels.jsonl", "alibi_guilt_labels.csv", "alibi_rest.csv",
          "alibi_conditions.jsonl", "alibi_conditions.csv",
          "alibi_benchmark_conditions.jsonl", "alibi_benchmark_conditions.csv",
          "alibi_analysis_all_rows.txt"]]
FILES += [("results/named_cot_alibi.jsonl", "results/named_cot/named_cot_alibi.jsonl"),
          ("results/pa_alibi.jsonl", "results/answer_probability/pa_alibi.jsonl")]

def remote_size(path):
    info = api.get_paths_info(REPO, [path], repo_type="dataset")
    return info[0].size if info else None

ok = True
for local, remote in FILES:
    lp = pathlib.Path(local)
    if not lp.exists():
        print(f"(no local {local})", flush=True); continue
    if remote_size(remote) != lp.stat().st_size:
        upload(lp, remote)
    good = remote_size(remote) == lp.stat().st_size
    print(f"{'OK ' if good else 'BAD'} {remote}", flush=True)
    ok &= good

log = pathlib.Path("run_alibi_rest.log").read_text()
if "ALIBI ACTIVATIONS UPDATED" in log and "VERIFY activations/alibi.tar.gz: OK" in log:
    print("OK  activations/alibi.tar.gz (updated by the pipeline)", flush=True)
else:
    print("activation update did not complete - backing up the new tensors", flush=True)
    tb = pathlib.Path("/workspace/alibi_new_rows_unmerged.tar.gz")
    subprocess.run(["tar", "-czf", str(tb), "-C", "/workspace/lies-experiment/results", "activations",
                    "-C", "/workspace/fp32_staging", "named_cot_alibi"], check=False)
    upload(tb, "activations/backup/alibi_new_rows_unmerged.tar.gz")
    good = remote_size("activations/backup/alibi_new_rows_unmerged.tar.gz") == tb.stat().st_size
    print(f"{'OK ' if good else 'BAD'} activations/backup/alibi_new_rows_unmerged.tar.gz", flush=True)
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
