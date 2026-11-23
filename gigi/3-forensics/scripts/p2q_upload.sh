#!/bin/bash
# Upload the re-run stages, score files and logs of the DeepSeek -> Qwen3.6 round to the private HF dataset.
set -u
set -a; . /workspace/.env; set +a
export PATH=$HOME/.local/bin:$PATH
H=/workspace/forensics/gigi/3-forensics
LOG=/workspace/logs/p2q_upload.log
mkdir -p /workspace/p2q_logs_up
cp /workspace/logs/p2q*.txt /workspace/logs/p2q*.out /workspace/logs/p2q*.log /workspace/p2q_logs_up/ 2>/dev/null
cp $H/results/distill_p2ds_eval.csv $H/results/distill_p2ds_eval_per_run.csv /workspace/p2q_logs_up/
if grep -rlE 'sk-or-v1-[0-9a-f]{20,}|hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|fw_[A-Za-z0-9]{20,}' /workspace/p2q_logs_up; then
  echo "KEY SHAPE FOUND IN LOGS, logs not uploaded" >> $LOG; SKIPLOGS=1; else SKIPLOGS=0; fi
R=model-hermeneutics/gigi-forensics-transcripts
for s in base c24 c48 c72 c96 gate; do
  st=distill_p2ds_$s
  hf upload $R "$H/results/raw/$st" "distill/p2_deepseekv4_v1/eval/raw/$st" --repo-type dataset --exclude "*.part" --exclude "**/fs/**" >> $LOG 2>&1 && echo "UPLOADED $st" >> $LOG || echo "UPLOAD FAILED $st" >> $LOG
done
if [ "$SKIPLOGS" = 0 ]; then
  hf upload $R /workspace/p2q_logs_up "distill/p2_deepseekv4_v1/eval/logs" --repo-type dataset >> $LOG 2>&1 && echo "UPLOADED logs" >> $LOG || echo "UPLOAD FAILED logs" >> $LOG
fi
echo "P2Q UPLOAD ALL DONE" >> $LOG
