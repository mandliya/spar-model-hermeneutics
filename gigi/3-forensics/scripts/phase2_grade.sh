#!/bin/bash
# phase2_grade.sh (23 Nov 2026): Opus grading of DeepSeek V4 Pro's runs in the 11 judged tasks (GG's yes on 23 Nov,
# estimate ~$460 for 488 runs). Runs ON THE LAUNCH BOX, detached:
#   setsid nohup bash scripts/phase2_grade.sh > /workspace/logs/p2_grading/launch.txt 2>&1 &
# - Judge claude-opus-4-8 from the first pass: Opus 5 refuses transcripts built around another model's reasoning
#   (memory swap-grading-opus48) and still bills them; every judged Qwen3.6 / gemma run so far was graded by 4.8 too.
# - Only the kept runs: the (stage, task) batches in set_aside.tsv were re-run whole and are never graded.
# - Hard cap $1,400 (3x the estimate, a runaway guard only: memory spend-caps-generous). Graders skip graded runs,
#   so a stopped pass (cap, workspace budget, timeout) resumes by running this script again.
# - Then the grading.json files go to the private HF dataset next to the runs (phase2/deepseek-v4-pro/raw/<stage>).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/conda/bin:$PATH"
H=/workspace/forensics/gigi/3-forensics; L=/workspace/logs/p2_grading; mkdir -p "$L"
S="$L/stages.txt"; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
set -a; . /workspace/.env; set +a   # OPENROUTER_API_KEY, HF_TOKEN; never printed
STAGES=$(printf "p2_deepseekv4_f_%s," 1 2 3 4 5 6 7 8; printf "p2_deepseekv4_s_%s," 1 2 3 4 5 6 7; echo p2_deepseekv4_s_8)
USAGE=$(python3 "$H/scripts/phase2_money.py" --save-start "$L/money_start.json" 2>/dev/null | grep -oE "key_usage=[0-9.]+" | cut -d= -f2)
say "START grading: 11 judged tasks, judge claude-opus-4-8, cap \$1400, key usage at start ${USAGE:-unknown}"
cd "$H" && HERO_SOLO=1 HERO_ALL_JUDGED=1 HERO_MODEL=deepseek-v4-pro HERO_STAGES="$STAGES" \
  SET_ASIDE_FILE=/workspace/logs/p2_deepseekv4_set_aside.tsv GRADE_LOG_DIR="$L" FORENSICS_ENV_FILE=/workspace/.env \
  JUDGE_MODEL=claude-opus-4-8 GRADE_CAP_USD=1400 TOTAL_CEILING_USD=1400 CEILING_BASELINE_USD="${USAGE:-6559.74}" \
  python3 scripts/grade_modal.py >> "$L/launch.txt" 2>&1
say "grader exited $? (per task: $L/grade.txt)"
N=$(find "$H/results/raw" -path "*p2_deepseekv4_[fs]_*" -name grading.json | wc -l)
say "grading.json files under the stages: $N (488 expected; set-aside batches have none)"
for st in ${STAGES//,/ }; do
  hf upload model-hermeneutics/gigi-forensics-transcripts "$H/results/raw/$st" "phase2/deepseek-v4-pro/raw/$st" \
    --repo-type dataset --include "**/grading.json" >> "$L/upload.txt" 2>&1 || say "UPLOAD FAILED for $st (upload.txt)"
done
hf upload model-hermeneutics/gigi-forensics-transcripts "$L" "phase2/deepseek-v4-pro/logs/grading" --repo-type dataset \
  >> "$L/upload.txt" 2>&1 || say "UPLOAD FAILED for the grading logs"
say "P2 GRADING DONE"
