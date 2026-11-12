#!/usr/bin/env bash
# 12 Nov 2026: finish hero9 swap grading after the key limit was raised (Opus 4.8), then score and rebuild the dashboard.
cd "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics"
echo "$(date '+%F %T') regrade start" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
FORENSICS_ENV_FILE="/Users/gg/Desktop/aisafety/spar-model-hermeneutics/.env" HERO2_SWAP=1 SWAP_HERO_NAME=hero9_swap JUDGE_MODEL=claude-opus-4-8 GRADE_CAP_USD=800 TOTAL_CEILING_USD=7544 GRADE_LOG_DIR="/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/grading" python3 scripts/grade_modal.py >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/grade.txt" 2>&1
echo "$(date '+%F %T') regrade grading: $(tail -1 /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/grading/grade.txt)" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
python3 scripts/hero9_swap_score.py >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/score.txt" 2>&1 && echo "$(date '+%F %T') scored: $(tail -1 /Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/score.txt)" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
python3 scripts/hero9_swap_dashboard.py >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt" 2>&1
echo "$(date '+%F %T') regrade done" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
