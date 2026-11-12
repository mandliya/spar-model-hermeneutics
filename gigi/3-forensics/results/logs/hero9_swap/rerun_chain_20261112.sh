#!/usr/bin/env bash
# 12 Nov 2026: rerun the 15 continuations lost to provider errors or a crash, then lean fetch + Opus 4.8 grading + score + dashboard.
cd "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics"
NO_FOLDIN=1 bash "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero9_swap_paced.sh" "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/commands_rerun_20261112.txt" 100 >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/paced.out" 2>&1
echo "$(date '+%F %T') rerun launches done" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
JUDGE_MODEL=claude-opus-4-8 CEIL=7544 GRADE_CAP=100 bash "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero9_swap_lean_foldin.sh"
python3 "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero9_swap_dashboard.py" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt" 2>&1
echo "$(date '+%F %T') rerun chain done" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
