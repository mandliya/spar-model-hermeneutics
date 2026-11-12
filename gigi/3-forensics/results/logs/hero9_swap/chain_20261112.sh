#!/usr/bin/env bash
# 12 Nov 2026: top up the 44 missing continuations, then lean fetch + Opus 4.8 grading + score.
cd "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics"
NO_FOLDIN=1 bash "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero9_swap_paced.sh" "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/commands_topup_20261111.txt" 3078 >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/paced.out" 2>&1
echo "$(date '+%F %T') topup launches done" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
JUDGE_MODEL=claude-opus-4-8 CEIL=7544 GRADE_CAP=1200 bash "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/scripts/hero9_swap_lean_foldin.sh"
echo "$(date '+%F %T') chain done" >> "/Users/gg/Desktop/aisafety/spar-model-hermeneutics/gigi/3-forensics/results/logs/hero9_swap/chain.txt"
