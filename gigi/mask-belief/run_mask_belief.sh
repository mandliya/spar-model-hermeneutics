#!/usr/bin/env bash
# MASK belief categories (known_facts, statistics, continuations,
# disinformation; 299 rows). One detached job; each stage needs the one before,
# so a failure stops the chain loudly (reason in the log). Everything is
# resumable. Results go to HF after each stage. NO automatic stop.
cd /workspace/lies-experiment
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
stop_chain () { say "!!! $* - stopping the chain here (box left running)"; kill $HB 2>/dev/null; exit 1; }
up_results () { python -c "
from src.hf_sync import upload
import glob, os
for f in sorted(glob.glob('results/mask_belief_*_$1.jsonl')):
    upload(f, 'results/mask_belief/' + os.path.basename(f))"; }
( while true; do sleep 600; say "heartbeat"; done ) & HB=$!

say "=== 0/4 smoke test: 3 known_facts rows through generate + judge ==="
python 12_mask_belief_generate.py --subtypes known_facts --limit 3 || stop_chain "generate smoke test crashed"
python 13_mask_belief_judge.py --subtypes known_facts --limit 3 || stop_chain "judge smoke test crashed"
n=$(python -c "
import json
r=[json.loads(l) for l in open('results/mask_belief_known_facts_judged.jsonl')]
print(sum(1 for x in r if 'error' not in x and x['statement_verdict'] is not None and None not in x['belief_verdicts']))")
[ "$n" -ge 2 ] || stop_chain "smoke test: only $n of 3 rows fully judged"
say "smoke test passed ($n of 3 rows fully judged)"

say "=== 1/4 belief + pressured answers, all subtypes ==="
python 12_mask_belief_generate.py --subtypes all || stop_chain "generation failed"
up_results answers

say "=== 2/4 judge ==="
python 13_mask_belief_judge.py --subtypes all || stop_chain "judging failed"
up_results judged

say "=== 3/4 the four follow-up questions ==="
python 14_mask_belief_followups.py --subtypes all || stop_chain "follow-ups failed"
up_results followups

say "=== 4/4 upload activations and verify ==="
python 15_mask_belief_upload.py || say "!!! upload step failed (everything still on /workspace)"

kill $HB 2>/dev/null
say "=== ALL STAGES FINISHED - box is now IDLE and STILL BILLING; stop it by hand ==="
