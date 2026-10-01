#!/usr/bin/env bash
# games/alibi: the remaining 781 rows (everything outside the 500-row subset).
# One detached job. Each stage depends on the one before, so a failure stops
# the chain loudly (with the reason in the log); everything is resumable.
# Results go to HF as each stage finishes. There is NO automatic stop: the box
# is stopped only when the user says so.
cd /workspace/lies-experiment
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
up () { python -c "
from src.hf_sync import upload
import sys
for f in sys.argv[1:]:
    upload('results/' + f, 'results/alibi/' + f)" "$@"; }
stop_chain () { say "!!! $* - stopping the chain here (box left running)"; kill $HB 2>/dev/null; exit 1; }
( while true; do sleep 600; say "heartbeat"; done ) & HB=$!

say "=== 1/6 guilt labels, the other 781 rows ==="
python 01_label_alibi.py --rest --limit 5 || stop_chain "label smoke test crashed"
n=$(python -c "import json; print(sum(1 for l in open('results/alibi_guilt_labels.jsonl') if json.loads(l).get('guilt')))")
[ "$n" -ge 504 ] || stop_chain "label smoke test: expected >=504 labelled rows, got $n"
python 01_label_alibi.py --rest || stop_chain "labelling failed"
up alibi_guilt_labels.jsonl alibi_guilt_labels.csv alibi_rest.csv

say "=== 2/6 first run, new rows ==="
python 00_alibi.py --limit 5 || stop_chain "first-run smoke test crashed"
n=$(python -c "import json; print(sum(1 for l in open('results/alibi_conditions.jsonl') if json.loads(l).get('result')))")
[ "$n" -ge 502 ] || stop_chain "first-run smoke test: expected >=502 judged rows, got $n"
python 00_alibi.py || stop_chain "first run failed"
up alibi_conditions.jsonl alibi_conditions.csv
python 02_analyse_alibi.py --gate || stop_chain "arm-size gate failed"

say "=== 3/6 follow-ups: neutral, neutral_cot, named ==="
python 00_alibi_selfreport_benchmark.py || stop_chain "follow-ups failed"
up alibi_benchmark_conditions.jsonl alibi_benchmark_conditions.csv

say "=== 4/6 follow-up: named_cot ==="
python 05_named_cot.py --category alibi || stop_chain "named_cot failed"
python -c "from src.hf_sync import upload; upload('results/named_cot_alibi.jsonl', 'results/named_cot/named_cot_alibi.jsonl')"

say "=== 5/6 P(A) for the new answers ==="
python 06_answer_probability.py --categories alibi || say "!!! P(A) failed (independent of 6/6 - continuing)"

say "=== 6/6 activations: rebuild activations/alibi.tar.gz, add first-run files ==="
python 10_alibi_activations_update.py || say "!!! activation update failed (tensors still on /workspace)"

python 02_analyse_alibi.py > results/alibi_analysis_all_rows.txt 2>&1
kill $HB 2>/dev/null
say "=== ALL STAGES FINISHED - box is now IDLE and STILL BILLING; stop it by hand ==="
