"""Entry points, run through ../run.sh (which logs every invocation).

    extract.py         GPU: conversations -> stored hidden states
    evaluate.py        CPU: stored hidden states -> the dataset's check, results to the Hub
    inspect_rows.py    audit what stored rows contain
    compare_labels.py  AISI data under AISI's label vs the turn-level label, probe vs just asking
"""
