"""Entry points, run through ../run.sh (which logs every invocation).

    extract.py           GPU: conversations -> stored hidden states
    evaluate.py          CPU: stored hidden states -> the dataset's check, results to the Hub
    inspect_rows.py      audit what stored rows contain
    compare_labels.py    AISI data under AISI's label vs the turn-level label, probe vs just asking
    compare_wordings.py  CPU: just asking vs the probe, each with its best wording, across Gigi's datasets
    train_probe.py       CPU: train a probe the way AISI did on any dataset, test it on any others
    compare_generalization.py  CPU: probes trained on each Gigi category, tested on the others, three wordings
    compare_outcome_flags.py   CPU: what the released probe flags on each kind of answer to "did you lie?"
    compare_no_followup.py     CPU: the transfer matrix with no follow-up question, per read position
    compare_diversity.py       CPU: held-out AUROC against the number of categories trained on
"""
