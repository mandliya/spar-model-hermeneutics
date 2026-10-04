"""Analyses across Gigi's categories, each a module of tables and a module drawing them.

    wordings.py, wording_charts.py
        just asking vs the probe, each with its best wording, and the hard-to-catch lies
    generalization.py, generalization_charts.py
        probes trained on one category, tested on the others (John's question); the charts draw no_followup too
    outcome_flags.py, outcome_flag_charts.py
        what the released probe flags on each kind of answer to "did you lie?"
    no_followup.py
        the same transfer matrix with no follow-up question, per read position (drawn by generalization_charts)
    chart_style.py
        the colours and style every chart shares

Each is run by its cli/compare_*.py from results already on the Hub; none of
them touches activations.
"""
