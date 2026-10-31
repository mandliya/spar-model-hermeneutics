"""Experiments built on the DYL pipeline (../dyl-probe-repro), one package each; see README.md.

    probe_vs_just_asking/     the released probe vs just asking on Gigi's categories, and what it flags
    cross_category_transfer/  does a probe trained on one kind of lie catch the others? (Gemma, Qwen)
    hermeneutics/             the probe faithfulness gap: a substitute model reads the target's conversations
    alignment_faking_gate/    can a reader's faking probe catch HAL9000's alignment faking with the reasoning hidden?
    common/                   what every chart shares

Every step runs through ../dyl-probe-repro/run.sh, which puts both folders on PYTHONPATH and logs the run.
"""
