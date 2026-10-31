"""Reproduce AISI's official "Did you lie?" probe for Gemma-3-27B-IT, then apply it.

    core/      works on any dataset: the probe, the model, prefill, on-policy generation, storage, metrics
    datasets/  one sub-package per data source, each with its own check
    cli/       entry points (extract, evaluate, inspect), run through ../run.sh

The interface downstream code uses is dyl_repro.core.probe.DylProbe: activations
in, scores out. The experiments that use it live in tianq/experiments/.
"""
