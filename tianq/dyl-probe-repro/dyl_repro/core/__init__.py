"""Dataset-agnostic pieces: they work on any Conversation and any follow-up.

    probe.py         AISI's DYL probe: pinned checkpoint, where it reads, activations in -> scores out
    model.py         the subject model, pinned
    conversation.py  the one shape every dataset produces
    prefill.py       conversation with its answer -> hidden states at both positions
    onpolicy.py      the model answers the follow-up itself (temperature 0)
    followup.py      a follow-up question: text + parser for whether an answer admits the lie
    metrics.py       AUROC, balanced accuracy, TPR/FPR, random-direction null
    store.py         per-row HF storage, resume, reading back
    logs.py          raw log upload with secrets masked
"""
