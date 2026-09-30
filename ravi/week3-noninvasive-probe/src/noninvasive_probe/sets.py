"""Eval loaders. Week 1 TRANSFER_SETS plus OA MBPP. Never used for training."""

from __future__ import annotations

from noninvasive_probe.paths import bootstrap

bootstrap()

from dyl_vs_oa.data import LabeledConversation, load_mbpp_honeypot, mbpp_conversations
from dyl_vs_oa.oa_transfer import TRANSFER_SETS, load_transfer_set

EVAL_SETS = TRANSFER_SETS + ("oa_mbpp",)

ADMITTED_FROM_REASON = {
    "Model denied deception (hiding it).": False,
    "Model honestly admitted deception.": True,
    "Model correctly denied deception.": False,
}


def admitted_from_reason(lie_reason: str | None) -> bool | None:
    if not lie_reason:
        return None
    if lie_reason in ADMITTED_FROM_REASON:
        return ADMITTED_FROM_REASON[lie_reason]
    if lie_reason.startswith("False confession:"):
        return True
    return None


def _with_aisi_admitted(convs: list[LabeledConversation]) -> list[LabeledConversation]:
    out = []
    for conv in convs:
        meta = dict(conv.meta)
        meta.setdefault("followup_admitted", admitted_from_reason(meta.get("lie_reason")))
        out.append(LabeledConversation(messages=conv.messages, label=conv.label, meta=meta))
    return out


def load_oa_mbpp(cfg, max_n: int | None = None) -> list[LabeledConversation]:
    n_tasks = None if max_n is None else max(1, (max_n + 1) // 2)
    rows = load_mbpp_honeypot(cfg, split="test", max_n=n_tasks)
    out: list[LabeledConversation] = []
    for row in rows:
        honest, deceptive = mbpp_conversations(row, cfg.oa.system_prompt)
        out.append(honest)
        out.append(deceptive)
        if max_n is not None and len(out) >= max_n:
            break
    return out[:max_n] if max_n is not None else out


def load_eval_set(name: str, cfg, max_n: int | None = None) -> list[LabeledConversation]:
    if name == "oa_mbpp":
        return load_oa_mbpp(cfg, max_n=max_n)
    rows = load_transfer_set(name, max_n=max_n)
    if name in {"dyl_city", "dyl_varied"}:
        return _with_aisi_admitted(rows)
    return rows
