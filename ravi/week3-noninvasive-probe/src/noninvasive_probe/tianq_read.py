"""Tianq's no-follow-up read, so our scores sit next to his.

His positions (tianq/.../core/prefill.py, whole_answer=True):

    prompt_last   last token of "<start_of_turn>model\\n", before the reply
    reply_last    last token of the reply text (not <end_of_turn>)
    reply_mean    mean over the reply's tokens

His classifier (tianq/.../core/training.py): layer 58, standardised logistic
regression, mean log-loss + λ(||w||² + b²), λ = 0.01, L-BFGS to convergence.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm

from noninvasive_probe.paths import bootstrap

bootstrap()

LAYER = 58
L2_LAMBDA = 0.01
POSITIONS = ("prompt_last", "reply_last", "reply_mean")

# His category-trained probes, no follow-up, from the 2026-10-04 Slack note.
# In-category is 5-fold CV on that category. Transfer is the mean of the other categories.
TIANQ_NO_FOLLOWUP = {
    "prompt_last": {"in_category": 0.71, "transfer": 0.50},
    "reply_last": {"in_category": 0.73, "transfer": 0.50},
    "reply_mean": {"in_category": 0.80, "transfer": 0.50},
}


def fit_aisi_logistic(x: np.ndarray, y: np.ndarray, l2_lambda: float = L2_LAMBDA):
    """(mean, std, weight, bias). Same objective as tianq dyl_repro.core.training.fit_weights."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError(f"training needs both labels; got {np.unique(y).tolist()}")
    mean, std = x.mean(0), x.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    z = (x - mean) / std
    n, d = z.shape

    def objective(theta: np.ndarray):
        w, b = theta[:d], theta[d]
        logits = z @ w + b
        residual = (expit(logits) - y) / n
        value = (np.logaddexp(0, logits) - y * logits).mean() + l2_lambda * (w @ w + b * b)
        gradient = np.concatenate([z.T @ residual + 2 * l2_lambda * w, [residual.sum() + 2 * l2_lambda * b]])
        return value, gradient

    fit = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 20_000, "gtol": 1e-9})
    if not fit.success:
        raise RuntimeError(f"probe fit did not converge: {fit.message}")
    return mean, std, fit.x[:d], float(fit.x[d])


def aisi_logits(x: np.ndarray, mean, std, weight, bias) -> np.ndarray:
    std = np.where(np.asarray(std) < 1e-8, 1.0, std)
    return ((np.asarray(x, dtype=np.float64) - mean) / std) @ weight + bias


def _as_id_list(ids) -> list[int]:
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def reply_token_ids(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], dict[str, int], tuple[int, int]]:
    """Token ids and Tianq's indices for a conversation that already ends on the reply."""
    if messages[-1]["role"] != "assistant":
        raise ValueError("conversation must end on the assistant reply")
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    prompt_ids = _as_id_list(prompt)
    answer_ids = _as_id_list(tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"])
    if not answer_ids:
        raise ValueError("empty reply")
    start = len(prompt_ids)
    end = start + len(answer_ids)
    ids = prompt_ids + list(answer_ids)
    return ids, {"prompt_last": start - 1, "reply_last": end - 1}, (start, end)


def _shift(index: int, dropped: int) -> int | None:
    shifted = index - dropped
    return shifted if shifted >= 0 else None


@torch.inference_mode()
def hidden_states_tianq_positions(
    subject,
    conversations: list,
    *,
    layer: int = LAYER,
    max_length: int = 4096,
    batch_size: int = 2,
) -> tuple[dict[str, np.ndarray], int]:
    """Layer-58 residuals at prompt_last, reply_last, and reply_mean.

    Returns activations aligned with `conversations`, and how many rows lost
    prompt_last because left-truncation ate the token before the reply.
    Those rows still have reply_last and reply_mean when the reply itself fit.
    """
    tok = subject.tokenizer
    pooled = {name: [] for name in POSITIONS}
    n_lost_prompt = 0
    texts_meta = []
    for conv in conversations:
        messages = conv.messages if hasattr(conv, "messages") else conv
        ids, positions, span = reply_token_ids(tok, messages)
        texts_meta.append((ids, positions, span))

    for start in tqdm(range(0, len(texts_meta), batch_size), desc="tianq reply positions"):
        chunk = texts_meta[start : start + batch_size]
        dropped = []
        kept_ids = []
        for ids, _positions, _span in chunk:
            drop = max(0, len(ids) - max_length)
            dropped.append(drop)
            kept_ids.append(ids[drop:])
        width = max(len(row) for row in kept_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids = torch.full((len(kept_ids), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, row in enumerate(kept_ids):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        enc = {"input_ids": input_ids.to(subject.device), "attention_mask": attention_mask.to(subject.device)}
        out = subject.model(**enc, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer + 1]
        for i, ((_ids, positions, span), drop) in enumerate(zip(chunk, dropped)):
            reply_start = _shift(span[0], drop)
            reply_end = _shift(span[1] - 1, drop)
            prompt = _shift(positions["prompt_last"], drop)
            last = _shift(positions["reply_last"], drop)
            if prompt is None:
                n_lost_prompt += 1
                prompt_vec = np.full(hidden.shape[-1], np.nan, dtype=np.float32)
            else:
                prompt_vec = hidden[i, prompt].float().cpu().numpy()
            if last is None or reply_start is None or reply_end is None or reply_end < reply_start:
                raise RuntimeError("reply was truncated away; raise --max-length")
            pooled["prompt_last"].append(prompt_vec)
            pooled["reply_last"].append(hidden[i, last].float().cpu().numpy())
            pooled["reply_mean"].append(hidden[i, reply_start : reply_end + 1].float().mean(0).cpu().numpy())
    return {name: np.stack(rows, axis=0) for name, rows in pooled.items()}, n_lost_prompt
