"""Original-turn token pools (no follow-up). See NOTES.md §2."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from noninvasive_probe.paths import bootstrap

bootstrap()

from dyl_vs_oa.chat import apply_chat
from dyl_vs_oa.extract import hidden_states_last_token

ORIGINAL_TURN_POSITIONS = (
    "last_content",
    "end_of_turn",
    "last_newline",
    "mean_answer",
    "mean_excl_last_5",
    "mean_first_5",
)
EXCLUDE_LAST_N = 5
FIRST_K = 5


def _last_index(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def gemma_template_ids(tokenizer) -> tuple[int, int, int]:
    start = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if start is None or start < 0 or eot is None or eot < 0:
        raise ValueError("tokenizer is missing Gemma turn tokens")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise ValueError(f"newline did not encode as one token: {newline_ids}")
    return start, eot, newline_ids[0]


def original_turn_span(
    input_ids: list[int],
    *,
    start_id: int,
    eot_id: int,
    newline_id: int,
) -> dict[str, int]:
    """Inclusive indices for the last assistant turn in a closed Gemma chat."""
    if not input_ids:
        raise ValueError("empty sequence")
    last = len(input_ids) - 1
    eot_positions = [i for i, tok in enumerate(input_ids) if tok == eot_id]
    if not eot_positions:
        return {
            "content_start": 0,
            "content_end": last,
            "last_content": last,
            "end_of_turn": last,
            "last_newline": last,
            "last_user": 0,
        }
    eot = eot_positions[-1]
    last_content = eot - 1 if eot > 0 else eot
    starts = [i for i, tok in enumerate(input_ids[:eot]) if tok == start_id]
    if starts:
        i = starts[-1] + 1
        while i < eot and input_ids[i] != newline_id:
            i += 1
        content_start = i + 1 if i < eot else starts[-1] + 1
    else:
        content_start = 0
    if content_start > last_content:
        content_start = last_content
    # Token before the turn that precedes the assistant: the user's last content token.
    prev_eots = [i for i in eot_positions if starts and i < starts[-1]]
    if prev_eots:
        user_eot = prev_eots[-1]
        last_user = user_eot - 1 if user_eot > 0 else 0
    else:
        last_user = 0
    return {
        "content_start": content_start,
        "content_end": last_content,
        "last_content": last_content,
        "end_of_turn": eot,
        "last_newline": last,
        "last_user": last_user,
    }


def original_turn_pool_indices(
    span: dict[str, int],
    name: str,
    *,
    exclude_last: int = EXCLUDE_LAST_N,
    first_k: int = FIRST_K,
) -> list[int]:
    start, end = span["content_start"], span["content_end"]
    content = list(range(start, end + 1)) if end >= start else [span["last_content"]]
    if name == "last_content":
        return [span["last_content"]]
    if name == "end_of_turn":
        return [span["end_of_turn"]]
    if name == "last_newline":
        return [span["last_newline"]]
    if name == "mean_answer":
        return content
    if name == "mean_excl_last_5":
        if len(content) <= exclude_last:
            return content[:1]
        return content[:-exclude_last]
    if name == "mean_first_5":
        return content[: min(first_k, len(content))]
    if name == "last_user":
        return [span["last_user"]]
    raise KeyError(f"unknown original-turn position {name!r}")


def original_turn_texts(tokenizer, conversations) -> list[str]:
    return [apply_chat(tokenizer, conv.messages, add_generation_prompt=False) for conv in conversations]


def _pool_hidden(tensor, batch_ix: int, indices: list[int]):
    idx = torch.tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor[batch_ix, idx].mean(dim=0)


@torch.inference_mode()
def hidden_states_original_turn_positions(
    subject,
    texts: list[str],
    layers: list[int] | None = None,
    max_length: int = 2048,
    positions: tuple[str, ...] = ORIGINAL_TURN_POSITIONS,
) -> dict[str, dict[int, np.ndarray]]:
    tok = subject.tokenizer
    start_id, eot_id, newline_id = gemma_template_ids(tok)
    prev_trunc = getattr(tok, "truncation_side", "right")
    tok.truncation_side = "left"
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    tok.truncation_side = prev_trunc
    last = _last_index(enc["attention_mask"])
    ids = enc["input_ids"]
    batch = last.shape[0]
    spans = []
    for i in range(batch):
        end = int(last[i].item()) + 1
        spans.append(
            original_turn_span(
                ids[i, :end].tolist(),
                start_id=start_id,
                eot_id=eot_id,
                newline_id=newline_id,
            )
        )
    enc_dev = {k: v.to(subject.device) for k, v in enc.items()}
    out = subject.model(**enc_dev, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    wanted = layers if layers is not None else list(range(subject.n_layers))
    acts: dict[str, dict[int, np.ndarray]] = {name: {} for name in positions}
    for name in positions:
        pooled = {layer: [] for layer in wanted}
        for i, span in enumerate(spans):
            idx = original_turn_pool_indices(span, name)
            for layer in wanted:
                pooled[layer].append(_pool_hidden(hs[layer + 1], i, idx))
        for layer in wanted:
            acts[name][layer] = torch.stack(pooled[layer], dim=0).float().cpu().numpy()
    return acts


def extract_original_turn_activations_at_positions(
    subject,
    conversations,
    layers: list[int] | None = None,
    batch_size: int = 2,
    max_length: int = 2048,
    positions: tuple[str, ...] = ORIGINAL_TURN_POSITIONS,
) -> dict[str, dict[int, np.ndarray]]:
    texts = original_turn_texts(subject.tokenizer, conversations)
    acc: dict[str, dict[int, list[np.ndarray]]] = {name: {} for name in positions}
    for i in tqdm(range(0, len(texts), batch_size), desc="original-turn acts"):
        batch_acts = hidden_states_original_turn_positions(
            subject,
            texts[i : i + batch_size],
            layers=layers,
            max_length=max_length,
            positions=positions,
        )
        for name, by_layer in batch_acts.items():
            for layer, arr in by_layer.items():
                acc[name].setdefault(layer, []).append(arr)
    return {
        name: {layer: np.concatenate(chunks, axis=0) for layer, chunks in by_layer.items()}
        for name, by_layer in acc.items()
    }


def extract_last_token_activations(
    subject,
    texts: list[str],
    layers: list[int] | None = None,
    batch_size: int = 2,
    max_length: int = 2048,
) -> dict[int, np.ndarray]:
    acc: dict[int, list[np.ndarray]] = {}
    for i in tqdm(range(0, len(texts), batch_size), desc="last-token acts"):
        batch_acts = hidden_states_last_token(
            subject, texts[i : i + batch_size], layers=layers, max_length=max_length
        )
        for layer, arr in batch_acts.items():
            acc.setdefault(layer, []).append(arr)
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in acc.items()}
