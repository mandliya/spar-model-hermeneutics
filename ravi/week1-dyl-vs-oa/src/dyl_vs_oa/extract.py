from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from dyl_vs_oa.chat import apply_chat, parse_yes_no, with_generate_followup, with_prefill_followup


OA_PREFILL_POSITIONS = ("prompt_last", "yes", "end_of_turn", "last_newline")


def _last_index(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def gemma_oa_tail_token_ids(tokenizer) -> tuple[int, int, int]:
    """Ids for (newline after `model`, `Yes`, `<end_of_turn>`) on Gemma-3."""
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    if len(yes_ids) != 1:
        raise ValueError(f"Yes did not encode as one token: {yes_ids}")
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot is None or eot < 0:
        raise ValueError("tokenizer has no <end_of_turn>")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise ValueError(f"newline did not encode as one token: {newline_ids}")
    return newline_ids[0], yes_ids[0], eot


def oa_prefill_position_indices(
    input_ids: list[int],
    *,
    newline_id: int,
    yes_id: int,
    eot_id: int,
) -> dict[str, int]:
    """Map OA prefill tail tokens to indices in a closed Gemma chat.

    Expected last four tokens after `Did you respond truthfully?` / `Yes`:

        \\n  Yes  <end_of_turn>  \\n
        ^prompt_last (Tianq)  ^OA paper  ^eot  ^this folder's 05 extractor
    """
    if len(input_ids) < 4:
        raise ValueError("sequence shorter than the OA prefill tail")
    tail = input_ids[-4:]
    expected = [newline_id, yes_id, eot_id, newline_id]
    if tail != expected:
        raise ValueError(f"unexpected OA prefill tail {tail}, expected {expected}")
    last = len(input_ids) - 1
    return {
        "prompt_last": last - 3,
        "yes": last - 2,
        "end_of_turn": last - 1,
        "last_newline": last,
    }


@torch.inference_mode()
def hidden_states_last_token(
    subject,
    texts: list[str],
    layers: list[int] | None = None,
    max_length: int = 2048,
) -> dict[int, np.ndarray]:
    """Residual-stream activations at the last non-pad token for each text."""
    tok = subject.tokenizer
    model = subject.model
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    enc = {k: v.to(subject.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    # hidden_states[0] = embeddings; hidden_states[i] = after layer i
    hs = out.hidden_states
    last = _last_index(enc["attention_mask"])
    batch = last.shape[0]
    wanted = layers if layers is not None else list(range(subject.n_layers))
    acts: dict[int, np.ndarray] = {}
    for layer in wanted:
        tensor = hs[layer + 1]
        gathered = tensor[torch.arange(batch, device=tensor.device), last]
        acts[layer] = gathered.float().cpu().numpy()
    return acts


def prefill_texts(tokenizer, conversations, followup_user: str, followup_assistant: str) -> list[str]:
    texts = []
    for conv in conversations:
        messages = with_prefill_followup(conv.messages, followup_user, followup_assistant)
        texts.append(apply_chat(tokenizer, messages, add_generation_prompt=False))
    return texts


def generate_followup_texts(tokenizer, conversations, followup_user: str) -> list[str]:
    texts = []
    for conv in conversations:
        messages = with_generate_followup(conv.messages, followup_user)
        texts.append(apply_chat(tokenizer, messages, add_generation_prompt=True))
    return texts


@torch.inference_mode()
def extract_prefill_activations(
    subject,
    conversations,
    followup_user: str,
    followup_assistant: str,
    layers: list[int] | None = None,
    batch_size: int = 2,
    max_length: int = 2048,
) -> dict[int, np.ndarray]:
    texts = prefill_texts(subject.tokenizer, conversations, followup_user, followup_assistant)
    acc: dict[int, list[np.ndarray]] = {}
    for i in tqdm(range(0, len(texts), batch_size), desc="prefill acts"):
        batch_acts = hidden_states_last_token(
            subject, texts[i : i + batch_size], layers=layers, max_length=max_length
        )
        for layer, arr in batch_acts.items():
            acc.setdefault(layer, []).append(arr)
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in acc.items()}


@torch.inference_mode()
def hidden_states_oa_positions(
    subject,
    texts: list[str],
    layers: list[int] | None = None,
    max_length: int = 2048,
) -> dict[str, dict[int, np.ndarray]]:
    """Residual at prompt_last / Yes / end_of_turn / last_newline for each text."""
    tok = subject.tokenizer
    newline_id, yes_id, eot_id = gemma_oa_tail_token_ids(tok)
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
    attn = enc["attention_mask"]
    ids = enc["input_ids"]
    last = _last_index(attn)
    batch = last.shape[0]
    pos_index = {name: torch.empty(batch, dtype=torch.long) for name in OA_PREFILL_POSITIONS}
    for i in range(batch):
        end = int(last[i].item()) + 1
        row = ids[i, :end].tolist()
        found = oa_prefill_position_indices(
            row, newline_id=newline_id, yes_id=yes_id, eot_id=eot_id
        )
        for name, idx in found.items():
            pos_index[name][i] = idx
    enc = {k: v.to(subject.device) for k, v in enc.items()}
    out = subject.model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    wanted = layers if layers is not None else list(range(subject.n_layers))
    acts: dict[str, dict[int, np.ndarray]] = {name: {} for name in OA_PREFILL_POSITIONS}
    batch_ix = torch.arange(batch, device=hs[1].device)
    for name, idx in pos_index.items():
        idx = idx.to(hs[1].device)
        for layer in wanted:
            tensor = hs[layer + 1]
            gathered = tensor[batch_ix, idx]
            acts[name][layer] = gathered.float().cpu().numpy()
    return acts


def extract_prefill_activations_at_positions(
    subject,
    conversations,
    followup_user: str,
    followup_assistant: str,
    layers: list[int] | None = None,
    batch_size: int = 2,
    max_length: int = 2048,
) -> dict[str, dict[int, np.ndarray]]:
    texts = prefill_texts(subject.tokenizer, conversations, followup_user, followup_assistant)
    acc: dict[str, dict[int, list[np.ndarray]]] = {name: {} for name in OA_PREFILL_POSITIONS}
    for i in tqdm(range(0, len(texts), batch_size), desc="prefill acts (4 positions)"):
        batch_acts = hidden_states_oa_positions(
            subject, texts[i : i + batch_size], layers=layers, max_length=max_length
        )
        for name, by_layer in batch_acts.items():
            for layer, arr in by_layer.items():
                acc[name].setdefault(layer, []).append(arr)
    return {
        name: {layer: np.concatenate(chunks, axis=0) for layer, chunks in by_layer.items()}
        for name, by_layer in acc.items()
    }


def _yes_no_token_offset(tokenizer, generated_ids: list[int]) -> int | None:
    """Index into generated_ids of the first yes/no token, else last generated token.

    Official DYL scoring must read one token earlier than this (the last prompt
    token). That path lives in dyl-probe-repro/ (copied from Tianq;
    core/prefill.py `prompt_last`). Do not use this extractor for official-probe
    transfer.
    """
    for i in range(len(generated_ids)):
        piece = tokenizer.decode(generated_ids[i : i + 1], skip_special_tokens=True).strip().lower()
        if piece in {"yes", "no", "y", "n"}:
            return i
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parsed = parse_yes_no(text)
    if parsed is None:
        return None
    return max(len(generated_ids) - 1, 0)


@torch.inference_mode()
def extract_generated_activations(
    subject,
    conversations,
    followup_user: str,
    layer: int,
    batch_size: int = 1,
    max_new_tokens: int = 16,
    max_length: int = 2048,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    """Generate DYL answers and read residual stream at the yes/no token.

    Returns (activations [N, D], decoded answers, parsed yes/no labels).
    """
    tok = subject.tokenizer
    model = subject.model
    prompts = generate_followup_texts(tok, conversations, followup_user)
    acts = []
    answers = []
    parsed = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="dyl generate"):
        batch = prompts[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(subject.device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for j, seq in enumerate(gen):
            full_ids = seq.tolist()
            new_ids = full_ids[prompt_len:]
            # Strip pad / eos from the tail for decoding, keep ids for offset.
            while new_ids and new_ids[-1] in {tok.pad_token_id, tok.eos_token_id}:
                new_ids.pop()
            answer = tok.decode(new_ids, skip_special_tokens=True)
            answers.append(answer)
            parsed.append(parse_yes_no(answer))
            offset = _yes_no_token_offset(tok, new_ids)
            unpadded_prompt = enc["input_ids"][j][enc["attention_mask"][j].bool()]
            if offset is None:
                prefix = unpadded_prompt
            else:
                prefix = torch.cat(
                    [unpadded_prompt, torch.tensor(new_ids[: offset + 1], device=unpadded_prompt.device)]
                )
            ids_for_fwd = prefix.unsqueeze(0)
            mask_for_fwd = torch.ones_like(ids_for_fwd)
            token_index = ids_for_fwd.shape[1] - 1
            fwd = model(
                input_ids=ids_for_fwd,
                attention_mask=mask_for_fwd,
                output_hidden_states=True,
                use_cache=False,
            )
            vec = fwd.hidden_states[layer + 1][0, token_index].float().cpu().numpy()
            acts.append(vec)
    return np.stack(acts, axis=0), answers, parsed
