from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from dyl_vs_oa.chat import apply_chat, parse_yes_no, with_generate_followup, with_prefill_followup


def _last_index(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


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
