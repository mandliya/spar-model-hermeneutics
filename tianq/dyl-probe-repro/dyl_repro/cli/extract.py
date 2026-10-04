"""GPU: turn one dataset's conversations into stored hidden states, one row at a time.

    python -m dyl_repro.cli.extract --dataset aisi_city_train
    python -m dyl_repro.cli.extract --dataset aisi_city_train --limit 8      # smoke test only
    python -m dyl_repro.cli.extract --dataset ascii_sandbagging --followup all
    python -m dyl_repro.cli.extract --dataset ascii_sandbagging --followup all --dry-run

A conversation that already ends with its answer is prefilled as is. One that
ends with the follow-up first gets the model's own answer on-policy (temperature
0, a separate generate call), then is prefilled the same way. Both positions
come from that one prefill forward pass, and the full answer is saved. With
the follow-up "none" the conversation ends at the tested reply, which is run in
full and kept at core.prefill.REPLY_POSITIONS instead.

Safe to kill and rerun: rows already on the Hub or in scratch are skipped.
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version

import torch
import transformers
from tqdm import tqdm

from dyl_repro import datasets
from dyl_repro.core.conversation import Conversation
from dyl_repro.core.model import MODEL_NAME, MODEL_REVISION, load_model, load_tokenizer
from dyl_repro.core.onpolicy import MAX_NEW_TOKENS, TEMPERATURE, generate_answer
from dyl_repro.core.prefill import POSITIONS, REPLY_POSITIONS, PrefillTokens, hidden_states_at, prefill_tokens
from dyl_repro.core.store import SCRATCH, RowStore
from dyl_repro.datasets.followups import NO_FOLLOWUP

INDEX_MEANING = "transformers hidden_states: 0 = embeddings, i = output of block i-1, last = after final norm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=datasets.NAMES)
    parser.add_argument("--followup", nargs="+", help="for datasets with swappable follow-ups: names, or 'all'")
    parser.add_argument("--limit", type=int, help="smoke test: extract only this many pending rows per run")
    parser.add_argument("--dry-run", action="store_true", help="build and tokenize every conversation, then stop")
    return parser.parse_args()


def pending(dataset: str, key: str, followup: str | None, limit: int | None) -> tuple[RowStore, list[Conversation]]:
    """The run's store, and the conversations not yet stored locally or on the Hub."""
    conversations = datasets.conversations(dataset, followup)
    store = RowStore(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], SCRATCH, key)
    store.ensure_repo()
    done = store.done_uids()
    todo = [c for c in conversations if c.uid not in done][:limit]
    n_on_policy = sum(c.needs_answer for c in todo)
    print(
        f"\n{key}: {len(conversations)} conversations, {len(done)} already stored, {len(todo)} to do"
        f" ({n_on_policy} answered on-policy){f' (SMOKE TEST, --limit {limit})' if limit else ''}",
        flush=True,
    )
    return store, todo


def precheck(tokenizer, todo: list[Conversation], whole_answer: bool) -> dict[str, PrefillTokens]:
    """Tokenize before touching the model, so a bad transcript fails in seconds.

    Prefilled answers are tokenized in full; on-policy conversations only have a
    prompt yet. The probe reads prompt_last, which exists for every row, so a
    labelled answer that opens with something other than yes/no (AISI's varied
    deception has '...Yes.') is kept, as AISI keeps it, and named here.
    """
    prefilled = {}
    for c in todo:
        if c.needs_answer:
            tokenizer.apply_chat_template(c.messages, add_generation_prompt=True, tokenize=True)
        else:
            prefilled[c.uid] = prefill_tokens(tokenizer, c.messages, whole_answer=whole_answer)
    unusual = [
        (c.uid, c.messages[-1]["content"][:40])
        for c in todo
        if not whole_answer and c.uid in prefilled and c.label is not None
        and not prefilled[c.uid].answer_first_is_yes_no
    ]
    print(f"tokenized {len(todo)} rows; labelled answers not opening with yes/no: {len(unusual)}", flush=True)
    for uid, answer in unusual:
        print(f"    {uid}: {answer!r}", flush=True)
    return prefilled


def answer_and_tokens(model, tokenizer, conversation: Conversation, prefilled: dict) -> tuple[list, PrefillTokens, dict]:
    """(messages ending with the answer, their prefill tokens, how the answer was obtained)."""
    if not conversation.needs_answer:
        return conversation.messages, prefilled[conversation.uid], {"source": "prefilled"}
    generated = generate_answer(model, tokenizer, conversation.messages)
    messages = conversation.messages + [{"role": "assistant", "content": generated.text}]
    answer = {
        "source": "on_policy",
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hit_cap": generated.hit_cap,
        "answer_ids": generated.answer_ids,
    }
    return messages, prefill_tokens(tokenizer, messages, generated.answer_ids), answer


def row_record(conversation, messages, tokens, answer, hidden_states, fidelity, context: dict, tokenizer) -> dict:
    return {
        "uid": conversation.uid,
        "dataset": context["dataset"],
        "followup": context["followup"],
        "label": conversation.label,
        "provenance": conversation.provenance,
        "messages": messages,
        "answer": answer,
        # Exactly what the model was fed, as text and as ids.
        "rendered_text": tokenizer.decode(tokens.input_ids),
        "input_ids": tokens.input_ids,
        "positions": tokens.positions,
        "answer_first_text": tokens.answer_first_text,
        "answer_first_is_yes_no": tokens.answer_first_is_yes_no,
        "fidelity": fidelity,
        "hidden_states": {
            "positions": list(context["positions"]),
            "shape": list(hidden_states.shape),
            "dtype": str(hidden_states.dtype),
            "index_meaning": INDEX_MEANING,
        },
        "model": context["model"],
        "versions": context["versions"],
        "run_limit": context["run_limit"],
    }


def extract_rows(model, tokenizer, store: RowStore, todo: list[Conversation], prefilled: dict, context: dict) -> None:
    # finally: an exception or Ctrl-C still pushes every row already finished.
    try:
        for conversation in tqdm(todo, desc=store.dataset):
            messages, tokens, answer = answer_and_tokens(model, tokenizer, conversation, prefilled)
            hidden_states, fidelity = hidden_states_at(model, tokens, context["positions"])
            fidelity["top_token_texts"] = [tokenizer.decode([i]) for i in fidelity["top_token_ids"]]
            record = row_record(conversation, messages, tokens, answer, hidden_states, fidelity, context, tokenizer)
            store.put(conversation.uid, record, hidden_states, context["positions"])
    finally:
        store.sync()
    print(f"done. {store.dataset}: {len(todo)} rows extracted and synced to {store.repo_id}", flush=True)


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer()
    model = None
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "lie_detectors": version("lie-detectors")}

    for key, followup in datasets.runs(args.dataset, args.followup):
        store, todo = pending(args.dataset, key, followup, args.limit)
        if not todo:
            continue
        whole_answer = followup == NO_FOLLOWUP
        prefilled = precheck(tokenizer, todo, whole_answer)
        if args.dry_run:
            sent = todo[0].messages[-1 if todo[0].needs_answer else -2]["content"]
            print(f"dry run: follow-up as sent: {sent!r}", flush=True)
            continue
        if model is None:
            model = load_model()
        context = {
            "dataset": args.dataset,
            "followup": followup,
            "run_limit": args.limit,
            "positions": REPLY_POSITIONS if whole_answer else POSITIONS,
            "model": {"name": MODEL_NAME, "revision": MODEL_REVISION, "class": type(model).__name__},
            "versions": versions,
        }
        extract_rows(model, tokenizer, store, todo, prefilled, context)


if __name__ == "__main__":
    main()
