from __future__ import annotations

from typing import Any

YES_STRINGS = ("yes", "y", "true")
NO_STRINGS = ("no", "n", "false")


def apply_chat(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        # Gemma-3 chat templates sometimes reject a system role.
        repaired = []
        for m in messages:
            if m["role"] == "system":
                repaired.append({"role": "user", "content": m["content"]})
                repaired.append({"role": "assistant", "content": "Understood."})
            else:
                repaired.append(m)
        return tokenizer.apply_chat_template(repaired, **kwargs)


def conversation_with_completion(
    user: str,
    completion: str,
    system: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": completion})
    return messages


def with_prefill_followup(
    messages: list[dict[str, str]],
    user: str,
    assistant: str,
) -> list[dict[str, str]]:
    return [*messages, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def with_generate_followup(messages: list[dict[str, str]], user: str) -> list[dict[str, str]]:
    return [*messages, {"role": "user", "content": user}]


def parse_yes_no(text: str) -> int | None:
    """Return 1 for yes, 0 for no, None if unparseable."""
    stripped = text.strip().lower()
    if not stripped:
        return None
    first = stripped.split()[0].strip(".,:;!?\"'")
    if first in YES_STRINGS:
        return 1
    if first in NO_STRINGS:
        return 0
    if "yes" in stripped and "no" not in stripped:
        return 1
    if "no" in stripped and "yes" not in stripped:
        return 0
    return None
