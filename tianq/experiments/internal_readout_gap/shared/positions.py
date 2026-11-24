"""Token positions inside a stretch of text, set by share of the text, so they line up across tokenizers.

`shares(tok, ids, n)` takes the token ids of one assistant turn as a chat template renders it (the turn's text, then
whatever the template closes it with) and returns
  inside   n indexes into ids: the first token by which 1/n, 2/n, ... n/n of the turn's text has been read. Text
           length is counted per token; special tokens count as zero. n/n is the turn's last text token.
  end      the index of the template's end-of-turn marker (the last special token after the text), or None if the
           turn is not closed
Tokens after the text that are special or whitespace only (the end marker and the newline a template puts after it)
are not part of the text. Returns None when the turn has no text.
"""

from __future__ import annotations


def shares(tok, ids: list[int], n: int) -> tuple[list[int], int | None] | None:
    lengths = [len(tok.decode([t], skip_special_tokens=True)) for t in ids]
    blank = [not tok.decode([t], skip_special_tokens=True).strip() for t in ids]
    last = len(ids) - 1
    while last >= 0 and blank[last]:
        last -= 1
    if last < 0:
        return None
    closing = [i for i in range(last + 1, len(ids)) if lengths[i] == 0]
    total = sum(lengths[: last + 1])
    inside, read, j = [], 0, -1
    for k in range(1, n + 1):
        while read * n < k * total:
            j += 1
            read += lengths[j]
        inside.append(j)
    return inside, (closing[-1] if closing else None)
