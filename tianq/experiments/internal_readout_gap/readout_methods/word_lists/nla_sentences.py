"""From NLA sentences to readouts. An NLA readout is a few sampled sentences about one activation.

A row's sentences are a list of word lists (shared.words.text_words of each sentence; a sentence the NLA left empty
is dropped). The plain way pools every word of every sentence into one bag, a word's value being how often it is
used (readout_type.from_counts on the pooled words). Two ways that keep more of the sentences:
  phrase_readouts     every run of two or three words inside a sentence is counted in place of single words
  agreement_readouts  a word's value is the number of the sentences that have it, so a word one sample made up
                      counts least
"""

from __future__ import annotations

import collections

from ..readout_type import Readouts, from_counts

Sentences = list[list[list[str]]]   # per row: its sentences, each a list of words
PHRASE_SIZES = (2, 3)


def phrase_readouts(sentences: Sentences) -> Readouts:
    return [from_counts(collections.Counter(" ".join(s[i: i + size]) for s in point for size in PHRASE_SIZES
                                            for i in range(len(s) - size + 1))) for point in sentences]


def agreement_readouts(sentences: Sentences) -> Readouts:
    return [from_counts(collections.Counter(w for s in point for w in set(s))) for point in sentences]
