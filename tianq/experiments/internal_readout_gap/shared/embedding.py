"""Readouts as meaning vectors: the one place that says how words and NLA sentences are embedded.

A small multilingual sentence-embedding model (EMBEDDER, run on CPU) turns each J-lens word and each NLA explanation
into a unit vector, so "retry" and "trying", or a word and its translation, land close together where identical-word
counts (shared.words) score them as unrelated. Both experiments use this: distillation_similarity for cosine between
two models' readouts, lie_detection_gap for classifier features.
"""

from __future__ import annotations

import numpy as np
import torch

EMBEDDER = "intfloat/multilingual-e5-small"


class Embedder:
    """Unit-length meaning vectors for short texts, cached by text."""

    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(EMBEDDER)
        self.model = AutoModel.from_pretrained(EMBEDDER).eval()
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def add(self, texts: set[str], batch: int = 256) -> None:
        todo = sorted(texts - set(self.cache), key=len)
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            enc = self.tok([f"query: {t}" for t in chunk], padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"][..., None].float()
            vec = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).numpy()
            self.cache.update(zip(chunk, vec))
        print(f"embedded {len(todo)} new texts ({len(self.cache)} in all)", flush=True)

    def mean(self, texts: list[str]) -> np.ndarray:
        """One unit vector for several texts; NaNs when there are none (an NLA that returned nothing)."""
        if not texts:
            return np.full(next(iter(self.cache.values())).shape, np.nan)
        v = np.mean([self.cache[t] for t in texts], axis=0)
        return v / np.linalg.norm(v)

    def matrix(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.cache[t] for t in texts])
