"""The cosine every vector method ends with: two [rows, words] matrices in, one [rows of A, rows of B] table out."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def cosine(A: sparse.csr_matrix, B: sparse.csr_matrix) -> np.ndarray:
    return np.asarray((A @ B.T).toarray() / np.maximum(
        np.sqrt(A.multiply(A).sum(1)) @ np.sqrt(B.multiply(B).sum(1)).T, 1e-12))
