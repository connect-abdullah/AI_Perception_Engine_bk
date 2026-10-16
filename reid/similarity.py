"""Similarity between ReID embeddings."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D embeddings.

    Returns a float in [-1, 1]; 1 means identical direction. Inputs need not be
    pre-normalised. OSNet embeddings are non-negative (ReLU output), so in
    practice scores fall in [0, 1].

    The score is a raw similarity, not an identity decision: no threshold is
    implied or applied here.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Embeddings must be 1-D, got shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes differ: {a.shape} vs {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("Embeddings contain non-finite values")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    score = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, score))
