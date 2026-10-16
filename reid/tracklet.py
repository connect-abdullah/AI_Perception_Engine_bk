"""Tracklet representation: several crop embeddings of one local track -> one embedding.

Baseline aggregation: L2-normalise each crop embedding, take the arithmetic mean,
then L2-normalise the mean. Every crop gets equal weight. Works for any embedding
dimension, so it is unaffected by swapping the underlying ReID model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from reid.embedder import SupportsEmbed

# Below this norm the mean vector has no meaningful direction (crop embeddings cancel out).
_MIN_MEAN_NORM = 1e-6


def aggregate_embeddings(embeddings: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Return the unit-norm mean direction of the given crop embeddings.

    `embeddings` is a sequence of 1-D vectors or a 2-D array of shape (n_crops, dim).
    Inputs need not be pre-normalised; each is normalised first so that every crop
    contributes equally. The result is a float32 1-D vector of length `dim` and does
    not depend on the order of the inputs.
    """
    vectors = [np.asarray(e, dtype=np.float64) for e in embeddings]
    if not vectors:
        raise ValueError("Need at least one embedding to aggregate")
    for k, v in enumerate(vectors):
        if v.ndim != 1 or v.size == 0:
            raise ValueError(f"Embedding {k} must be a non-empty 1-D vector, got shape {v.shape}")
    dims = {v.shape[0] for v in vectors}
    if len(dims) > 1:
        raise ValueError(f"Embeddings have mismatched dimensions: {sorted(dims)}")

    matrix = np.stack(vectors)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Embeddings contain a zero vector, which has no direction")
    unit = matrix / norms

    # Sum in a canonical row order so the floating-point result is identical for any input order.
    unit = unit[np.lexsort(unit.T[::-1])]
    mean = unit.mean(axis=0)

    mean_norm = np.linalg.norm(mean)
    if mean_norm < _MIN_MEAN_NORM:
        raise ValueError("Crop embeddings cancel out; the tracklet mean has no direction")
    return (mean / mean_norm).astype(np.float32)


def embed_tracklet(images: Iterable, embedder: SupportsEmbed) -> np.ndarray:
    """Embed each crop of one tracklet with `embedder` and aggregate the results.

    `images` are whatever `embedder.embed` accepts (file paths or BGR arrays for `reid.Embedder`).
    Crop selection is the caller's responsibility; every image passed in is used.
    """
    images = list(images)
    if not images:
        raise ValueError("Tracklet has no images")
    return aggregate_embeddings([embedder.embed(image) for image in images])
