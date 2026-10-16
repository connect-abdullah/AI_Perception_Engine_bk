"""In-memory registry of anonymous global identities (P001, P002, ...) with candidate ranking.

The registry only remembers identities and ranks them against a query embedding.
It never decides whether a tracklet matches someone: creating an identity and adding
a tracklet to an identity are always explicit caller actions, and ranking never
modifies the registry.

Each identity's representative embedding is the L2-normalised mean of the
L2-normalised tracklet embeddings explicitly added to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from reid.similarity import cosine_similarity

_PERSON_CODE = re.compile(r"P(\d{3,})")

# Below this norm the summed contributions have no meaningful direction.
_MIN_SUM_NORM = 1e-6


@dataclass(frozen=True)
class IdentityRecord:
    person_code: str
    embedding: np.ndarray  # unit-norm representative, float32, read-only
    tracklet_count: int


@dataclass(frozen=True)
class Candidate:
    person_code: str
    similarity: float  # cosine similarity in [-1, 1]; not a match probability


def _unit_vector(embedding, expected_dim: int | None) -> np.ndarray:
    v = np.asarray(embedding, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"Embedding must be a non-empty 1-D vector, got shape {v.shape}")
    if expected_dim is not None and v.shape[0] != expected_dim:
        raise ValueError(f"Embedding has dimension {v.shape[0]}, registry uses {expected_dim}")
    if not np.all(np.isfinite(v)):
        raise ValueError("Embedding contains non-finite values")
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Embedding is a zero vector, which has no direction")
    return v / norm


def _code_number(person_code: str) -> int:
    if not isinstance(person_code, str):
        raise ValueError(f"Invalid person code {person_code!r}; expected a string like 'P001'")
    match = _PERSON_CODE.fullmatch(person_code)
    if not match:
        raise ValueError(f"Invalid person code {person_code!r}; expected 'P' followed by 3+ digits, e.g. 'P001'")
    return int(match.group(1))


class IdentityRegistry:
    """Anonymous identity memory. All identities share one embedding dimension.

    The dimension is fixed by the first identity created unless given up front.
    """

    def __init__(self, embedding_dim: int | None = None):
        if embedding_dim is not None and embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self._next_number = 1
        # person_code -> (float64 sum of unit tracklet embeddings, tracklet count)
        self._sums: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._sums)

    def __contains__(self, person_code: object) -> bool:
        return person_code in self._sums

    @property
    def person_codes(self) -> list[str]:
        """Codes in creation order."""
        return list(self._sums)

    def create_identity(self, embedding) -> str:
        """Create a new anonymous identity from one tracklet embedding and return its code.

        Codes are sequential (P001, P002, ...) and never reused.
        """
        unit = _unit_vector(embedding, self.embedding_dim)
        if self.embedding_dim is None:
            self.embedding_dim = unit.shape[0]
        code = f"P{self._next_number:03d}"
        self._next_number += 1
        self._sums[code] = unit
        self._counts[code] = 1
        return code

    def add_tracklet(self, person_code: str, embedding) -> IdentityRecord:
        """Explicitly add one accepted tracklet embedding to an existing identity.

        The representative becomes the normalised mean of all contributing tracklets.
        The registry does not check whether the tracklet is a good match; that is the caller's decision.
        """
        self._require(person_code)
        unit = _unit_vector(embedding, self.embedding_dim)
        new_sum = self._sums[person_code] + unit
        if np.linalg.norm(new_sum) < _MIN_SUM_NORM:
            raise ValueError(f"Tracklet would cancel out {person_code}'s representative; not applied")
        self._sums[person_code] = new_sum
        self._counts[person_code] += 1
        return self.get_identity(person_code)

    def get_identity(self, person_code: str) -> IdentityRecord:
        self._require(person_code)
        total = self._sums[person_code]
        representative = (total / np.linalg.norm(total)).astype(np.float32)
        representative.setflags(write=False)
        return IdentityRecord(person_code, representative, self._counts[person_code])

    def rank_candidates(self, embedding, top_k: int | None = None) -> list[Candidate]:
        """Rank known identities by cosine similarity to `embedding`, highest first.

        Ties are broken by creation order (lower code first). Returns [] for an empty
        registry. Never creates or updates an identity.
        """
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
            raise ValueError(f"top_k must be a positive integer or None, got {top_k!r}")
        query = _unit_vector(embedding, self.embedding_dim)

        candidates = [
            Candidate(code, cosine_similarity(query, self.get_identity(code).embedding))
            for code in self._sums
        ]
        candidates.sort(key=lambda c: (-c.similarity, _code_number(c.person_code)))
        return candidates if top_k is None else candidates[:top_k]

    def _require(self, person_code: str) -> None:
        _code_number(person_code)
        if person_code not in self._sums:
            raise KeyError(f"Unknown person code {person_code!r}")
