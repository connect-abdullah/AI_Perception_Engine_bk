"""Identity decision policy: ranked candidates -> probable match, new person, or uncertain.

This module only returns a decision. It never creates, updates or otherwise mutates
an identity registry; the caller decides whether to act on the result by calling
`IdentityRegistry.create_identity` or `IdentityRegistry.add_tracklet`.

All policy values must be supplied by the caller. There are deliberately no
defaults: no thresholds have been calibrated on real footage yet, and any number
chosen here would wrongly look like one that had.

Scores are raw cosine similarities in [-1, 1]. They are evidence, not probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from reid.identity import Candidate, IdentityRegistry

# Tolerance for the computed best-minus-second margin only, so that a margin exactly
# equal to `min_margin` is not rejected because of floating-point subtraction error.
# This is a numerical tolerance, not a calibrated value.
_MARGIN_TOLERANCE = 1e-9


class DecisionStatus(str, Enum):
    # Deliberately "probable", never "confirmed": with uncalibrated scores we cannot
    # honestly claim more than that the evidence favours this identity.
    PROBABLE_MATCH = "probable_match"
    # Evidence for every known identity is weak; the caller MAY create a new identity.
    NEW = "new"
    # Evidence is ambiguous or insufficient; neither assign nor create.
    UNCERTAIN = "uncertain"


class DecisionReason(str, Enum):
    CLEAR_MATCH = "clear_match"  # best >= accept and separated from second by >= min_margin
    NO_KNOWN_IDENTITIES = "no_known_identities"  # empty candidate list
    ALL_CANDIDATES_WEAK = "all_candidates_weak"  # best < new_below
    AMBIGUOUS_TOP_CANDIDATES = "ambiguous_top_candidates"  # best >= accept but margin too small
    BETWEEN_THRESHOLDS = "between_thresholds"  # new_below <= best < accept


@dataclass(frozen=True)
class DecisionPolicy:
    """Explicit, caller-supplied decision parameters (raw cosine similarity units).

    accept_at:  best similarity must be >= this to propose an existing identity.
    new_below:  best similarity must be < this to propose a new identity.
                Must be <= accept_at; the gap between them is the uncertain region.
    min_margin: best minus second-best similarity must be >= this to propose an
                existing identity. Use 0 to disable the ambiguity check.
    """

    accept_at: float
    new_below: float
    min_margin: float

    def __post_init__(self):
        for name in ("accept_at", "new_below", "min_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if not -1.0 <= self.accept_at <= 1.0:
            raise ValueError(f"accept_at must be within the cosine range [-1, 1], got {self.accept_at}")
        if not -1.0 <= self.new_below <= 1.0:
            raise ValueError(f"new_below must be within the cosine range [-1, 1], got {self.new_below}")
        if self.new_below > self.accept_at:
            raise ValueError(
                f"new_below ({self.new_below}) must not exceed accept_at ({self.accept_at}); "
                "a score cannot be both strong enough to match and weak enough to be new"
            )
        if not 0.0 <= self.min_margin <= 2.0:
            raise ValueError(f"min_margin must be within [0, 2], got {self.min_margin}")


@dataclass(frozen=True)
class IdentityDecision:
    status: DecisionStatus
    reason: DecisionReason
    person_code: str | None  # set only for PROBABLE_MATCH; never set for NEW or UNCERTAIN
    best_candidate: Candidate | None
    best_similarity: float | None  # raw cosine, not a probability
    second_best_similarity: float | None  # raw cosine, not a probability
    margin: float | None  # best - second best; None with fewer than two candidates
    policy: DecisionPolicy


def _check_ranked(candidates: Sequence[Candidate]) -> None:
    codes = set()
    previous = math.inf
    for c in candidates:
        if not isinstance(c, Candidate):
            raise TypeError(f"Expected Candidate objects, got {type(c).__name__}")
        if not math.isfinite(c.similarity) or not -1.0 <= c.similarity <= 1.0:
            raise ValueError(f"Candidate {c.person_code} has invalid cosine similarity {c.similarity}")
        if c.similarity > previous:
            raise ValueError("Candidates must be ranked by similarity, highest first")
        if c.person_code in codes:
            raise ValueError(f"Duplicate candidate {c.person_code}")
        codes.add(c.person_code)
        previous = c.similarity


def decide(candidates: Sequence[Candidate], policy: DecisionPolicy) -> IdentityDecision:
    """Classify ranked candidates (as returned by `IdentityRegistry.rank_candidates`).

    Rules, in order:
      1. no candidates                                  -> NEW       (no_known_identities)
      2. best <  new_below                              -> NEW       (all_candidates_weak)
      3. best <  accept_at                              -> UNCERTAIN (between_thresholds)
      4. second exists and best - second < min_margin   -> UNCERTAIN (ambiguous_top_candidates)
      5. otherwise                                      -> PROBABLE_MATCH to the best candidate
    """
    if not isinstance(policy, DecisionPolicy):
        raise TypeError("policy must be a DecisionPolicy")
    candidates = list(candidates)
    _check_ranked(candidates)

    if not candidates:
        return IdentityDecision(
            DecisionStatus.NEW, DecisionReason.NO_KNOWN_IDENTITIES, None, None, None, None, None, policy
        )

    best = candidates[0]
    second = candidates[1].similarity if len(candidates) > 1 else None
    margin = None if second is None else best.similarity - second

    def result(status, reason, person_code=None):
        return IdentityDecision(status, reason, person_code, best, best.similarity, second, margin, policy)

    if best.similarity < policy.new_below:
        return result(DecisionStatus.NEW, DecisionReason.ALL_CANDIDATES_WEAK)
    if best.similarity < policy.accept_at:
        return result(DecisionStatus.UNCERTAIN, DecisionReason.BETWEEN_THRESHOLDS)
    if margin is not None and margin < policy.min_margin - _MARGIN_TOLERANCE:
        return result(DecisionStatus.UNCERTAIN, DecisionReason.AMBIGUOUS_TOP_CANDIDATES)
    return result(DecisionStatus.PROBABLE_MATCH, DecisionReason.CLEAR_MATCH, best.person_code)


def decide_for_embedding(registry: IdentityRegistry, embedding, policy: DecisionPolicy) -> IdentityDecision:
    """Rank all identities in `registry` against `embedding` and decide. Does not modify the registry."""
    return decide(registry.rank_candidates(embedding), policy)
