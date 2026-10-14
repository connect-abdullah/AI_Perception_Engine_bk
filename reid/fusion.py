"""Conservative fusion of a visual identity decision with camera/time context evidence.

Context can only veto: a physically impossible move downgrades a visual
PROBABLE_MATCH to UNCERTAIN. It never promotes weak visual evidence, never turns
NEW into a match, and UNKNOWN context changes nothing.

The caller must evaluate `context` for the move from the matched identity's last
sighting to the current tracklet; this module cannot check that pairing.

Pure function: no scores are changed, nothing is created or updated, and the
inputs are kept by reference in the result rather than copied or altered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from reid.context import ContextEvidence, ContextStatus
from reid.decision import DecisionStatus, IdentityDecision


class FusionReason(str, Enum):
    MATCH_CONTEXT_PLAUSIBLE = "match_context_plausible"  # PROBABLE_MATCH kept
    MATCH_CONTEXT_UNKNOWN = "match_context_unknown"  # PROBABLE_MATCH kept
    MATCH_NO_CONTEXT = "match_no_context"  # PROBABLE_MATCH kept; no context supplied
    MATCH_VETOED_BY_CONTEXT = "match_vetoed_by_context"  # PROBABLE_MATCH -> UNCERTAIN
    VISUAL_UNCERTAIN = "visual_uncertain"  # UNCERTAIN kept; context cannot promote
    VISUAL_NEW = "visual_new"  # NEW kept; context cannot turn it into a match


@dataclass(frozen=True)
class FusedDecision:
    status: DecisionStatus
    person_code: str | None  # set only when status is PROBABLE_MATCH
    reason: FusionReason
    visual: IdentityDecision  # original visual decision, unchanged (raw cosine evidence lives here)
    context: ContextEvidence | None  # original context evidence, unchanged; None if not supplied

    @property
    def visual_status(self) -> DecisionStatus:
        return self.visual.status

    @property
    def visual_person_code(self) -> str | None:
        return self.visual.person_code

    @property
    def context_status(self) -> ContextStatus | None:
        return None if self.context is None else self.context.status

    @property
    def changed_by_context(self) -> bool:
        return self.status is not self.visual.status


def fuse(visual: IdentityDecision, context: ContextEvidence | None) -> FusedDecision:
    """Combine a visual decision with context evidence (or None if no context is available).

    | visual         | context            | result                          |
    |----------------|--------------------|---------------------------------|
    | PROBABLE_MATCH | PLAUSIBLE          | PROBABLE_MATCH                  |
    | PROBABLE_MATCH | UNKNOWN / None     | PROBABLE_MATCH                  |
    | PROBABLE_MATCH | IMPOSSIBLE         | UNCERTAIN (vetoed)              |
    | UNCERTAIN      | any                | UNCERTAIN                       |
    | NEW            | any                | NEW                             |
    """
    if not isinstance(visual, IdentityDecision):
        raise TypeError(f"visual must be an IdentityDecision, got {type(visual).__name__}")
    if context is not None and not isinstance(context, ContextEvidence):
        raise TypeError(f"context must be a ContextEvidence or None, got {type(context).__name__}")

    def result(status, reason, person_code=None):
        return FusedDecision(status, person_code, reason, visual, context)

    if visual.status is DecisionStatus.UNCERTAIN:
        return result(DecisionStatus.UNCERTAIN, FusionReason.VISUAL_UNCERTAIN)
    if visual.status is DecisionStatus.NEW:
        # An impossible move is evidence against a specific known identity; that only
        # supports "not an existing person", so it is no reason to block NEW.
        return result(DecisionStatus.NEW, FusionReason.VISUAL_NEW)

    # PROBABLE_MATCH
    if context is None:
        return result(DecisionStatus.PROBABLE_MATCH, FusionReason.MATCH_NO_CONTEXT, visual.person_code)
    if context.status is ContextStatus.IMPOSSIBLE:
        return result(DecisionStatus.UNCERTAIN, FusionReason.MATCH_VETOED_BY_CONTEXT)
    if context.status is ContextStatus.PLAUSIBLE:
        return result(DecisionStatus.PROBABLE_MATCH, FusionReason.MATCH_CONTEXT_PLAUSIBLE, visual.person_code)
    return result(DecisionStatus.PROBABLE_MATCH, FusionReason.MATCH_CONTEXT_UNKNOWN, visual.person_code)
