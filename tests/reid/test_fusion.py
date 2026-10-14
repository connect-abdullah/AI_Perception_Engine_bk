"""Fusion tests. All policy values, cameras and timestamps are synthetic."""

import copy
import dataclasses

import numpy as np
import pytest

from reid.context import ContextStatus, ObservationContext, TransitionRule, TransitionRules
from reid.decision import DecisionPolicy, DecisionStatus, decide, decide_for_embedding
from reid.fusion import FusedDecision, FusionReason, fuse
from reid.identity import Candidate, IdentityRegistry

POLICY = DecisionPolicy(accept_at=0.7, new_below=0.4, min_margin=0.1)
RULES = TransitionRules([
    TransitionRule.window("CAM_A", "CAM_B", min_seconds=5, max_seconds=120),
    TransitionRule.disallowed("CAM_A", "CAM_D"),
])


def ranked(*scores):
    return [Candidate(f"P{k + 1:03d}", s) for k, s in enumerate(scores)]


VISUAL = {
    DecisionStatus.PROBABLE_MATCH: lambda: decide(ranked(0.9, 0.5), POLICY),
    DecisionStatus.UNCERTAIN: lambda: decide(ranked(0.85, 0.8), POLICY),
    DecisionStatus.NEW: lambda: decide(ranked(0.3, 0.1), POLICY),
}

CONTEXT = {
    ContextStatus.PLAUSIBLE: lambda: RULES.evaluate(ObservationContext("CAM_A", 0.0), ObservationContext("CAM_B", 40.0)),
    ContextStatus.UNKNOWN: lambda: RULES.evaluate(ObservationContext("CAM_A", 0.0), ObservationContext("CAM_C", 40.0)),
    ContextStatus.IMPOSSIBLE: lambda: RULES.evaluate(ObservationContext("CAM_A", 0.0), ObservationContext("CAM_B", 1.0)),
}


def test_fixtures_have_intended_statuses():
    for status, build in VISUAL.items():
        assert build().status is status
    for status, build in CONTEXT.items():
        assert build().status is status


# --------------------------------------------------------------------------- full 3 x 3 table (+ no context)

M, U, N = DecisionStatus.PROBABLE_MATCH, DecisionStatus.UNCERTAIN, DecisionStatus.NEW
P, K, X = ContextStatus.PLAUSIBLE, ContextStatus.UNKNOWN, ContextStatus.IMPOSSIBLE

TABLE = [
    (M, P, M, FusionReason.MATCH_CONTEXT_PLAUSIBLE),
    (M, K, M, FusionReason.MATCH_CONTEXT_UNKNOWN),
    (M, X, U, FusionReason.MATCH_VETOED_BY_CONTEXT),
    (U, P, U, FusionReason.VISUAL_UNCERTAIN),
    (U, K, U, FusionReason.VISUAL_UNCERTAIN),
    (U, X, U, FusionReason.VISUAL_UNCERTAIN),
    (N, P, N, FusionReason.VISUAL_NEW),
    (N, K, N, FusionReason.VISUAL_NEW),
    (N, X, N, FusionReason.VISUAL_NEW),
]


@pytest.mark.parametrize("visual_status, context_status, expected, reason", TABLE)
def test_fusion_table(visual_status, context_status, expected, reason):
    visual, context = VISUAL[visual_status](), CONTEXT[context_status]()
    fused = fuse(visual, context)
    assert fused.status is expected
    assert fused.reason is reason
    assert fused.visual_status is visual_status
    assert fused.context_status is context_status
    # only a kept match carries a person code
    assert fused.person_code == ("P001" if expected is M else None)
    assert fused.changed_by_context == (expected is not visual_status)


@pytest.mark.parametrize(
    "visual_status, expected, reason",
    [(M, M, FusionReason.MATCH_NO_CONTEXT), (U, U, FusionReason.VISUAL_UNCERTAIN), (N, N, FusionReason.VISUAL_NEW)],
)
def test_no_context_leaves_visual_unchanged(visual_status, expected, reason):
    fused = fuse(VISUAL[visual_status](), None)
    assert fused.status is expected and fused.reason is reason and fused.context is None
    assert fused.context_status is None and not fused.changed_by_context


# --------------------------------------------------------------------------- principles


def test_probable_match_downgraded_by_impossible_context():
    visual = VISUAL[M]()
    fused = fuse(visual, CONTEXT[X]())
    assert fused.status is DecisionStatus.UNCERTAIN and fused.person_code is None
    assert fused.reason is FusionReason.MATCH_VETOED_BY_CONTEXT
    assert fused.visual_person_code == "P001"  # the vetoed candidate is still visible for review
    assert fused.changed_by_context


def test_veto_gives_uncertain_not_new():
    # Context only says this identity is unlikely; it is not evidence that the person is new.
    assert fuse(VISUAL[M](), CONTEXT[X]()).status is DecisionStatus.UNCERTAIN


def test_plausible_never_promotes_uncertain():
    for visual in (decide(ranked(0.85, 0.8), POLICY), decide(ranked(0.55), POLICY)):  # ambiguous and grey zone
        fused = fuse(visual, CONTEXT[P]())
        assert fused.status is DecisionStatus.UNCERTAIN and fused.person_code is None


def test_plausible_never_promotes_new():
    for visual in (decide(ranked(0.3), POLICY), decide([], POLICY)):
        fused = fuse(visual, CONTEXT[P]())
        assert fused.status is DecisionStatus.NEW and fused.person_code is None


def test_unknown_context_never_changes_visual_result():
    for build in VISUAL.values():
        visual = build()
        fused = fuse(visual, CONTEXT[K]())
        assert fused.status is visual.status and fused.person_code == visual.person_code
        assert not fused.changed_by_context


def test_context_can_only_remove_a_match_never_add_one():
    for v in VISUAL.values():
        for c in list(CONTEXT.values()) + [lambda: None]:
            visual = v()
            fused = fuse(visual, c())
            if visual.status is not DecisionStatus.PROBABLE_MATCH:
                assert fused.status is visual.status and fused.person_code is None
            if fused.person_code is not None:
                assert fused.person_code == visual.person_code


# --------------------------------------------------------------------------- evidence retention / immutability


def test_evidence_retained_by_reference():
    visual, context = VISUAL[M](), CONTEXT[X]()
    fused = fuse(visual, context)
    assert fused.visual is visual and fused.context is context
    assert fused.visual.best_similarity == 0.9 and fused.visual.second_best_similarity == 0.5
    assert fused.visual.reason.value == "clear_match"
    assert fused.context.reason.value == "faster_than_minimum"
    assert fused.context.time_gap_seconds == 1.0


def test_inputs_unchanged():
    visual, context = VISUAL[M](), CONTEXT[X]()
    visual_copy, context_copy = copy.deepcopy(visual), copy.deepcopy(context)
    fuse(visual, context)
    assert visual == visual_copy and context == context_copy


def test_scores_not_modified():
    visual = decide(ranked(0.912345678, 0.1), POLICY)
    for build in CONTEXT.values():
        fused = fuse(visual, build())
        assert fused.visual.best_similarity == 0.912345678
        assert fused.visual.best_candidate == Candidate("P001", 0.912345678)
        assert fused.visual.second_best_similarity == 0.1


def test_result_has_no_probability_confidence_or_score_field():
    names = {f.name for f in dataclasses.fields(FusedDecision)}
    assert not any(k in n for n in names for k in ("prob", "confidence", "score", "weight", "similarity"))


def test_result_is_immutable():
    fused = fuse(VISUAL[M](), CONTEXT[P]())
    with pytest.raises(dataclasses.FrozenInstanceError):
        fused.status = DecisionStatus.NEW


def test_deterministic():
    for v in VISUAL.values():
        for c in CONTEXT.values():
            assert fuse(v(), c()) == fuse(v(), c())


# --------------------------------------------------------------------------- validation


def test_invalid_inputs_rejected():
    with pytest.raises(TypeError, match="visual"):
        fuse("probable_match", CONTEXT[P]())
    with pytest.raises(TypeError, match="context"):
        fuse(VISUAL[M](), "plausible")
    with pytest.raises(TypeError, match="context"):
        fuse(VISUAL[M](), ContextStatus.PLAUSIBLE)


# --------------------------------------------------------------------------- registry side effects


def test_no_registry_mutation_end_to_end():
    reg = IdentityRegistry()
    reg.create_identity(np.array([1.0, 0.0, 0.0]))
    reg.create_identity(np.array([0.0, 1.0, 0.0]))
    query = np.array([1.0, 0.1, 0.0])
    before = [(c, reg.get_identity(c).tracklet_count, reg.get_identity(c).embedding.copy()) for c in reg.person_codes]
    ranking_before = reg.rank_candidates(query)

    visual = decide_for_embedding(reg, query, POLICY)
    assert visual.status is DecisionStatus.PROBABLE_MATCH
    for build in CONTEXT.values():
        fuse(visual, build())
    vetoed = fuse(visual, CONTEXT[X]())
    assert vetoed.status is DecisionStatus.UNCERTAIN

    assert reg.person_codes == ["P001", "P002"]
    for code, count, emb in before:
        assert reg.get_identity(code).tracklet_count == count
        np.testing.assert_array_equal(reg.get_identity(code).embedding, emb)
    assert reg.rank_candidates(query) == ranking_before
