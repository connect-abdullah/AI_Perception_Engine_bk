"""Decision policy tests.

All policy values here are synthetic, chosen only to exercise the rules.
They are not calibrated and must not be copied into real use.
"""

import dataclasses

import numpy as np
import pytest

from reid.decision import (
    DecisionPolicy,
    DecisionReason,
    DecisionStatus,
    IdentityDecision,
    decide,
    decide_for_embedding,
)
from reid.identity import Candidate, IdentityRegistry

POLICY = DecisionPolicy(accept_at=0.7, new_below=0.4, min_margin=0.1)


def ranked(*scores):
    return [Candidate(f"P{k + 1:03d}", s) for k, s in enumerate(scores)]


def snapshot(registry):
    return [
        (code, registry.get_identity(code).tracklet_count, registry.get_identity(code).embedding.copy())
        for code in registry.person_codes
    ]


def assert_registry_unchanged(registry, before):
    after = snapshot(registry)
    assert [(c, n) for c, n, _ in after] == [(c, n) for c, n, _ in before]
    for (_, _, a), (_, _, b) in zip(after, before):
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- the five cases


def test_case_a_strong_separated_match():
    d = decide(ranked(0.9, 0.5, 0.2), POLICY)
    assert d.status is DecisionStatus.PROBABLE_MATCH
    assert d.reason is DecisionReason.CLEAR_MATCH
    assert d.person_code == "P001"
    assert d.best_candidate == Candidate("P001", 0.9)
    assert d.best_similarity == 0.9 and d.second_best_similarity == 0.5
    assert d.margin == pytest.approx(0.4)
    assert d.policy is POLICY


def test_case_b_all_weak_is_new():
    d = decide(ranked(0.35, 0.3, 0.1), POLICY)
    assert d.status is DecisionStatus.NEW
    assert d.reason is DecisionReason.ALL_CANDIDATES_WEAK
    assert d.person_code is None
    assert d.best_similarity == 0.35  # evidence is still reported


def test_case_c_ambiguous_top_two_is_uncertain():
    d = decide(ranked(0.85, 0.8), POLICY)
    assert d.status is DecisionStatus.UNCERTAIN
    assert d.reason is DecisionReason.AMBIGUOUS_TOP_CANDIDATES
    assert d.person_code is None


def test_case_d_between_thresholds_is_uncertain():
    d = decide(ranked(0.55, 0.1), POLICY)
    assert d.status is DecisionStatus.UNCERTAIN
    assert d.reason is DecisionReason.BETWEEN_THRESHOLDS
    assert d.person_code is None


def test_case_e_empty_is_new():
    d = decide([], POLICY)
    assert d.status is DecisionStatus.NEW
    assert d.reason is DecisionReason.NO_KNOWN_IDENTITIES
    assert d.person_code is None and d.best_candidate is None
    assert d.best_similarity is None and d.second_best_similarity is None and d.margin is None


def test_grey_region_with_close_second_reports_between_thresholds():
    # Both conditions fail; the weaker-evidence reason takes precedence.
    assert decide(ranked(0.6, 0.59), POLICY).reason is DecisionReason.BETWEEN_THRESHOLDS


def test_weak_but_close_candidates_are_still_new():
    # Ambiguity among weak candidates does not block NEW: none of them is a plausible match.
    d = decide(ranked(0.3, 0.3, 0.29), POLICY)
    assert d.status is DecisionStatus.NEW


# --------------------------------------------------------------------------- single candidate


def test_single_strong_candidate_is_match_without_margin():
    d = decide(ranked(0.8), POLICY)
    assert d.status is DecisionStatus.PROBABLE_MATCH and d.person_code == "P001"
    assert d.second_best_similarity is None and d.margin is None


def test_single_weak_candidate_is_new():
    assert decide(ranked(0.2), POLICY).status is DecisionStatus.NEW


def test_single_middle_candidate_is_uncertain():
    assert decide(ranked(0.5), POLICY).status is DecisionStatus.UNCERTAIN


# --------------------------------------------------------------------------- boundaries


@pytest.mark.parametrize(
    "best, expected",
    [
        (0.7, DecisionStatus.PROBABLE_MATCH),  # exactly accept_at -> accepted (>=)
        (np.nextafter(0.7, 0), DecisionStatus.UNCERTAIN),  # just below accept_at
        (0.4, DecisionStatus.UNCERTAIN),  # exactly new_below -> not new (< required)
        (np.nextafter(0.4, 0), DecisionStatus.NEW),  # just below new_below
    ],
)
def test_threshold_boundaries(best, expected):
    assert decide(ranked(float(best)), POLICY).status is expected


@pytest.mark.parametrize("best, second", [(0.8, 0.7), (0.9, 0.8), (1.0, 0.9), (0.75, 0.65)])
def test_margin_exactly_equal_to_min_margin_is_accepted(best, second):
    # Includes float pairs whose subtraction is slightly below 0.1 (e.g. 0.9 - 0.8).
    assert decide(ranked(best, second), POLICY).status is DecisionStatus.PROBABLE_MATCH


def test_margin_just_below_min_margin_is_uncertain():
    assert decide(ranked(0.8, 0.7001), POLICY).status is DecisionStatus.UNCERTAIN


def test_equal_thresholds_leave_no_grey_region():
    policy = DecisionPolicy(accept_at=0.5, new_below=0.5, min_margin=0.0)
    assert decide(ranked(0.5), policy).status is DecisionStatus.PROBABLE_MATCH
    assert decide(ranked(float(np.nextafter(0.5, 0))), policy).status is DecisionStatus.NEW


def test_zero_margin_disables_ambiguity_check_but_exact_tie_still_matches_first_ranked():
    policy = DecisionPolicy(accept_at=0.7, new_below=0.4, min_margin=0.0)
    d = decide(ranked(0.8, 0.8), policy)
    assert d.status is DecisionStatus.PROBABLE_MATCH and d.person_code == "P001"


# --------------------------------------------------------------------------- policy validation


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(accept_at=0.4, new_below=0.7, min_margin=0.1), "must not exceed"),
        (dict(accept_at=1.5, new_below=0.4, min_margin=0.1), "accept_at"),
        (dict(accept_at=0.7, new_below=-1.5, min_margin=0.1), "new_below"),
        (dict(accept_at=0.7, new_below=0.4, min_margin=-0.1), "min_margin"),
        (dict(accept_at=0.7, new_below=0.4, min_margin=2.5), "min_margin"),
        (dict(accept_at=float("nan"), new_below=0.4, min_margin=0.1), "finite"),
        (dict(accept_at=0.7, new_below=float("inf"), min_margin=0.1), "finite"),
        (dict(accept_at=0.7, new_below=0.4, min_margin=None), "finite"),
        (dict(accept_at="0.7", new_below=0.4, min_margin=0.1), "finite"),
        (dict(accept_at=True, new_below=0.4, min_margin=0.1), "finite"),
    ],
)
def test_invalid_policy_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DecisionPolicy(**kwargs)


def test_policy_has_no_defaults():
    with pytest.raises(TypeError):
        DecisionPolicy()
    with pytest.raises(TypeError):
        DecisionPolicy(accept_at=0.7, new_below=0.4)
    assert all(f.default is dataclasses.MISSING for f in dataclasses.fields(DecisionPolicy))


def test_policy_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        POLICY.accept_at = 0.1


def test_policy_required():
    with pytest.raises(TypeError):
        decide(ranked(0.9), {"accept_at": 0.7, "new_below": 0.4, "min_margin": 0.1})


# --------------------------------------------------------------------------- candidate validation


@pytest.mark.parametrize(
    "candidates, message",
    [
        (ranked(0.5, 0.9), "ranked"),
        (ranked(float("nan")), "invalid cosine"),
        (ranked(1.5), "invalid cosine"),
        ([Candidate("P001", 0.9), Candidate("P001", 0.5)], "Duplicate"),
    ],
)
def test_invalid_candidates_rejected(candidates, message):
    with pytest.raises(ValueError, match=message):
        decide(candidates, POLICY)


def test_non_candidate_rejected():
    with pytest.raises(TypeError):
        decide([("P001", 0.9)], POLICY)


# --------------------------------------------------------------------------- score semantics


def test_scores_are_raw_cosine_not_probabilities():
    # Negative cosine is valid evidence and is passed through unchanged, not clipped to [0, 1].
    d = decide(ranked(-0.2, -0.6), DecisionPolicy(accept_at=0.7, new_below=0.0, min_margin=0.1))
    assert d.status is DecisionStatus.NEW
    assert d.best_similarity == -0.2 and d.second_best_similarity == -0.6
    assert d.margin == pytest.approx(0.4)
    # Negative thresholds are legal in cosine units; probabilities could not be negative.
    DecisionPolicy(accept_at=-0.1, new_below=-0.5, min_margin=0.0)


def test_result_exposes_no_probability_or_confidence_field():
    names = {f.name for f in dataclasses.fields(IdentityDecision)}
    assert not any("prob" in n or "confidence" in n for n in names)


def test_best_candidate_score_passed_through_unchanged():
    d = decide(ranked(0.912345678, 0.1), POLICY)
    assert d.best_similarity == 0.912345678 == d.best_candidate.similarity


# --------------------------------------------------------------------------- determinism


def test_deterministic():
    candidates = ranked(0.9, 0.85, 0.3)
    assert decide(candidates, POLICY) == decide(list(candidates), POLICY)


def test_input_list_not_modified():
    candidates = ranked(0.9, 0.5)
    original = list(candidates)
    decide(candidates, POLICY)
    assert candidates == original


# --------------------------------------------------------------------------- registry side effects


@pytest.fixture
def registry():
    reg = IdentityRegistry()
    reg.create_identity(np.array([1.0, 0.0, 0.0]))  # P001
    reg.create_identity(np.array([0.0, 1.0, 0.0]))  # P002
    reg.create_identity(np.array([0.0, 0.0, 1.0]))  # P003
    return reg


# Query cosines against (P001, P002, P003) with the orthogonal registry above:
MATCH_QUERY = np.array([1.0, 0.0, 0.0])  # (1.0, 0, 0): clear match to P001
NEW_QUERY = np.array([-1.0, -1.0, -1.0])  # all -0.577: every candidate weak
AMBIGUOUS_QUERY = np.array([1.0, 0.95, 0.0])  # (0.725, 0.689, 0): above accept_at, margin 0.036
GREY_QUERY = np.array([1.0, 1.0, 1.0])  # all 0.577: between new_below and accept_at


def test_decide_for_embedding_match(registry):
    d = decide_for_embedding(registry, MATCH_QUERY, POLICY)
    assert d.status is DecisionStatus.PROBABLE_MATCH and d.person_code == "P001"


@pytest.mark.parametrize(
    "query, expected",
    [
        (MATCH_QUERY, DecisionStatus.PROBABLE_MATCH),
        (NEW_QUERY, DecisionStatus.NEW),
        (AMBIGUOUS_QUERY, DecisionStatus.UNCERTAIN),
        (GREY_QUERY, DecisionStatus.UNCERTAIN),
    ],
)
def test_no_registry_mutation_for_any_outcome(registry, query, expected):
    before = snapshot(registry)
    for _ in range(3):
        assert decide_for_embedding(registry, query, POLICY).status is expected
    assert_registry_unchanged(registry, before)


def test_new_does_not_create_identity(registry):
    d = decide_for_embedding(registry, NEW_QUERY, POLICY)
    assert d.status is DecisionStatus.NEW and d.person_code is None
    assert len(registry) == 3 and registry.person_codes == ["P001", "P002", "P003"]


def test_new_on_empty_registry_does_not_create_identity():
    reg = IdentityRegistry()
    d = decide_for_embedding(reg, np.ones(8), POLICY)
    assert d.status is DecisionStatus.NEW and d.reason is DecisionReason.NO_KNOWN_IDENTITIES
    assert len(reg) == 0
    # the caller creates explicitly, and only then does an identity exist
    assert reg.create_identity(np.ones(8)) == "P001"


def test_uncertain_never_assigns_or_creates(registry):
    for query, reason in (
        (AMBIGUOUS_QUERY, DecisionReason.AMBIGUOUS_TOP_CANDIDATES),
        (GREY_QUERY, DecisionReason.BETWEEN_THRESHOLDS),
    ):
        before = snapshot(registry)
        d = decide_for_embedding(registry, query, POLICY)
        assert d.status is DecisionStatus.UNCERTAIN and d.reason is reason
        assert d.person_code is None
        assert_registry_unchanged(registry, before)


def test_only_probable_match_carries_a_person_code():
    for scores in [(0.9, 0.5), (0.35,), (0.85, 0.8), (0.55,), ()]:
        d = decide(ranked(*scores), POLICY)
        assert (d.person_code is not None) == (d.status is DecisionStatus.PROBABLE_MATCH)
