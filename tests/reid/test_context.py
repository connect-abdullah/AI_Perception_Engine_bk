"""Context plausibility tests. Camera names, timestamps and travel times are synthetic."""

import dataclasses
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from reid.context import (
    ContextEvidence,
    ContextReason,
    ContextStatus,
    ObservationContext,
    TransitionRule,
    TransitionRules,
    time_gap_seconds,
)
from reid.decision import DecisionPolicy, decide_for_embedding
from reid.identity import IdentityRegistry

RULES = TransitionRules([
    TransitionRule.window("CAM_A", "CAM_B", min_seconds=5, max_seconds=120),
    TransitionRule.window("CAM_B", "CAM_A", min_seconds=5, max_seconds=120),
    TransitionRule.window("CAM_B", "CAM_C", min_seconds=30, max_seconds=None),  # no upper bound
    TransitionRule.disallowed("CAM_A", "CAM_D"),
])


def obs(camera, t, room=None, floor=None):
    return ObservationContext(camera, t, room=room, floor=floor)


def evaluate(src_cam, dst_cam, gap, rules=RULES):
    return rules.evaluate(obs(src_cam, 1000.0), obs(dst_cam, 1000.0 + gap))


# --------------------------------------------------------------------------- core cases


def test_same_camera_forward_time_is_plausible():
    e = evaluate("CAM_A", "CAM_A", 3.0)
    assert e.status is ContextStatus.PLAUSIBLE and e.reason is ContextReason.SAME_CAMERA
    assert e.rule is None and e.time_gap_seconds == 3.0


def test_same_camera_zero_gap_is_plausible():
    assert evaluate("CAM_A", "CAM_A", 0.0).status is ContextStatus.PLAUSIBLE


def test_explicit_same_camera_rule_takes_precedence():
    rules = TransitionRules([TransitionRule.window("CAM_A", "CAM_A", min_seconds=10, max_seconds=None)])
    e = evaluate("CAM_A", "CAM_A", 3.0, rules)
    assert e.status is ContextStatus.IMPOSSIBLE and e.reason is ContextReason.FASTER_THAN_MINIMUM


def test_known_transition_within_window_is_plausible():
    e = evaluate("CAM_A", "CAM_B", 40.0)
    assert e.status is ContextStatus.PLAUSIBLE and e.reason is ContextReason.WITHIN_TRANSITION_WINDOW
    assert e.rule == TransitionRule.window("CAM_A", "CAM_B", 5, 120)


def test_too_fast_transition_is_impossible():
    e = evaluate("CAM_A", "CAM_B", 2.0)
    assert e.status is ContextStatus.IMPOSSIBLE and e.reason is ContextReason.FASTER_THAN_MINIMUM


def test_slower_than_maximum_is_unknown_not_impossible():
    e = evaluate("CAM_A", "CAM_B", 600.0)
    assert e.status is ContextStatus.UNKNOWN and e.reason is ContextReason.SLOWER_THAN_MAXIMUM


def test_rule_without_maximum_never_times_out():
    assert evaluate("CAM_B", "CAM_C", 10_000.0).status is ContextStatus.PLAUSIBLE


def test_rule_without_minimum():
    rules = TransitionRules([TransitionRule.window("CAM_A", "CAM_B", min_seconds=None, max_seconds=60)])
    assert evaluate("CAM_A", "CAM_B", 0.0, rules).status is ContextStatus.PLAUSIBLE
    assert evaluate("CAM_A", "CAM_B", 61.0, rules).status is ContextStatus.UNKNOWN


def test_disallowed_transition_is_impossible_at_any_gap():
    for gap in (0.0, 30.0, 1e6):
        e = evaluate("CAM_A", "CAM_D", gap)
        assert e.status is ContextStatus.IMPOSSIBLE and e.reason is ContextReason.TRANSITION_DISALLOWED


def test_no_rule_is_unknown():
    e = evaluate("CAM_C", "CAM_A", 50.0)
    assert e.status is ContextStatus.UNKNOWN and e.reason is ContextReason.NO_TRANSITION_RULE
    assert e.rule is None


def test_rules_are_directional():
    # CAM_A -> CAM_D is disallowed, but nothing is said about CAM_D -> CAM_A.
    assert evaluate("CAM_D", "CAM_A", 50.0).status is ContextStatus.UNKNOWN
    # CAM_B -> CAM_C exists, CAM_C -> CAM_B does not.
    assert evaluate("CAM_C", "CAM_B", 50.0).reason is ContextReason.NO_TRANSITION_RULE


def test_empty_rule_set():
    rules = TransitionRules()
    assert len(rules) == 0
    assert evaluate("CAM_A", "CAM_B", 10.0, rules).status is ContextStatus.UNKNOWN
    assert evaluate("CAM_A", "CAM_A", 10.0, rules).status is ContextStatus.PLAUSIBLE


# --------------------------------------------------------------------------- boundaries


@pytest.mark.parametrize(
    "gap, expected",
    [
        (5.0, ContextStatus.PLAUSIBLE),  # exactly min -> allowed (inclusive)
        (float(np.nextafter(5.0, 0)), ContextStatus.IMPOSSIBLE),  # just under min
        (120.0, ContextStatus.PLAUSIBLE),  # exactly max -> allowed (inclusive)
        (float(np.nextafter(120.0, 200)), ContextStatus.UNKNOWN),  # just over max
    ],
)
def test_exact_boundaries(gap, expected):
    rules = TransitionRules([TransitionRule.window("CAM_A", "CAM_B", 5.0, 120.0)])
    e = rules.evaluate(obs("CAM_A", 0.0), obs("CAM_B", gap))
    assert e.status is expected


def test_zero_width_window():
    rules = TransitionRules([TransitionRule.window("CAM_A", "CAM_B", 10.0, 10.0)])
    assert rules.evaluate(obs("CAM_A", 0.0), obs("CAM_B", 10.0)).status is ContextStatus.PLAUSIBLE


# --------------------------------------------------------------------------- time handling


def test_negative_time_gap_rejected():
    with pytest.raises(ValueError, match="time order"):
        RULES.evaluate(obs("CAM_A", 100.0), obs("CAM_B", 99.0))
    with pytest.raises(ValueError, match="time order"):
        RULES.evaluate(obs("CAM_A", 100.0), obs("CAM_A", 99.9))  # same camera too


def test_datetime_timestamps():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    e = RULES.evaluate(obs("CAM_A", t0), obs("CAM_B", t0 + timedelta(seconds=30)))
    assert e.time_gap_seconds == 30.0 and e.status is ContextStatus.PLAUSIBLE
    with pytest.raises(ValueError, match="time order"):
        RULES.evaluate(obs("CAM_A", t0), obs("CAM_B", t0 - timedelta(seconds=1)))


def test_datetimes_across_timezones():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 17, 0, 10, tzinfo=timezone(timedelta(hours=5)))  # 10 s later
    assert time_gap_seconds(obs("CAM_A", t0), obs("CAM_A", t1)) == 10.0


def test_mixed_timestamp_kinds_rejected():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    with pytest.raises(TypeError, match="both"):
        RULES.evaluate(obs("CAM_A", t0), obs("CAM_B", 100.0))
    with pytest.raises(TypeError, match="naive"):
        RULES.evaluate(obs("CAM_A", t0), obs("CAM_B", t0.replace(tzinfo=timezone.utc)))


# --------------------------------------------------------------------------- room / floor


def test_room_and_floor_carried_into_evidence():
    e = RULES.evaluate(
        obs("CAM_A", 0.0, room="lobby", floor="1"),
        obs("CAM_B", 30.0, room="corridor", floor="2"),
    )
    assert (e.source_room, e.source_floor) == ("lobby", "1")
    assert (e.destination_room, e.destination_floor) == ("corridor", "2")


def test_missing_room_and_floor_are_fine():
    e = RULES.evaluate(obs("CAM_A", 0.0), obs("CAM_B", 30.0, room="corridor"))
    assert e.status is ContextStatus.PLAUSIBLE
    assert e.source_room is None and e.source_floor is None and e.destination_floor is None


def test_same_camera_in_two_rooms_rejected():
    with pytest.raises(ValueError, match="two different rooms"):
        RULES.evaluate(obs("CAM_A", 0.0, room="lobby"), obs("CAM_A", 5.0, room="kitchen"))
    with pytest.raises(ValueError, match="two different rooms"):
        RULES.evaluate(obs("CAM_A", 0.0, floor="1"), obs("CAM_A", 5.0, floor="2"))
    # one side missing is not a conflict
    assert RULES.evaluate(obs("CAM_A", 0.0, room="lobby"), obs("CAM_A", 5.0)).status is ContextStatus.PLAUSIBLE


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(camera_id="", timestamp=0.0), "camera_id"),
        (dict(camera_id=None, timestamp=0.0), "camera_id"),
        (dict(camera_id="CAM_A", timestamp=float("nan")), "timestamp"),
        (dict(camera_id="CAM_A", timestamp="noon"), "timestamp"),
        (dict(camera_id="CAM_A", timestamp=True), "timestamp"),
        (dict(camera_id="CAM_A", timestamp=0.0, room=""), "room"),
        (dict(camera_id="CAM_A", timestamp=0.0, floor=2), "floor"),
    ],
)
def test_invalid_observation_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ObservationContext(**kwargs)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: TransitionRule.window("CAM_A", "CAM_B", 10, 5), "below min_seconds"),
        (lambda: TransitionRule.window("CAM_A", "CAM_B", -1, 5), "min_seconds"),
        (lambda: TransitionRule.window("CAM_A", "CAM_B", 1, float("inf")), "max_seconds"),
        (lambda: TransitionRule.window("CAM_A", "CAM_B", None, None), "needs min_seconds"),
        (lambda: TransitionRule("CAM_A", "CAM_B", False, 5, None), "cannot have travel times"),
        (lambda: TransitionRule("CAM_A", "CAM_B", 1, 5, None), "allowed must be a bool"),
        (lambda: TransitionRule.window("", "CAM_B", 1, 5), "source_camera"),
    ],
)
def test_invalid_rule_rejected(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_duplicate_rule_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        TransitionRules([TransitionRule.window("CAM_A", "CAM_B", 1, 5), TransitionRule.disallowed("CAM_A", "CAM_B")])


def test_non_rule_rejected():
    with pytest.raises(TypeError):
        TransitionRules([("CAM_A", "CAM_B", 1, 5)])


# --------------------------------------------------------------------------- semantics / determinism


def test_evidence_has_no_probability_or_score_field():
    names = {f.name for f in dataclasses.fields(ContextEvidence)}
    assert not any(k in n for n in names for k in ("prob", "confidence", "score", "weight"))


def test_deterministic():
    a = evaluate("CAM_A", "CAM_B", 40.0)
    b = evaluate("CAM_A", "CAM_B", 40.0)
    assert a == b


def test_inputs_and_rules_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs("CAM_A", 0.0).camera_id = "CAM_B"
    with pytest.raises(dataclasses.FrozenInstanceError):
        TransitionRule.window("CAM_A", "CAM_B", 1, 5).min_seconds = 0


# --------------------------------------------------------------------------- no side effects on identity layers


def test_context_module_is_independent_of_identity_and_similarity():
    import reid.context as context

    source = Path(context.__file__).read_text(encoding="utf-8")
    for module in ("reid.identity", "reid.decision", "reid.similarity", "reid.embedder", "torch", "numpy"):
        assert f"import {module}" not in source and f"from {module}" not in source
    assert not any(m in sys.modules for m in ("app", "fastapi", "sqlalchemy", "psycopg"))


def test_no_registry_mutation_and_no_decision_or_similarity_change():
    reg = IdentityRegistry()
    reg.create_identity(np.array([1.0, 0.0, 0.0]))
    reg.create_identity(np.array([0.0, 1.0, 0.0]))
    query = np.array([1.0, 0.2, 0.0])
    policy = DecisionPolicy(accept_at=0.7, new_below=0.4, min_margin=0.1)  # synthetic values

    candidates_before = reg.rank_candidates(query)
    decision_before = decide_for_embedding(reg, query, policy)
    embeddings_before = [reg.get_identity(c).embedding.copy() for c in reg.person_codes]

    # Every context outcome, including IMPOSSIBLE for the best candidate's camera move.
    for src, dst, gap in [("CAM_A", "CAM_B", 2.0), ("CAM_A", "CAM_D", 30.0), ("CAM_A", "CAM_B", 40.0), ("CAM_C", "CAM_A", 9.0)]:
        evaluate(src, dst, gap)

    assert reg.person_codes == ["P001", "P002"]
    assert [reg.get_identity(c).tracklet_count for c in reg.person_codes] == [1, 1]
    for code, before in zip(reg.person_codes, embeddings_before):
        np.testing.assert_array_equal(reg.get_identity(code).embedding, before)
    assert reg.rank_candidates(query) == candidates_before
    assert decide_for_embedding(reg, query, policy) == decision_before
