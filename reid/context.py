"""Camera / time plausibility of one person moving between two observations.

Given where and when a person was last seen (source) and where and when a candidate
track starts (destination), report whether the move is PLAUSIBLE, IMPOSSIBLE or
UNKNOWN under caller-supplied camera transition rules.

This is contextual evidence only. It does not look at appearance, does not change
similarity scores, does not decide identities and does not touch the identity registry.
All travel times come from the caller; none are built in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

Timestamp = float | datetime  # float = seconds on any consistent clock; datetime = wall clock


class ContextStatus(str, Enum):
    PLAUSIBLE = "plausible"
    IMPOSSIBLE = "impossible"  # an explicit rule rules the move out
    UNKNOWN = "unknown"  # no rule covers it, or the rules cannot judge it


class ContextReason(str, Enum):
    SAME_CAMERA = "same_camera"  # no explicit rule; same camera, time moving forward
    WITHIN_TRANSITION_WINDOW = "within_transition_window"
    FASTER_THAN_MINIMUM = "faster_than_minimum"  # gap < rule.min_seconds
    TRANSITION_DISALLOWED = "transition_disallowed"  # rule forbids a direct move
    SLOWER_THAN_MAXIMUM = "slower_than_maximum"  # gap > rule.max_seconds
    NO_TRANSITION_RULE = "no_transition_rule"


def _check_name(value, field: str, optional: bool) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")


def _check_seconds(value, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite number >= 0 or None, got {value!r}")


@dataclass(frozen=True)
class ObservationContext:
    """Where and when a person was observed.

    For a track, use its last frame as a source and its first frame as a destination.
    Room and floor are optional labels (strings, e.g. floor="2") carried into the evidence;
    the only check made on them is that one camera is not reported in two places.
    """

    camera_id: str
    timestamp: Timestamp
    room: str | None = None
    floor: str | None = None

    def __post_init__(self):
        _check_name(self.camera_id, "camera_id", optional=False)
        _check_name(self.room, "room", optional=True)
        _check_name(self.floor, "floor", optional=True)
        ts = self.timestamp
        if isinstance(ts, datetime):
            return
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError(f"timestamp must be a finite number of seconds or a datetime, got {ts!r}")


@dataclass(frozen=True)
class TransitionRule:
    """Directional rule for moving from `source_camera` to `destination_camera`.

    Build with `TransitionRule.window(...)` or `TransitionRule.disallowed(...)`.
    """

    source_camera: str
    destination_camera: str
    allowed: bool
    min_seconds: float | None
    max_seconds: float | None

    def __post_init__(self):
        _check_name(self.source_camera, "source_camera", optional=False)
        _check_name(self.destination_camera, "destination_camera", optional=False)
        if not isinstance(self.allowed, bool):
            raise ValueError(f"allowed must be a bool, got {self.allowed!r}")
        _check_seconds(self.min_seconds, "min_seconds")
        _check_seconds(self.max_seconds, "max_seconds")
        if not self.allowed and (self.min_seconds is not None or self.max_seconds is not None):
            raise ValueError("A disallowed transition cannot have travel times")
        if self.allowed and self.min_seconds is None and self.max_seconds is None:
            raise ValueError("An allowed transition needs min_seconds and/or max_seconds")
        if self.min_seconds is not None and self.max_seconds is not None and self.max_seconds < self.min_seconds:
            raise ValueError(f"max_seconds ({self.max_seconds}) is below min_seconds ({self.min_seconds})")

    @classmethod
    def window(cls, source_camera: str, destination_camera: str,
               min_seconds: float | None, max_seconds: float | None) -> TransitionRule:
        """Allowed move taking at least `min_seconds` and normally at most `max_seconds`."""
        return cls(source_camera, destination_camera, True, min_seconds, max_seconds)

    @classmethod
    def disallowed(cls, source_camera: str, destination_camera: str) -> TransitionRule:
        """A direct move between these cameras cannot happen."""
        return cls(source_camera, destination_camera, False, None, None)


@dataclass(frozen=True)
class ContextEvidence:
    status: ContextStatus
    reason: ContextReason
    time_gap_seconds: float
    source_camera: str
    destination_camera: str
    source_room: str | None
    source_floor: str | None
    destination_room: str | None
    destination_floor: str | None
    rule: TransitionRule | None  # the rule applied, if any


def _conflicting(a: str | None, b: str | None) -> bool:
    """Both known and different. Missing metadata never conflicts."""
    return a is not None and b is not None and a != b


def time_gap_seconds(source: ObservationContext, destination: ObservationContext) -> float:
    """Seconds from source to destination. Raises if the clocks are incompatible or time runs backwards."""
    a, b = source.timestamp, destination.timestamp
    if isinstance(a, datetime) != isinstance(b, datetime):
        raise TypeError("Source and destination timestamps must both be datetimes or both be seconds")
    if isinstance(a, datetime):
        if (a.tzinfo is None) != (b.tzinfo is None):
            raise TypeError("Cannot compare a timezone-aware datetime with a naive one")
        gap = (b - a).total_seconds()
    else:
        gap = float(b - a)
    if gap < 0:
        raise ValueError(
            f"Destination is {-gap:g} s before source; pass observations in time order"
        )
    return gap


class TransitionRules:
    """A set of directional camera transition rules (at most one per camera pair)."""

    def __init__(self, rules: Iterable[TransitionRule] = ()):
        self._rules: dict[tuple[str, str], TransitionRule] = {}
        for rule in rules:
            if not isinstance(rule, TransitionRule):
                raise TypeError(f"Expected TransitionRule, got {type(rule).__name__}")
            key = (rule.source_camera, rule.destination_camera)
            if key in self._rules:
                raise ValueError(f"Duplicate rule for {key[0]} -> {key[1]}")
            self._rules[key] = rule

    def __len__(self) -> int:
        return len(self._rules)

    def rule_for(self, source_camera: str, destination_camera: str) -> TransitionRule | None:
        return self._rules.get((source_camera, destination_camera))

    def evaluate(self, source: ObservationContext, destination: ObservationContext) -> ContextEvidence:
        """Judge the move from `source` (earlier) to `destination` (later).

        Rules, in order:
          explicit rule, disallowed                -> IMPOSSIBLE (transition_disallowed)
          explicit rule, gap < min_seconds         -> IMPOSSIBLE (faster_than_minimum)
          explicit rule, gap > max_seconds         -> UNKNOWN    (slower_than_maximum)
          explicit rule, otherwise                 -> PLAUSIBLE  (within_transition_window)
          no rule, same camera                     -> PLAUSIBLE  (same_camera)
          no rule, different cameras               -> UNKNOWN    (no_transition_rule)
        Both window ends are inclusive.
        """
        if source.camera_id == destination.camera_id and (
            _conflicting(source.room, destination.room) or _conflicting(source.floor, destination.floor)
        ):
            raise ValueError(f"Camera {source.camera_id} is reported in two different rooms/floors")

        gap = time_gap_seconds(source, destination)
        rule = self.rule_for(source.camera_id, destination.camera_id)

        if rule is None:
            if source.camera_id == destination.camera_id:
                status, reason = ContextStatus.PLAUSIBLE, ContextReason.SAME_CAMERA
            else:
                status, reason = ContextStatus.UNKNOWN, ContextReason.NO_TRANSITION_RULE
        elif not rule.allowed:
            status, reason = ContextStatus.IMPOSSIBLE, ContextReason.TRANSITION_DISALLOWED
        elif rule.min_seconds is not None and gap < rule.min_seconds:
            status, reason = ContextStatus.IMPOSSIBLE, ContextReason.FASTER_THAN_MINIMUM
        elif rule.max_seconds is not None and gap > rule.max_seconds:
            # A person can always take longer than expected (detours, waiting), so a long gap
            # is not proof of a different person; it only means the rule cannot vouch for it.
            status, reason = ContextStatus.UNKNOWN, ContextReason.SLOWER_THAN_MAXIMUM
        else:
            status, reason = ContextStatus.PLAUSIBLE, ContextReason.WITHIN_TRANSITION_WINDOW

        return ContextEvidence(
            status=status,
            reason=reason,
            time_gap_seconds=gap,
            source_camera=source.camera_id,
            destination_camera=destination.camera_id,
            source_room=source.room,
            source_floor=source.floor,
            destination_room=destination.room,
            destination_floor=destination.floor,
            rule=rule,
        )
