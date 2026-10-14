"""Tracker tests. All detections are synthetic; no detector model, weights or network is used.

Several tests pin supervision 0.30.9 ByteTrack behaviour that the tracking contract relies on.
"""

import copy
import dataclasses
import re
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception_minimal.detector import PersonDetection
from perception_minimal.tracker import (
    BYTETRACK_SETTINGS,
    ByteTrackPersonTracker,
    PersonTracker,
    TrackedDetection,
    check_frame_rate,
)
from perception_minimal.video import VideoReader

REPO_ROOT = Path(__file__).resolve().parents[2]
FPS = 10.0


def person(x, y=100, w=40, h=100, score=0.9):
    return PersonDetection((x, y, w, h), score)


def walk(start_x, n, step=4, **kw):
    """One person moving right by `step` px per frame."""
    return [person(start_x + step * t, **kw) for t in range(n)]


def run(frames, frame_rate=FPS):
    tracker = ByteTrackPersonTracker(frame_rate)
    return [tracker.update(dets) for dets in frames]


def ids(frame_output):
    return [t.local_track_id for t in frame_output]


# --------------------------------------------------------------------------- identity of local tracks


def test_one_moving_person_keeps_one_id():
    frames = [[d] for d in walk(50, 30)]
    out = run(frames)
    assert all(ids(o) == ["1"] for o in out)


def test_two_people_keep_distinct_stable_ids():
    a, b = walk(20, 25, step=3), walk(400, 25, step=-3, y=150)
    out = run([[da, db] for da, db in zip(a, b)])
    assert all(ids(o) == ["1", "2"] for o in out)
    by_id = {tid: [] for tid in ("1", "2")}
    for o in out:
        for t in o:
            by_id[t.local_track_id].append(t.bbox)
    assert by_id["1"] == [d.bbox for d in a]
    assert by_id["2"] == [d.bbox for d in b]


def test_new_person_gets_new_id_and_is_emitted_from_second_frame():
    # Pins ByteTrack: after frame 1, a new track is confirmed (and emitted) on its second matched frame.
    a, b = walk(20, 12), walk(400, 12, y=200)
    frames = [[a[t]] + ([b[t]] if t >= 5 else []) for t in range(12)]
    out = run(frames)
    assert [ids(o) for o in out[:6]] == [["1"]] * 6  # frame 5 is b's first sighting: not emitted
    assert all(ids(o) == ["1", "2"] for o in out[6:])
    assert [t.bbox for t in out[6] if t.local_track_id == "2"] == [b[6].bbox]


def test_people_present_on_first_frame_emitted_immediately():
    out = run([[person(20), person(300)]])
    assert ids(out[0]) == ["1", "2"]


def test_ids_are_numeric_strings_sorted_numerically():
    people = [person(10 + 55 * k, w=40) for k in range(11)]  # 11 people -> ids "1".."11"
    out = run([people, [person(d.bbox[0] + 2) for d in people]])
    for o in out:
        assert ids(o) == [str(k) for k in range(1, 12)]  # "10", "11" after "9", not after "1"
        assert all(isinstance(t.local_track_id, str) and t.local_track_id.isdigit() for t in o)


# --------------------------------------------------------------------------- detector box / score preservation


def test_detector_box_and_score_preserved_exactly_not_kalman_box():
    rng = np.random.default_rng(3)
    dets = [
        PersonDetection((100 + 5 * t + int(rng.integers(-3, 4)), 80 + int(rng.integers(-3, 4)),
                         40 + int(rng.integers(-2, 3)), 100 + int(rng.integers(-2, 3))),
                        float(rng.uniform(0.5, 1.0)))
        for t in range(30)
    ]
    tracker = ByteTrackPersonTracker(FPS)
    differs_from_kalman = 0
    for d in dets:
        (t,) = tracker.update([d])
        assert t.bbox == d.bbox and t.detection_score == d.score
        assert type(t.detection_score) is float and all(type(v) is int for v in t.bbox)
        (internal,) = tracker._tracker.tracked_tracks
        x1, y1, x2, y2 = internal.tlbr
        if not np.allclose([x1, y1, x2 - x1, y2 - y1], d.bbox, atol=0.5):
            differs_from_kalman += 1
    # the tracker's own (smoothed) box really differs, so equality above is not a coincidence
    assert differs_from_kalman > 10


def test_equal_scores_still_linked_to_own_detection():
    a = walk(20, 20, step=5, score=0.8)
    b = walk(300, 20, step=-5, y=180, score=0.8)
    out = run([[db, da] if t % 2 else [da, db] for t, (da, db) in enumerate(zip(a, b))])  # input order varies
    id_a = next(t.local_track_id for t in out[0] if t.bbox == a[0].bbox)
    id_b = next(t.local_track_id for t in out[0] if t.bbox == b[0].bbox)
    assert id_a != id_b
    for t_out, da, db in zip(out, a, b):
        assert {t.local_track_id: t.bbox for t in t_out} == {id_a: da.bbox, id_b: db.bbox}
        assert ids(t_out) == sorted(ids(t_out), key=int)


# --------------------------------------------------------------------------- no fake observations


def test_empty_detections():
    tracker = ByteTrackPersonTracker(FPS)
    assert tracker.update([]) == []
    assert tracker.update(()) == []


def test_missed_frame_produces_no_observation_and_track_resumes():
    a = walk(50, 12)
    frames = [[d] for d in a]
    frames[5] = []  # detector missed the person: tracker only predicts this frame
    frames[6] = []
    out = run(frames)
    assert out[5] == [] and out[6] == []
    assert all(ids(o) == ["1"] for i, o in enumerate(out) if i not in (5, 6))
    assert out[7][0].bbox == a[7].bbox


def test_predicted_track_not_emitted_while_other_person_is():
    a, b = walk(20, 10), walk(400, 10, y=200)
    frames = [[a[t], b[t]] for t in range(10)]
    frames[4] = [b[4]]  # a not detected in frame 4
    out = run(frames)
    assert [(t.local_track_id, t.bbox) for t in out[4]] == [("2", b[4].bbox)]
    assert ids(out[5]) == ["1", "2"]


def test_every_output_box_is_an_input_box_of_that_frame():
    rng = np.random.default_rng(7)
    tracker = ByteTrackPersonTracker(FPS)
    for t in range(40):
        dets = [person(30 + 120 * k + 3 * t, score=float(rng.uniform(0.3, 1.0)))
                for k in range(4) if rng.uniform() > 0.2]
        out = tracker.update(dets)
        inputs = {(d.bbox, d.score) for d in dets}
        assert all((o.bbox, o.detection_score) in inputs for o in out)
        assert len({o.bbox for o in out}) == len(out)  # each detection used at most once


# --------------------------------------------------------------------------- pinned ByteTrack filtering


def test_pinned_score_filtering():
    # score <= 0.1 ignored; new tracks need >= activation + 0.1 = 0.35; 0.1 < score < 0.25 only extends.
    assert run([[person(50, score=0.1)]]) == [[]]
    assert run([[person(50, score=0.34)]]) == [[]]
    assert ids(run([[person(50, score=0.35)]])[0]) == ["1"]
    out = run([[person(50, score=0.9)], [person(54, score=0.2)], [person(58, score=0.05)]])
    assert ids(out[1]) == ["1"] and out[1][0].detection_score == 0.2
    assert out[2] == []


def test_settings_match_library_defaults_and_config():
    tracker = ByteTrackPersonTracker(FPS)
    bt = tracker._tracker
    assert BYTETRACK_SETTINGS == {
        "track_activation_threshold": 0.25,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.8,
        "minimum_consecutive_frames": 1,
    }
    assert bt.track_activation_threshold == 0.25 and bt.minimum_matching_threshold == 0.8
    assert bt.minimum_consecutive_frames == 1
    c = tracker.config
    assert c["frame_rate"] == FPS and c["max_frames_lost"] == 10 and c["new_track_min_score"] == pytest.approx(0.35)


def test_supervision_version_pinned():
    import supervision

    assert supervision.__version__ == "0.30.9"


def test_construction_does_not_leak_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ByteTrackPersonTracker(FPS)
    assert not [w for w in caught if "ByteTrack" in str(w.message)]


# --------------------------------------------------------------------------- frame rate


@pytest.mark.parametrize("bad", [True, False, 0, -1, -0.5, float("nan"), float("inf"), "10", None])
def test_invalid_frame_rate_rejected(bad):
    with pytest.raises(ValueError, match="frame_rate"):
        check_frame_rate(bad)
    with pytest.raises(ValueError, match="frame_rate"):
        ByteTrackPersonTracker(bad)


def test_frame_rate_is_required():
    with pytest.raises(TypeError):
        ByteTrackPersonTracker()


@pytest.mark.parametrize("fps, stride, max_lost", [(30.0, 1, 30), (30.0, 3, 10), (25.0, 2, 12), (12.5, 1, 12)])
def test_effective_frame_rate_sets_lost_buffer(fps, stride, max_lost):
    tracker = ByteTrackPersonTracker(fps / stride)
    assert tracker.frame_rate == fps / stride
    assert tracker.config["max_frames_lost"] == max_lost  # int(frame_rate / 30 * 30)


def test_lost_track_dropped_after_buffer():
    # frame_rate 5 -> track kept for 5 missed updates
    a = walk(50, 20)
    frames = [[d] for d in a]
    for t in range(3, 3 + 7):
        frames[t] = []
    out = run(frames, frame_rate=5.0)
    assert ids(out[0]) == ["1"]
    later = [o for o in out[10:] if o]
    assert later and all(ids(o) == ["2"] for o in later)  # beyond the buffer: a new local id


# --------------------------------------------------------------------------- validation of input


@pytest.mark.parametrize("bad", ["abc", None, 5, {PersonDetection((0, 0, 1, 1), 0.9)}])
def test_non_sequence_rejected(bad):
    with pytest.raises(TypeError):
        ByteTrackPersonTracker(FPS).update(bad)


def test_non_detection_items_rejected():
    with pytest.raises(TypeError):
        ByteTrackPersonTracker(FPS).update([(0, 0, 10, 10)])


def test_zero_size_box_rejected():
    with pytest.raises(ValueError):
        ByteTrackPersonTracker(FPS).update([PersonDetection((0, 0, 0, 10), 0.9)])


# --------------------------------------------------------------------------- determinism / immutability


def test_deterministic():
    rng = np.random.default_rng(11)
    frames = [[person(20 + 100 * k + 2 * t, score=float(rng.uniform(0.3, 1))) for k in range(5) if rng.uniform() > 0.25]
              for t in range(40)]
    assert run(frames) == run(frames)


def test_inputs_unchanged():
    frames = [[d, person(400 - 4 * i, y=200)] for i, d in enumerate(walk(20, 10))]
    before = copy.deepcopy(frames)
    run(frames)
    assert frames == before


# --------------------------------------------------------------------------- scope: local ids only, no identity


def test_output_has_only_local_fields():
    assert [f.name for f in dataclasses.fields(TrackedDetection)] == ["local_track_id", "bbox", "detection_score"]
    t = TrackedDetection("1", (0, 0, 1, 1), 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.local_track_id = "P001"


def test_ids_are_not_person_codes():
    out = run([[person(10 + 50 * k) for k in range(12)]])
    assert not any(re.fullmatch(r"P\d{3,}", t.local_track_id) for t in out[0])


def test_ids_restart_per_tracker_and_after_reset():
    video_a = [[d] for d in walk(50, 5)]
    video_b = [[d] for d in walk(300, 5, y=50)]
    assert ids(run(video_a)[0]) == ids(run(video_b)[0]) == ["1"]  # same local id, unrelated videos

    tracker = ByteTrackPersonTracker(FPS)
    for dets in video_a:
        tracker.update(dets)
    tracker.update([person(500, y=300)])
    tracker.update([person(504, y=300)])  # a second track, id "2"
    tracker.reset()
    assert ids(tracker.update(video_b[0])) == ["1"]


def test_satisfies_protocol():
    def first(t: PersonTracker):
        return t.update([person(10)])

    assert ids(first(ByteTrackPersonTracker(FPS))) == ["1"]


def test_does_not_import_reid():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from perception_minimal.tracker import ByteTrackPersonTracker;"
        "from perception_minimal.detector import PersonDetection;"
        "t = ByteTrackPersonTracker(10.0); t.update([PersonDetection((0, 0, 10, 20), 0.9)]);"
        "bad = sorted(m for m in sys.modules if m == 'reid' or m.startswith('reid.') or m == 'app' or m.startswith('app.'));"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- integration: video -> fake detections -> tracker


def bright_box_detector(frame_bgr):
    """Stand-in detector: the bounding box of bright pixels, as one PersonDetection."""
    ys, xs = np.nonzero(frame_bgr.max(axis=2) > 128)
    if len(xs) == 0:
        return []
    return [PersonDetection((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)), 0.9)]


def test_synthetic_video_to_stable_local_track(tmp_path):
    fps, stride, n = 10.0, 2, 30
    path = tmp_path / "walk.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for t in range(n):
        img = np.zeros((120, 160, 3), np.uint8)
        cv2.rectangle(img, (10 + 3 * t, 30), (10 + 3 * t + 19, 30 + 49), (255, 255, 255), -1)
        writer.write(img)
    writer.release()

    seen = []
    with VideoReader(path) as video:
        tracker = ByteTrackPersonTracker(video.metadata.fps / stride)
        for frame in video.frames(stride=stride):
            dets = bright_box_detector(frame.frame_bgr)
            for t in tracker.update(dets):
                assert t.bbox == dets[0].bbox
                seen.append((frame.frame_number, t.local_track_id, t.bbox))

    assert [f for f, _, _ in seen] == list(range(0, n, stride))
    assert {tid for _, tid, _ in seen} == {"1"}
    xs = [b[0] for _, _, b in seen]
    assert xs == sorted(xs) and xs[-1] - xs[0] > 60  # it really moved
