"""Activity intervals: posture rule, motion, smoothing, short-fragment handling, output shape."""

import pytest

from activity_events.activity import ActivitySettings, foot_speeds, raw_label, run_activity, track_intervals
from activity_events.pose import posture_from_pose
from activity_events.runfolder import RunFolderError, load_run

from .conftest import FakePose, pose


def still(n, x=300, start=0):
    return [(start + i, (x, 100, 50, 150)) for i in range(n)]


def walking(n, start=0, step=6):
    return [(start + i, (50 + i * step, 100, 50, 150)) for i in range(n)]  # 6 px/frame = 4 heights/s at 10 fps


# --------------------------------------------------------------------------- building blocks


def test_posture_rule():
    assert posture_from_pose(pose("standing")) == "standing"
    assert posture_from_pose(pose("sitting")) == "sitting"
    assert posture_from_pose(None) is None


def test_posture_unknown_when_knees_not_confident():
    p = pose("standing")
    p.confidence[13] = p.confidence[14] = 0.1
    assert posture_from_pose(p) is None


@pytest.mark.parametrize("posture, speed, expected", [
    ("sitting", 2.0, "sitting"),
    ("standing", 1.0, "walking"),
    ("standing", 0.0, "standing"),
    (None, 1.0, "moving"),
    (None, 0.0, "still"),
    (None, None, "still"),
])
def test_raw_label(posture, speed, expected):
    assert raw_label(posture, speed, 0.35) == expected


def test_foot_speed_is_in_body_heights(make_run):
    run = load_run(make_run({1: walking(20)}))
    speeds = [s for s in foot_speeds(run.tracks["1"], 1.0) if s is not None]
    assert speeds and all(s == pytest.approx(0.4, rel=0.05) for s in speeds)  # 6 px * 10 fps / 150 px


# --------------------------------------------------------------------------- intervals


def test_sit_then_stand_gives_two_intervals(make_run):
    run = load_run(make_run({1: still(60)}))
    est = FakePose({f: "sitting" for f in range(30)}, default="standing")
    ivs = track_intervals(run, run.tracks["1"], est)
    assert [i.activity for i in ivs] == ["sitting", "standing"]
    assert ivs[0].start_seconds == 0.0 and ivs[1].end_seconds == 5.9
    assert ivs[0].end_seconds == ivs[1].start_seconds  # contiguous
    assert all(0.0 <= i.confidence <= 1.0 for i in ivs)


def test_walking_detected(make_run):
    run = load_run(make_run({1: walking(40, step=10)}))
    ivs = track_intervals(run, run.tracks["1"], FakePose(default="standing"))
    assert [i.activity for i in ivs] == ["walking"]


def test_flicker_is_smoothed_away(make_run):
    run = load_run(make_run({1: still(50)}))
    # one isolated "sitting" crop in a standing track
    ivs = track_intervals(run, run.tracks["1"], FakePose({25: "sitting"}, default="standing"))
    assert [i.activity for i in ivs] == ["standing"]
    assert ivs[0].confidence < 1.0


def test_no_pose_gives_motion_labels_only(make_run):
    run = load_run(make_run({1: still(30), 2: walking(30, step=10)}))
    ivs, summary = run_activity(run, estimator=None)
    assert {(i.local_track_id, i.activity) for i in ivs} == {("1", "still"), ("2", "moving")}
    assert summary["pose_model"] is None


def test_short_fragments_are_skipped(make_run):
    run = load_run(make_run({1: still(5), 2: still(30)}))  # track 1 lasts 0.4 s
    ivs, summary = run_activity(run, FakePose())
    assert {i.local_track_id for i in ivs} == {"2"}
    assert summary["tracks_skipped_short"] == 1


def test_missing_crops_fall_back_to_motion(make_run):
    run = load_run(make_run({1: still(30)}, crops=False))
    est = FakePose()
    ivs = track_intervals(run, run.tracks["1"], est)
    assert est.calls == 0 and [i.activity for i in ivs] == ["still"]


# --------------------------------------------------------------------------- times and backend payload


def test_payload_matches_backend_activity_schema(make_run):
    from app.entities.ingest.schema import ActivityIngestRequest

    run = load_run(make_run({1: still(30)}))
    iv = track_intervals(run, run.tracks["1"], FakePose())[0]
    body = iv.ingest_payload()
    req = ActivityIngestRequest(**body)
    assert req.local_track_id == "1" and req.activity == "standing"
    assert body["start_time"] == "2026-10-17T10:00:00+00:00"


def test_no_start_time_means_offsets_only(make_run):
    run = load_run(make_run({1: still(30)}, start_time=None))
    iv = track_intervals(run, run.tracks["1"], FakePose())[0]
    assert iv.start_time is None and iv.start_seconds == 0.0
    with pytest.raises(ValueError):
        iv.ingest_payload()


# --------------------------------------------------------------------------- run folder checks


def test_incomplete_run_is_refused(make_run):
    root = make_run({1: still(10)})
    (root / "_INCOMPLETE").write_text("")
    with pytest.raises(RunFolderError):
        load_run(root)


def test_run_without_manifest_is_refused(make_run):
    root = make_run({1: still(10)})
    (root / "manifest.json").unlink()
    with pytest.raises(RunFolderError):
        load_run(root)


def test_settings_are_respected(make_run):
    run = load_run(make_run({1: still(15)}))  # 1.4 s long
    assert track_intervals(run, run.tracks["1"], None, ActivitySettings(min_track_seconds=2.0)) == []
