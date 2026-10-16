"""RunWriter tests: crop selection, observation/summary files, completion and failure states."""

import json

import cv2
import numpy as np
import pytest

from perception_minimal import export as export_mod
from perception_minimal.export import (
    INCOMPLETE_MARKER,
    MANIFEST,
    OBSERVATIONS,
    PARTIAL_OBSERVATIONS,
    RunWriter,
    check_camera_code,
    parse_video_start_time,
    prepare_output_dir,
)
from perception_minimal.tracker import TrackedDetection

W, H = 160, 120


def frame(seed):
    return np.random.default_rng(seed).integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def tracked(tid, x, y=10, w=20, h=40, score=0.9):
    return TrackedDetection(str(tid), (x, y, w, h), score)


def read_png(path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)


def observations(out):
    return [json.loads(line) for line in (out / OBSERVATIONS).read_text(encoding="utf-8").splitlines()]


def write_track(out, n, crop_every, start=0):
    """One track observed on frames start..start+n-1; returns {frame_number: frame}."""
    writer = RunWriter(out, source_video_id="vid", crop_every=crop_every)
    frames = {}
    for i in range(n):
        f = frame(i)
        frames[start + i] = f
        writer.add_frame(start + i, (start + i) / 10.0, f, [tracked(1, 5 + i)])
    manifest = writer.finish({})
    return manifest, frames


# --------------------------------------------------------------------------- crop selection


@pytest.mark.parametrize("n, k, expected", [
    (10, 1, list(range(10))),
    (10, 3, [0, 3, 6, 9]),  # last already on the schedule
    (10, 4, [0, 4, 8, 9]),  # last added at finish
    (10, 100, [0, 9]),  # only first and last
    (1, 5, [0]),  # single observation: first == last, one crop
    (2, 5, [0, 1]),
])
def test_crop_every_first_and_last(tmp_path, n, k, expected):
    out = tmp_path / "run"
    manifest, _ = write_track(out, n, k)
    obs = observations(out)
    with_crop = [o["frame_number"] for o in obs if o["crop_path"] is not None]
    assert with_crop == expected
    assert sorted(int(p.stem) for p in (out / "crops" / "1").iterdir()) == expected
    (track,) = manifest["tracks"]
    assert track["num_crops"] == len(expected) and track["num_observations"] == n


def test_crop_pixels_equal_frame_slice_and_paths(tmp_path):
    out = tmp_path / "run"
    _, frames = write_track(out, 7, 3, start=40)
    for o in observations(out):
        if o["crop_path"] is None:
            continue
        x, y, w, h = o["bbox"]
        assert o["crop_path"] == f"crops/1/{o['frame_number']:08d}.png"
        crop = read_png(out / o["crop_path"])
        np.testing.assert_array_equal(crop, frames[o["frame_number"]][y:y + h, x:x + w])
        assert crop.shape == (h, w, 3)


def test_pending_crop_is_a_copy_of_only_the_box(tmp_path):
    writer = RunWriter(tmp_path / "run", source_video_id="vid", crop_every=10)
    writer.add_frame(0, 0.0, frame(0), [tracked(1, 5)])
    f1 = frame(1)
    writer.add_frame(1, 0.1, f1, [tracked(1, 6)])
    pending = writer._tracks["1"].pending_crop
    assert pending.shape == (40, 20, 3) and pending.base is None  # owns its data; no frame kept
    expected = f1[10:50, 6:26].copy()
    f1[:] = 0  # caller reuses its frame buffer
    np.testing.assert_array_equal(pending, expected)
    writer.finish({})
    np.testing.assert_array_equal(read_png(tmp_path / "run" / "crops" / "1" / "00000001.png"), expected)


def test_several_tracks_summaries_match_observations(tmp_path):
    out = tmp_path / "run"
    writer = RunWriter(out, source_video_id="vid", crop_every=2)
    plan = {0: [1], 1: [1, 2], 2: [2, 1], 3: [2], 5: [2, 10], 6: [10]}  # frame -> track ids (unsorted)
    for fn, tids in plan.items():
        writer.add_frame(fn, fn / 10.0, frame(fn), [tracked(t, 5 + 30 * (t % 4)) for t in tids])
    manifest = writer.finish({"extra": 1})
    obs = observations(out)

    assert [(o["frame_number"], o["local_track_id"]) for o in obs] == [
        (0, "1"), (1, "1"), (1, "2"), (2, "1"), (2, "2"), (3, "2"), (5, "2"), (5, "10"), (6, "10")]
    assert manifest["num_observations"] == len(obs)
    assert [t["local_track_id"] for t in manifest["tracks"]] == ["1", "2", "10"]
    for t in manifest["tracks"]:
        mine = [o for o in obs if o["local_track_id"] == t["local_track_id"]]
        assert t["first_frame_number"] == mine[0]["frame_number"]
        assert t["last_frame_number"] == mine[-1]["frame_number"]
        assert t["first_timestamp_seconds"] == mine[0]["timestamp_seconds"]
        assert t["last_timestamp_seconds"] == mine[-1]["timestamp_seconds"]
        assert t["num_observations"] == len(mine)
        assert t["num_crops"] == sum(o["crop_path"] is not None for o in mine) >= 1
        assert mine[0]["crop_path"] is not None and mine[-1]["crop_path"] is not None
    assert manifest["extra"] == 1


def test_observation_fields_and_types(tmp_path):
    out = tmp_path / "run"
    writer = RunWriter(out, source_video_id="sha256:abc", crop_every=1)
    writer.add_frame(3, 0.3, frame(0), [TrackedDetection("7", (1, 2, 30, 40), 0.123456789)])
    writer.finish({})
    (o,) = observations(out)
    assert o == {
        "source_video_id": "sha256:abc",
        "local_track_id": "7",
        "frame_number": 3,
        "timestamp_seconds": 0.3,
        "bbox": [1, 2, 30, 40],
        "detection_confidence": 0.123456789,
        "crop_path": "crops/7/00000003.png",
    }


# --------------------------------------------------------------------------- completion / failure


def test_complete_run_files(tmp_path):
    out = tmp_path / "run"
    write_track(out, 3, 1)
    assert sorted(p.name for p in out.iterdir()) == sorted([MANIFEST, OBSERVATIONS, "crops"])


def test_context_manager_aborts_unfinished_run(tmp_path):
    out = tmp_path / "run"
    with RunWriter(out, source_video_id="vid", crop_every=1) as writer:
        writer.add_frame(0, 0.0, frame(0), [tracked(1, 5)])
    assert writer._partial.closed
    assert (out / INCOMPLETE_MARKER).exists() and not (out / MANIFEST).exists()


def test_in_progress_and_aborted_run_has_no_manifest(tmp_path):
    out = tmp_path / "run"
    writer = RunWriter(out, source_video_id="vid", crop_every=1)
    writer.add_frame(0, 0.0, frame(0), [tracked(1, 5)])
    assert (out / INCOMPLETE_MARKER).exists() and (out / PARTIAL_OBSERVATIONS).exists()
    assert not (out / MANIFEST).exists() and not (out / OBSERVATIONS).exists()
    writer.abort()
    assert (out / INCOMPLETE_MARKER).exists() and not (out / MANIFEST).exists() and not (out / OBSERVATIONS).exists()
    with pytest.raises(RuntimeError):
        writer.add_frame(1, 0.1, frame(1), [])


def test_crop_write_failure(tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(export_mod.cv2, "imencode", lambda *a, **k: (False, None))
    with pytest.raises(OSError, match="encode"):
        with RunWriter(out, source_video_id="vid", crop_every=1) as writer:
            writer.add_frame(0, 0.0, frame(0), [tracked(1, 5)])
    assert (out / INCOMPLETE_MARKER).exists() and not (out / MANIFEST).exists()


@pytest.mark.parametrize("bad", [
    "not a list",
    [("1", (0, 0, 5, 5), 0.9)],
    [TrackedDetection("1", (0, 0, 5, 5), 0.9), TrackedDetection("1", (10, 0, 5, 5), 0.8)],  # duplicate id
    [TrackedDetection("../x", (0, 0, 5, 5), 0.9)],  # unsafe id for a path
    [TrackedDetection("", (0, 0, 5, 5), 0.9)],
    [TrackedDetection(1, (0, 0, 5, 5), 0.9)],
    [TrackedDetection("1", (0.0, 0, 5, 5), 0.9)],
    [TrackedDetection("1", (0, 0, 0, 5), 0.9)],
    [TrackedDetection("1", (150, 0, 20, 5), 0.9)],  # beyond right edge
    [TrackedDetection("1", (-1, 0, 5, 5), 0.9)],
    [TrackedDetection("1", (0, 0, 5, 5), 1.5)],
    [TrackedDetection("1", (0, 0, 5, 5), float("nan"))],
    [TrackedDetection("1", (0, 0, 5, 5), True)],
], ids=lambda v: repr(v)[:40])
def test_malformed_tracker_result_rejected(tmp_path, bad):
    with RunWriter(tmp_path / "run", source_video_id="vid", crop_every=1) as writer:
        with pytest.raises((TypeError, ValueError)):
            writer.add_frame(0, 0.0, frame(0), bad)


def test_frames_must_increase(tmp_path):
    with RunWriter(tmp_path / "run", source_video_id="vid", crop_every=1) as writer:
        writer.add_frame(5, 0.5, frame(0), [])
        for fn in (5, 4):
            with pytest.raises(ValueError, match="increasing"):
                writer.add_frame(fn, 0.6, frame(0), [])


def test_manifest_fields_cannot_override_contract_fields(tmp_path):
    with RunWriter(tmp_path / "run", source_video_id="vid", crop_every=1) as writer:
        with pytest.raises(ValueError):
            writer.finish({"contract_version": "2"})
    assert not (tmp_path / "run" / MANIFEST).exists()


@pytest.mark.parametrize("value", [0, -1, 1.0, True, "2", None])
def test_invalid_crop_every(tmp_path, value):
    with pytest.raises(ValueError, match="crop_every"):
        RunWriter(tmp_path / "run", source_video_id="vid", crop_every=value)


# --------------------------------------------------------------------------- output directory


def test_existing_empty_dir_is_used(tmp_path):
    (tmp_path / "run").mkdir()
    write_track(tmp_path / "run", 2, 1)
    assert (tmp_path / "run" / MANIFEST).exists()


def test_non_empty_dir_refused_and_untouched(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        RunWriter(out, source_video_id="vid", crop_every=1)
    with pytest.raises(FileExistsError, match="not from a run"):
        RunWriter(out, source_video_id="vid", crop_every=1, overwrite=True)
    assert [p.name for p in out.iterdir()] == ["notes.txt"]


def test_previous_run_refused_without_overwrite_and_replaced_with_it(tmp_path):
    out = tmp_path / "run"
    write_track(out, 5, 1)
    before = sorted(p.relative_to(out) for p in out.rglob("*"))
    with pytest.raises(FileExistsError):
        RunWriter(out, source_video_id="vid", crop_every=1)
    assert sorted(p.relative_to(out) for p in out.rglob("*")) == before

    writer =RunWriter(out, source_video_id="vid", crop_every=1, overwrite=True)
    writer.add_frame(100, 10.0, frame(0), [tracked(4, 5)])
    writer.finish({})
    assert sorted(p.relative_to(out).as_posix() for p in (out / "crops").rglob("*.png")) == ["crops/4/00000100.png"]


def test_overwrite_refuses_foreign_file_inside_crops(tmp_path):
    out = tmp_path / "run"
    write_track(out, 2, 1)
    (out / "crops" / "1" / "readme.txt").write_text("x")
    with pytest.raises(FileExistsError):
        prepare_output_dir(out, overwrite=True)
    assert (out / MANIFEST).exists()


def test_output_path_is_a_file(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(FileExistsError):
        RunWriter(f, source_video_id="vid", crop_every=1, overwrite=True)


# --------------------------------------------------------------------------- argument validators


@pytest.mark.parametrize("bad", ["", "   ", " CAM", "CAM ", "CAM\n1", "C" * 51, None, 5])
def test_invalid_camera_code(bad):
    with pytest.raises(ValueError, match="camera_code"):
        check_camera_code(bad)


def test_valid_camera_code():
    assert check_camera_code("CAM-LOBBY-01") == "CAM-LOBBY-01"
    assert check_camera_code("C" * 50) == "C" * 50


@pytest.mark.parametrize("value, expected", [
    (None, None),
    ("2026-10-14T09:00:00+05:00", "2026-10-14T09:00:00+05:00"),
    ("2026-10-14T09:00:00.250-04:00", "2026-10-14T09:00:00.250000-04:00"),
    ("2026-10-14T09:00:00Z", "2026-10-14T09:00:00+00:00"),
])
def test_video_start_time_valid(value, expected):
    assert parse_video_start_time(value) == expected


@pytest.mark.parametrize("bad", ["2026-10-14T09:00:00", "2026-10-14", "yesterday", "", 1700000000])
def test_video_start_time_invalid(bad):
    with pytest.raises(ValueError, match="video_start_time"):
        parse_video_start_time(bad)
