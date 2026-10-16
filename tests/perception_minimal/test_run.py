"""End-to-end runner tests: synthetic video -> fake detector -> real ByteTrack wrapper -> contract files.

No real detector model, weights, network or real-person data.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import perception_minimal.detector as detector_mod
from perception_minimal.detector import PersonDetection
from perception_minimal.export import INCOMPLETE_MARKER, MANIFEST, OBSERVATIONS
from perception_minimal.run import main, run_pipeline
from perception_minimal.tracker import ByteTrackPersonTracker
from perception_minimal.video import VideoError, VideoReader

REPO_ROOT = Path(__file__).resolve().parents[2]
FPS, W, H, N = 10.0, 160, 120, 24


def write_video(path, n=N, seed=0):
    """Two bright boxes moving over a dark textured background."""
    rng = np.random.default_rng(seed)
    background = rng.integers(0, 100, size=(H, W, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for t in range(n):
        img = background.copy()
        cv2.rectangle(img, (5 + 2 * t, 20), (5 + 2 * t + 19, 20 + 49), (230, 230, 230), -1)
        cv2.rectangle(img, (130 - 2 * t, 60), (130 - 2 * t + 15, 60 + 39), (240, 240, 240), -1)
        cv2.circle(img, (5 + 2 * t + 10, 35), 4, (0, 0, 255), -1)  # some detail inside the box
        writer.write(img)
    writer.release()
    return path


class BlobDetector:
    """Deterministic stand-in detector: one PersonDetection per bright connected region."""

    score_threshold = 0.5

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def detect(self, frame_bgr):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("detector failed")
        mask = (frame_bgr.min(axis=2) > 180).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        dets = []
        for x, y, w, h, area in stats[1:]:
            if area >= 50:
                dets.append(PersonDetection((int(x), int(y), int(w), int(h)), round(0.6 + area / 10000, 6)))
        return sorted(dets, key=lambda d: d.bbox)


@pytest.fixture
def video(tmp_path):
    d = tmp_path / "videos"
    d.mkdir()
    return write_video(d / "synthetic.avi")


def run(video, out, **kw):
    args = dict(out_dir=out, camera_code="CAM_A", detector=BlobDetector(), stride=1, crop_every=3)
    args.update(kw)
    return run_pipeline(video, **args)


def load(out):
    manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    lines = (out / OBSERVATIONS).read_text(encoding="utf-8").splitlines()
    return manifest, [json.loads(line) for line in lines]


def decoded_frames(video, stride=1):
    with VideoReader(video) as v:
        return {f.frame_number: f.frame_bgr for f in v.frames(stride)}


# --------------------------------------------------------------------------- complete flow


def test_complete_flow(video, tmp_path):
    out = tmp_path / "runs" / "r1"
    manifest = run(video, out)
    on_disk, obs = load(out)
    assert on_disk == manifest
    assert sorted(p.name for p in out.iterdir()) == sorted([MANIFEST, OBSERVATIONS, "crops"])
    assert len(manifest["tracks"]) == 2 and [t["local_track_id"] for t in manifest["tracks"]] == ["1", "2"]
    assert manifest["num_observations"] == len(obs) == 2 * N  # both boxes found and matched every frame


def test_observation_lines_and_fields(video, tmp_path):
    out = tmp_path / "r"
    run(video, out)
    text = (out / OBSERVATIONS).read_text(encoding="utf-8")
    assert text.endswith("\n")
    required = {"source_video_id", "local_track_id", "frame_number", "timestamp_seconds", "bbox",
                "detection_confidence", "crop_path"}
    for line in text.splitlines():
        o = json.loads(line)
        assert set(o) == required
        assert isinstance(o["local_track_id"], str) and isinstance(o["frame_number"], int)
        assert isinstance(o["bbox"], list) and len(o["bbox"]) == 4 and all(type(v) is int for v in o["bbox"])


def test_bbox_and_score_are_exact_detector_values(video, tmp_path):
    out = tmp_path / "r"
    run(video, out)
    _, obs = load(out)
    frames = decoded_frames(video)
    detector = BlobDetector()
    expected = {fn: {(d.bbox, d.score) for d in detector.detect(f)} for fn, f in frames.items()}
    for o in obs:
        assert (tuple(o["bbox"]), o["detection_confidence"]) in expected[o["frame_number"]]


def test_crops_equal_frame_slices(video, tmp_path):
    out = tmp_path / "r"
    run(video, out, crop_every=4)
    _, obs = load(out)
    frames = decoded_frames(video)
    n_crops = 0
    for o in obs:
        if o["crop_path"] is None:
            continue
        n_crops += 1
        assert re.fullmatch(r"crops/\d+/\d{8}\.png", o["crop_path"])
        assert o["crop_path"] == f"crops/{o['local_track_id']}/{o['frame_number']:08d}.png"
        crop = cv2.imdecode(np.frombuffer((out / o["crop_path"]).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        x, y, w, h = o["bbox"]
        np.testing.assert_array_equal(crop, frames[o["frame_number"]][y:y + h, x:x + w])
    assert n_crops == len(list((out / "crops").rglob("*.png")))


@pytest.mark.parametrize("k", [1, 2, 5, 7, 100])
def test_crop_every_first_last_and_counts(video, tmp_path, k):
    out = tmp_path / f"r{k}"
    manifest = run(video, out, crop_every=k)
    _, obs = load(out)
    for t in manifest["tracks"]:
        mine = [o for o in obs if o["local_track_id"] == t["local_track_id"]]
        saved = [i for i, o in enumerate(mine) if o["crop_path"] is not None]
        expected = sorted(set(range(0, len(mine), k)) | {len(mine) - 1})
        assert saved == expected  # every k-th, plus first (index 0) and last
        assert t["num_crops"] == len(saved) >= 1
        assert t["num_observations"] == len(mine)
        assert t["first_frame_number"] == mine[0]["frame_number"] and t["last_frame_number"] == mine[-1]["frame_number"]
        assert t["first_timestamp_seconds"] == mine[0]["timestamp_seconds"]
        assert t["last_timestamp_seconds"] == mine[-1]["timestamp_seconds"]
        assert len(list((out / "crops" / t["local_track_id"]).iterdir())) == t["num_crops"]


# --------------------------------------------------------------------------- stride / frame rate / time


def test_stride_keeps_source_frame_numbers_and_sets_tracker_rate(video, tmp_path):
    rates = []

    def factory(rate):
        rates.append(rate)
        return ByteTrackPersonTracker(rate)

    out = tmp_path / "r"
    manifest = run(video, out, stride=3, tracker_factory=factory)
    _, obs = load(out)
    assert rates == [FPS / 3]
    assert sorted({o["frame_number"] for o in obs}) == list(range(0, N, 3))
    assert all(o["timestamp_seconds"] == o["frame_number"] / FPS for o in obs)
    p = manifest["processing"]
    assert manifest["fps"] == FPS and p["stride"] == 3 and p["tracker_frame_rate"] == FPS / 3
    assert p["processed_frames"] == len(range(0, N, 3)) and p["last_processed_frame_number"] == 21
    assert manifest["producer"]["tracker"]["frame_rate"] == FPS / 3


def test_video_start_time(video, tmp_path):
    assert run(video, tmp_path / "a")["video_start_time"] is None
    m = run(video, tmp_path / "b", video_start_time="2026-10-14T09:00:00+05:00")
    assert m["video_start_time"] == "2026-10-14T09:00:00+05:00"
    with pytest.raises(ValueError, match="timezone"):
        run(video, tmp_path / "c", video_start_time="2026-10-14T09:00:00")
    assert not (tmp_path / "c").exists()


# --------------------------------------------------------------------------- ids


def test_explicit_source_video_id(video, tmp_path):
    out = tmp_path / "r"
    m = run(video, out, source_video_id="cam-a-2026-10-14-0900")
    _, obs = load(out)
    assert m["source_video_id"] == "cam-a-2026-10-14-0900"
    assert {o["source_video_id"] for o in obs} == {"cam-a-2026-10-14-0900"}


def test_default_source_video_id_is_sha256_of_file(video, tmp_path):
    expected = "sha256:" + hashlib.sha256(video.read_bytes()).hexdigest()
    assert run(video, tmp_path / "a")["source_video_id"] == expected
    assert run(video, tmp_path / "b")["source_video_id"] == expected  # reproducible
    other = write_video(tmp_path / "videos" / "other.avi", seed=1)
    assert run(other, tmp_path / "c")["source_video_id"] != expected


@pytest.mark.parametrize("bad", ["", "  ", " id", "id\t", 7])
def test_invalid_source_video_id(video, tmp_path, bad):
    with pytest.raises(ValueError, match="source_video_id"):
        run(video, tmp_path / "r", source_video_id=bad)
    assert not (tmp_path / "r").exists()


def test_camera_code(video, tmp_path):
    assert run(video, tmp_path / "a", camera_code="CAM-LOBBY-01")["camera_code"] == "CAM-LOBBY-01"
    for bad in ["", " CAM", "C" * 51, None]:
        with pytest.raises(ValueError, match="camera_code"):
            run(video, tmp_path / "b", camera_code=bad)
    assert not (tmp_path / "b").exists()


# --------------------------------------------------------------------------- manifest


def test_manifest_contents(video, tmp_path):
    m = run(video, tmp_path / "r", stride=2, crop_every=5)
    assert m["contract_version"] == "1"
    for key in ("source_video_id", "camera_code", "fps", "width", "height", "frame_count", "timestamp_method",
                "video_start_time", "processing", "producer", "num_observations", "tracks"):
        assert key in m
    assert (m["width"], m["height"], m["frame_count"]) == (W, H, N)
    assert m["timestamp_method"] == "frame_index/fps"
    assert m["processing"]["crop_every"] == 5
    producer = m["producer"]
    assert producer["name"] == "perception_minimal"
    assert producer["score_threshold"] == 0.5
    assert producer["detector"] == {"class": "BlobDetector"}
    assert producer["tracker"]["tracker"] == "RejectAndSplitPersonTracker"
    assert producer["tracker"]["gate"] == "reject_and_split_hsv"
    assert producer["tracker"]["inner"]["tracker"].endswith("ByteTrack")
    assert set(producer["packages"]) == {"numpy", "opencv-python-headless", "torch", "torchvision", "supervision", "scipy"}
    assert producer["packages"]["supervision"] == "0.30.9"


def test_no_identity_fields(video, tmp_path):
    out = tmp_path / "r"
    m = run(video, out)
    _, obs = load(out)

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    all_keys = set(keys(m)) | set(keys(obs))
    assert not [k for k in all_keys if re.search(r"person|identity|global|embedding|name", k, re.I) and k != "name"]
    text = (out / MANIFEST).read_text() + (out / OBSERVATIONS).read_text()
    assert not re.search(r"\bP\d{3,}\b", text)


def test_deterministic(video, tmp_path):
    run(video, tmp_path / "a", stride=2, crop_every=3)
    run(video, tmp_path / "b", stride=2, crop_every=3)
    for name in (MANIFEST, OBSERVATIONS):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    crops_a = {p.relative_to(tmp_path / "a"): p.read_bytes() for p in (tmp_path / "a").rglob("*.png")}
    crops_b = {p.relative_to(tmp_path / "b"): p.read_bytes() for p in (tmp_path / "b").rglob("*.png")}
    assert crops_a == crops_b and crops_a


# --------------------------------------------------------------------------- failures


def test_unreadable_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        run(bad, tmp_path / "r")
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "missing.mp4", tmp_path / "r")
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("field, value", [("stride", 0), ("stride", 1.5), ("crop_every", 0), ("crop_every", True)])
def test_invalid_intervals(video, tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        run(video, tmp_path / "r", **{field: value})
    assert not (tmp_path / "r").exists()


def test_failed_run_has_no_manifest(video, tmp_path):
    out = tmp_path / "r"
    with pytest.raises(RuntimeError, match="detector failed"):
        run(video, out, detector=BlobDetector(fail_on_call=10))
    assert (out / INCOMPLETE_MARKER).exists()
    assert not (out / MANIFEST).exists() and not (out / OBSERVATIONS).exists()
    # the incomplete directory is not silently reused either
    with pytest.raises(FileExistsError):
        run(video, out)
    run(video, out, overwrite=True)
    assert (out / MANIFEST).exists() and not (out / INCOMPLETE_MARKER).exists()


def test_malformed_detector_result(video, tmp_path):
    class Bad:
        def detect(self, frame):
            return [((0, 0, 10, 10), 0.9)]

    with pytest.raises(TypeError):
        run(video, tmp_path / "r", detector=Bad())
    assert not (tmp_path / "r" / MANIFEST).exists()


def test_malformed_tracker_result(video, tmp_path):
    class Bad:
        def update(self, detections, frame_bgr=None):
            from perception_minimal.tracker import TrackedDetection
            return [TrackedDetection("1", (W - 5, 0, 20, 20), 0.9)]  # outside the frame

    with pytest.raises(ValueError, match="inside"):
        run(video, tmp_path / "r", tracker_factory=lambda rate: Bad())
    assert not (tmp_path / "r" / MANIFEST).exists()


def test_output_collision(video, tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    (out / "other.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        run(video, out)
    with pytest.raises(FileExistsError):
        run(video, out, overwrite=True)
    assert [p.name for p in out.iterdir()] == ["other.txt"]


# --------------------------------------------------------------------------- isolation


def test_writes_only_inside_out(video, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "runs" / "r"

    def snapshot():
        return sorted(p for p in tmp_path.rglob("*") if out not in p.parents and p != out)

    before = snapshot()
    run(video, out)
    after = snapshot()
    assert [p for p in after if p not in before] == [tmp_path / "runs"]  # only the parent of --out
    assert list(cwd.iterdir()) == []


def test_does_not_import_reid(video, tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]);"
        "from test_run import BlobDetector;"
        "from perception_minimal.run import run_pipeline;"
        "run_pipeline(sys.argv[3], out_dir=sys.argv[4], camera_code='CAM_A', detector=BlobDetector(), stride=2, crop_every=3);"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('reid', 'app'));"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT), str(Path(__file__).parent), str(video), str(tmp_path / "r")],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "r" / MANIFEST).exists()


# --------------------------------------------------------------------------- CLI


class FakeTorchvisionDetector(BlobDetector):
    built = []

    def __init__(self, score_threshold):
        super().__init__()
        self.score_threshold = score_threshold
        self.provenance = {"detector": "fake", "score_threshold": score_threshold}
        FakeTorchvisionDetector.built.append(self)


@pytest.fixture
def fake_model(monkeypatch):
    FakeTorchvisionDetector.built = []
    monkeypatch.setattr(detector_mod, "TorchvisionPersonDetector", FakeTorchvisionDetector)
    return FakeTorchvisionDetector


def cli(video, out, *extra):
    return ["--camera-code", "CAM_A", "--out", str(out), "--score-threshold", "0.37", "--stride", "2",
            "--crop-every", "4", *extra, str(video)]


def test_cli_run(video, tmp_path, fake_model, capsys):
    out = tmp_path / "runs" / "cli"
    assert main(cli(video, out, "--video-start-time", "2026-10-14T09:00:00+00:00", "--source-video-id", "v1")) == 0
    m, obs = load(out)
    assert fake_model.built[0].score_threshold == 0.37
    assert m["producer"]["score_threshold"] == 0.37 and m["producer"]["detector"]["score_threshold"] == 0.37
    assert m["source_video_id"] == "v1" and m["video_start_time"] == "2026-10-14T09:00:00+00:00"
    assert m["processing"]["stride"] == 2 and m["processing"]["crop_every"] == 4
    assert "wrote" in capsys.readouterr().out


@pytest.mark.parametrize("flag, value", [
    ("--score-threshold", "nan"), ("--score-threshold", "1.5"), ("--score-threshold", "-0.1"),
    ("--stride", "0"), ("--crop-every", "0"), ("--camera-code", " "), ("--video-start-time", "2026-10-14T09:00:00"),
    ("--source-video-id", " "),
])
def test_cli_rejects_bad_values_before_loading_model(video, tmp_path, fake_model, capsys, flag, value):
    out = tmp_path / "r"
    argv = cli(video, out)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv[-1:-1] = [flag, value]
    assert main(argv) == 1
    assert "error:" in capsys.readouterr().err
    assert fake_model.built == [] and not out.exists()


def test_cli_rejects_output_conflict_and_bad_video_before_loading_model(video, tmp_path, fake_model):
    out = tmp_path / "r"
    out.mkdir()
    (out / "x").write_text("x")
    assert main(cli(video, out)) == 1
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"nope")
    assert main(cli(bad, tmp_path / "r2")) == 1
    assert fake_model.built == [] and not (tmp_path / "r2").exists()


def test_cli_requires_explicit_settings(video, tmp_path):
    for missing in ("--camera-code", "--out", "--score-threshold", "--stride", "--crop-every"):
        argv = cli(video, tmp_path / "r")
        i = argv.index(missing)
        del argv[i:i + 2]
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2


# --------------------------------------------------------------------------- portability of exported provenance


def test_manifest_has_no_local_paths(video, tmp_path, monkeypatch):
    # Real TorchvisionPersonDetector class with its model replaced (no weights load, no network).
    import torch
    import torchvision.models.detection as tv_detection

    from tests.perception_minimal.test_detector import FakeModel, model_output

    monkeypatch.setattr(tv_detection, "fasterrcnn_resnet50_fpn_v2", lambda **kw: FakeModel(model_output([]), **kw))
    out = tmp_path / "r"
    manifest = run(video, out, detector=detector_mod.TorchvisionPersonDetector(0.5))
    text = (out / MANIFEST).read_text(encoding="utf-8")

    provenance = manifest["producer"]["detector"]
    assert provenance["weights_file"] == "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    assert "weights_cache_path" not in provenance
    for local in (str(Path.home()), torch.hub.get_dir(), str(tmp_path), str(REPO_ROOT), str(video)):
        for form in (local, local.replace("\\", "/"), json.dumps(local)[1:-1]):
            assert form not in text
    assert not re.search(r'"[A-Za-z]:\\', text) and not re.search(r'"/(home|Users|root)/', text)
