"""Video ingestion tests. Every video is synthetic and written to pytest's tmp_path."""

import cv2
import numpy as np
import pytest

from perception_minimal import video as video_mod
from perception_minimal.video import TIMESTAMP_METHOD, VideoError, VideoFrame, VideoReader, iter_frames

FPS = 12.5
WIDTH, HEIGHT = 64, 48
N_FRAMES = 10
LEVEL_STEP = 20  # frame i is a flat grey image of value i * LEVEL_STEP


def write_video(path, n_frames=N_FRAMES, fps=FPS, size=(WIDTH, HEIGHT)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            writer.write(np.full((size[1], size[0], 3), i * LEVEL_STEP, np.uint8))
    finally:
        writer.release()
    return path


def source_index(frame):
    """Recover which source frame was decoded from its grey level (MJPG is lossy)."""
    return int(round(float(frame.mean()) / LEVEL_STEP))


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "synthetic.avi")


# --------------------------------------------------------------------------- metadata / decoding


def test_metadata(video):
    with VideoReader(video) as v:
        m = v.metadata
    assert m.path == video
    assert m.fps == pytest.approx(FPS)
    assert (m.width, m.height) == (WIDTH, HEIGHT)
    assert m.reported_frame_count == N_FRAMES
    assert m.timestamp_method == TIMESTAMP_METHOD == "frame_index/fps"


def test_frames_decode_as_bgr_uint8(video):
    frames = list(iter_frames(video))
    assert len(frames) == N_FRAMES
    for f in frames:
        assert isinstance(f, VideoFrame)
        assert isinstance(f.frame_bgr, np.ndarray)
        assert f.frame_bgr.dtype == np.uint8 and f.frame_bgr.shape == (HEIGHT, WIDTH, 3)


def test_frame_numbers_are_zero_based_source_order(video):
    frames = list(iter_frames(video, stride=1))
    assert [f.frame_number for f in frames] == list(range(N_FRAMES))
    assert [source_index(f.frame_bgr) for f in frames] == list(range(N_FRAMES))


def test_timestamps_are_frame_number_over_fps(video):
    with VideoReader(video) as v:
        fps = v.metadata.fps
        frames = list(v.frames())
    assert frames[0].timestamp_seconds == 0.0
    for f in frames:
        assert f.timestamp_seconds == f.frame_number / fps
    assert frames[-1].timestamp_seconds == pytest.approx((N_FRAMES - 1) / FPS)


# --------------------------------------------------------------------------- stride


@pytest.mark.parametrize("stride, expected", [
    (1, list(range(N_FRAMES))),
    (2, [0, 2, 4, 6, 8]),
    (3, [0, 3, 6, 9]),
    (4, [0, 4, 8]),
    (N_FRAMES - 1, [0, N_FRAMES - 1]),
    (N_FRAMES, [0]),
    (100, [0]),
])
def test_stride_preserves_source_frame_numbers(video, stride, expected):
    frames = list(iter_frames(video, stride=stride))
    assert [f.frame_number for f in frames] == expected
    # the pixels really are those source frames, not renumbered consecutive ones
    assert [source_index(f.frame_bgr) for f in frames] == expected
    for f in frames:
        assert f.timestamp_seconds == pytest.approx(f.frame_number / FPS)


@pytest.mark.parametrize("stride", [0, -1, 1.0, 2.5, "2", None, True])
def test_invalid_stride_rejected(video, stride):
    with VideoReader(video) as v:
        with pytest.raises(ValueError, match="stride"):
            v.frames(stride)
    with pytest.raises(ValueError, match="stride"):
        next(iter_frames(video, stride=stride))


def test_frames_can_be_iterated_only_once(video):
    with VideoReader(video) as v:
        list(v.frames())
        with pytest.raises(VideoError):
            v.frames()


# --------------------------------------------------------------------------- invalid inputs


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        next(iter_frames(tmp_path / "does_not_exist.mp4"))


def test_directory_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path)


@pytest.mark.parametrize("name, content", [
    ("text.mp4", b"this is not a video"),
    ("empty.avi", b""),
    ("noise.avi", bytes(range(256)) * 64),
], ids=["text", "empty", "noise"])
def test_non_video_rejected(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(VideoError):
        VideoReader(path)


def test_video_with_no_frames_rejected(tmp_path):
    path = write_video(tmp_path / "zero.avi", n_frames=0)
    with pytest.raises(VideoError):
        VideoReader(path)


# --------------------------------------------------------------------------- resource release


class CaptureSpy:
    """Wraps cv2.VideoCapture and records every instance so release can be checked."""

    instances = []
    _real = cv2.VideoCapture  # bound before monkeypatching replaces cv2.VideoCapture

    def __init__(self, *args):
        self._cap = CaptureSpy._real(*args)
        self.release_calls = 0
        CaptureSpy.instances.append(self)

    def release(self):
        self.release_calls += 1
        self._cap.release()

    def __getattr__(self, name):
        return getattr(self._cap, name)


@pytest.fixture
def spy(monkeypatch):
    CaptureSpy.instances = []
    monkeypatch.setattr(video_mod.cv2, "VideoCapture", CaptureSpy)
    return CaptureSpy


def released(spy):
    return [c.release_calls >= 1 and not c._cap.isOpened() for c in spy.instances]


def test_released_after_full_iteration(video, spy):
    v = VideoReader(video)
    list(v.frames())  # no explicit close
    assert released(spy) == [True] and v.closed


def test_released_by_context_manager_on_early_break(video, spy):
    with VideoReader(video) as v:
        for f in v.frames():
            if f.frame_number == 2:
                break
    assert released(spy) == [True] and v.closed


def test_released_on_exception_inside_with(video, spy):
    with pytest.raises(RuntimeError):
        with VideoReader(video) as v:
            next(v.frames())
            raise RuntimeError("consumer failed")
    assert released(spy) == [True]


def test_released_when_iter_frames_generator_closed_early(video, spy):
    gen = iter_frames(video, stride=2)
    next(gen)
    gen.close()
    assert released(spy) == [True]


def test_released_when_iter_frames_exhausted(video, spy):
    list(iter_frames(video))
    assert released(spy) == [True]


def test_released_when_never_iterated(video, spy):
    with VideoReader(video):
        pass
    assert released(spy) == [True]


def test_released_when_open_fails(tmp_path, spy):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        VideoReader(bad)
    zero = write_video(tmp_path / "zero.avi", n_frames=0)
    with pytest.raises(VideoError):
        VideoReader(zero)
    assert spy.instances and all(released(spy))


def test_released_on_invalid_stride_via_iter_frames(video, spy):
    with pytest.raises(ValueError):
        next(iter_frames(video, stride=0))
    assert released(spy) == [True]


def test_close_is_idempotent(video):
    v = VideoReader(video)
    v.close()
    v.close()
    assert v.closed
    with pytest.raises(VideoError):
        v.frames()


# --------------------------------------------------------------------------- no stray files


def test_reading_writes_no_files(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    path = write_video(video_dir / "synthetic.avi")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    def snapshot():
        return sorted((p.relative_to(tmp_path), p.stat().st_size) for p in tmp_path.rglob("*"))

    before = snapshot()
    with VideoReader(path) as v:
        list(v.frames(stride=3))
    list(iter_frames(path))
    assert snapshot() == before
    assert list(cwd.iterdir()) == []
