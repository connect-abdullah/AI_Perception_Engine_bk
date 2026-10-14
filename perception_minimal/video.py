"""Local video ingestion with OpenCV.

Frames are numbered from 0 in source-video order. With a stride, skipped frames still
count, so stride=3 yields source frames 0, 3, 6, ... with their original numbers.

Timestamps are `frame_number / fps` (method "frame_index/fps"). This assumes a constant
frame rate equal to the container's reported fps; it is not accurate for variable
frame rate video.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TIMESTAMP_METHOD = "frame_index/fps"


class VideoError(ValueError):
    """The file exists but cannot be used as a video."""


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float  # as reported by the container
    width: int  # of decoded frames
    height: int  # of decoded frames
    reported_frame_count: int | None  # container estimate; None if not reported
    timestamp_method: str = TIMESTAMP_METHOD


@dataclass(frozen=True)
class VideoFrame:
    frame_number: int  # 0-based source-video frame number
    timestamp_seconds: float  # seconds from this video's start
    frame_bgr: np.ndarray  # uint8, shape (height, width, 3)


def _check_frame(frame, width: int | None = None, height: int | None = None) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        desc = (frame.dtype, frame.shape) if isinstance(frame, np.ndarray) else type(frame).__name__
        raise VideoError(f"Decoded frame is not a uint8 BGR image: {desc}")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise VideoError(f"Decoded frame is empty: shape {frame.shape}")
    if width is not None and (frame.shape[1], frame.shape[0]) != (width, height):
        raise VideoError(f"Frame size changed mid-video: {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}")


class VideoReader:
    """Open and validate a local video; read its frames once.

    Use as a context manager so the capture is always released:

        with VideoReader(path) as video:
            for frame in video.frames(stride=3):
                ...
    """

    def __init__(self, path: str | os.PathLike):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        self._cap = cv2.VideoCapture(str(path))
        try:
            if not self._cap.isOpened():
                raise VideoError(f"Cannot open video: {path}")
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise VideoError(f"Invalid fps {fps!r} in {path}")
            reported_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            reported_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not (math.isfinite(reported_w) and math.isfinite(reported_h) and reported_w > 0 and reported_h > 0):
                raise VideoError(f"Invalid frame size {reported_w!r}x{reported_h!r} in {path}")
            count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # Decode the first frame now so an undecodable file fails here, not mid-pipeline.
            # It is kept and yielded first, so no seek back is needed.
            ok, first = self._cap.read()
            if not ok or first is None:
                raise VideoError(f"No frame could be decoded from {path}")
            _check_frame(first)
        except BaseException:
            self._cap.release()
            raise

        self._first: np.ndarray | None = first
        self._started = False
        # The decoded frame size is authoritative (it is what bboxes will refer to).
        self.metadata = VideoMetadata(
            path=path,
            fps=float(fps),
            width=int(first.shape[1]),
            height=int(first.shape[0]),
            reported_frame_count=int(count) if math.isfinite(count) and count > 0 else None,
        )

    @property
    def closed(self) -> bool:
        return self._cap is None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._first = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def frames(self, stride: int = 1) -> Iterator[VideoFrame]:
        """Yield every `stride`-th source frame (0, stride, 2*stride, ...) once.

        Stops at the end of the stream and releases the capture. Can be called only once.
        """
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"stride must be an integer >= 1, got {stride!r}")
        if self.closed:
            raise VideoError("Video is closed")
        if self._started:
            raise VideoError("frames() can only be iterated once per VideoReader")
        self._started = True
        return self._frames(stride)

    def _frames(self, stride: int) -> Iterator[VideoFrame]:
        fps, width, height = self.metadata.fps, self.metadata.width, self.metadata.height
        try:
            first, self._first = self._first, None
            yield VideoFrame(0, 0 / fps, first)
            frame_number = 0
            while self._cap is not None:
                if not self._cap.grab():  # end of stream (or an undecodable packet; OpenCV cannot tell)
                    break
                frame_number += 1
                if frame_number % stride:
                    continue
                ok, frame = self._cap.retrieve()
                if not ok or frame is None:
                    raise VideoError(f"Frame {frame_number} could not be decoded")
                _check_frame(frame, width, height)
                yield VideoFrame(frame_number, frame_number / fps, frame)
        finally:
            self.close()


def iter_frames(path: str | os.PathLike, stride: int = 1) -> Iterator[VideoFrame]:
    """Open `path`, yield its frames (see VideoReader.frames) and release it when done or closed."""
    with VideoReader(path) as video:
        yield from video.frames(stride)
