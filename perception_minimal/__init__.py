"""Temporary minimal perception pipeline (video -> detection -> tracking -> crops).

Exists only to unblock ReID integration and is meant to be replaced. Independent of
reid/ and app/; its output boundary is docs/reid_handoff_contract.md.
"""

from perception_minimal.detector import PersonDetection, PersonDetector, TorchvisionPersonDetector
from perception_minimal.video import TIMESTAMP_METHOD, VideoError, VideoFrame, VideoMetadata, VideoReader, iter_frames

__all__ = [
    "TIMESTAMP_METHOD",
    "PersonDetection",
    "PersonDetector",
    "TorchvisionPersonDetector",
    "VideoError",
    "VideoFrame",
    "VideoMetadata",
    "VideoReader",
    "iter_frames",
]
