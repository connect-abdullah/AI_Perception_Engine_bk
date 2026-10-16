"""Detector tests. The torchvision model is replaced by a fake, so no weights or network are needed.

The one real-model smoke test runs only if the pretrained weights are already in the
torch hub cache; it never downloads.
"""

import dataclasses
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.models.detection as tv_detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from perception_minimal.detector import (
    PersonDetection,
    PersonDetector,
    TorchvisionPersonDetector,
    check_score_threshold,
    select_person_detections,
    xyxy_to_clipped_xywh,
)

PERSON = 1  # COCO "person" in torchvision's category list
CAR, DOG = 3, 18
W, H = 640, 480


def test_person_label_matches_weights_metadata():
    assert FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"][PERSON] == "person"


# --------------------------------------------------------------------------- fake model


class FakeModel:
    """Stands in for torchvision's Faster R-CNN: records its input, returns fixed output."""

    def __init__(self, output, **kwargs):
        self.output = output
        self.kwargs = kwargs
        self.inputs = []
        self.roi_heads = type("RoI", (), {"nms_thresh": 0.5, "detections_per_img": 100})()
        self.transform = type("T", (), {"min_size": (800,), "max_size": 1333})()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, images):
        with torch.inference_mode(False):  # keep plain copies the tests can inspect and modify
            self.inputs.append([torch.from_numpy(img.numpy().copy()) for img in images])
        return [self.output]


def model_output(rows):
    """rows: (x1, y1, x2, y2, label, score)."""
    rows = list(rows)
    return {
        "boxes": torch.tensor([r[:4] for r in rows], dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor([r[4] for r in rows], dtype=torch.int64),
        "scores": torch.tensor([r[5] for r in rows], dtype=torch.float32),
    }


@pytest.fixture
def make_detector(monkeypatch):
    built = {}

    def make(rows, threshold):
        def factory(**kwargs):
            built["model"] = FakeModel(model_output(rows), **kwargs)
            return built["model"]

        monkeypatch.setattr(tv_detection, "fasterrcnn_resnet50_fpn_v2", factory)
        detector = TorchvisionPersonDetector(threshold)
        return detector, built["model"]

    return make


def frame(h=H, w=W):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- class filtering


def test_only_person_class_kept(make_detector):
    rows = [
        (10, 10, 50, 100, PERSON, 0.9),
        (100, 10, 200, 100, CAR, 0.99),
        (300, 10, 350, 100, DOG, 0.95),
        (400, 20, 450, 120, PERSON, 0.6),
    ]
    detector, _ = make_detector(rows, 0.5)
    dets = detector.detect(frame())
    assert [d.bbox for d in dets] == [(10, 10, 40, 90), (400, 20, 50, 100)]
    assert all(isinstance(d, PersonDetection) for d in dets)


def test_no_person_gives_empty(make_detector):
    detector, _ = make_detector([(1, 1, 20, 20, CAR, 0.99)], 0.0)
    assert detector.detect(frame()) == []


def test_empty_model_output(make_detector):
    detector, _ = make_detector([], 0.5)
    assert detector.detect(frame()) == []


# --------------------------------------------------------------------------- threshold


def test_threshold_boundary(make_detector):
    t = 0.5
    below = float(np.nextafter(np.float32(t), np.float32(0)))  # largest float32 below t
    rows = [
        (0, 0, 10, 10, PERSON, t),  # exactly at threshold: kept
        (20, 0, 30, 10, PERSON, below),  # immediately below: removed
        (40, 0, 50, 10, PERSON, 0.75),
        (60, 0, 70, 10, PERSON, 0.25),
    ]
    detector, _ = make_detector(rows, t)
    assert [(d.bbox[0], d.score) for d in detector.detect(frame())] == [(40, 0.75), (0, 0.5)]


@pytest.mark.parametrize("threshold, n_kept", [(0.0, 3), (0.3, 2), (0.8, 1), (1.0, 0)])
def test_threshold_applied(make_detector, threshold, n_kept):
    rows = [(0, 0, 10, 10, PERSON, 0.2), (20, 0, 30, 10, PERSON, 0.5), (40, 0, 50, 10, PERSON, 0.9)]
    detector, _ = make_detector(rows, threshold)
    dets = detector.detect(frame())
    assert len(dets) == n_kept and all(d.score >= threshold for d in dets)


def test_threshold_passed_to_model_instead_of_builtin_default(make_detector):
    detector, model = make_detector([], 0.37)
    assert model.kwargs["box_score_thresh"] == 0.37
    assert model.kwargs["weights"] is FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    assert detector.provenance["score_threshold"] == 0.37


@pytest.mark.parametrize("bad", [True, False, "0.5", None, math.nan, math.inf, -math.inf, -0.01, 1.01, [0.5]])
def test_malformed_threshold_rejected(bad, make_detector):
    with pytest.raises(ValueError, match="score_threshold"):
        check_score_threshold(bad)
    with pytest.raises(ValueError, match="score_threshold"):
        make_detector([], bad)


def test_threshold_is_required():
    with pytest.raises(TypeError):
        TorchvisionPersonDetector()  # no default threshold


@pytest.mark.parametrize("ok", [0, 1, 0.0, 1.0, 0.5, np.float32(0.25), np.float64(0.75)])
def test_valid_thresholds_accepted(ok):
    assert check_score_threshold(ok) == float(ok)


# --------------------------------------------------------------------------- input handling


def test_bgr_converted_to_rgb_float(make_detector):
    detector, model = make_detector([], 0.5)
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30  # B, G, R
    detector.detect(img)
    (tensor,) = model.inputs[0]
    assert tensor.shape == (3, 4, 6) and tensor.dtype == torch.float32
    expected = torch.tensor([30, 20, 10], dtype=torch.float32) / 255  # R, G, B
    assert torch.allclose(tensor[:, 0, 0], expected)
    assert torch.allclose(tensor, expected[:, None, None].expand(3, 4, 6))


def test_input_keeps_original_size(make_detector):
    detector, model = make_detector([], 0.5)
    detector.detect(frame(123, 457))
    assert model.inputs[0][0].shape == (3, 123, 457)


def test_caller_frame_unchanged(make_detector):
    detector, model = make_detector([(0, 0, 10, 10, PERSON, 0.9)], 0.5)
    img = frame()
    before = img.copy()
    detector.detect(img)
    np.testing.assert_array_equal(img, before)
    model.inputs[0][0].fill_(0)  # mutating what the model received does not reach the caller
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize("bad", [
    np.zeros((10, 10), np.uint8),
    np.zeros((10, 10, 4), np.uint8),
    np.zeros((10, 10, 1), np.uint8),
    np.zeros((10, 10, 3), np.float32),
    np.zeros((0, 10, 3), np.uint8),
    np.zeros((10, 0, 3), np.uint8),
    [[0, 0, 0]],
    None,
], ids=["2d", "4ch", "1ch", "float", "no-rows", "no-cols", "list", "none"])
def test_malformed_frame_rejected(make_detector, bad):
    detector, model = make_detector([], 0.5)
    with pytest.raises(ValueError):
        detector.detect(bad)
    assert model.inputs == []


# --------------------------------------------------------------------------- boxes


def test_box_converted_to_integer_xywh(make_detector):
    detector, _ = make_detector([(10.2, 20.7, 50.5, 100.1, PERSON, 0.9)], 0.5)
    (det,) = detector.detect(frame())
    # smallest integer box containing the float box: x 10..51, y 20..101
    assert det.bbox == (10, 20, 41, 81)
    assert all(type(v) is int for v in det.bbox)
    assert type(det.score) is float


@pytest.mark.parametrize("xyxy, expected", [
    ((0, 0, 10, 10), (0, 0, 10, 10)),
    ((-5.5, -3, 10, 10), (0, 0, 10, 10)),  # clipped at top-left
    ((630, 470, 700, 900), (630, 470, 10, 10)),  # clipped at bottom-right
    ((-10, -10, 1000, 1000), (0, 0, W, H)),  # larger than frame
    ((639.5, 479.5, 640, 480), (639, 479, 1, 1)),  # sub-pixel at corner
    ((5, 5, 5.25, 5.75), (5, 5, 1, 1)),  # sub-pixel inside
])
def test_clipping(xyxy, expected):
    bbox = xyxy_to_clipped_xywh(*xyxy, W, H)
    assert bbox == expected
    x, y, w, h = bbox
    assert x >= 0 and y >= 0 and w >= 1 and h >= 1 and x + w <= W and y + h <= H


@pytest.mark.parametrize("xyxy", [
    (10, 10, 10, 20),  # zero width
    (10, 10, 20, 10),  # zero height
    (20, 10, 10, 20),  # inverted
    (-50, 10, -1, 20),  # entirely left of frame
    (700, 10, 800, 20),  # entirely right
    (10, 480, 20, 500),  # starts at bottom edge
    (math.nan, 0, 10, 10),
    (0, 0, math.inf, 10),
])
def test_degenerate_or_outside_boxes_dropped(xyxy):
    assert xyxy_to_clipped_xywh(*xyxy, W, H) is None
    assert select_person_detections([xyxy], [PERSON], [0.9], PERSON, 0.5, W, H) == []


def test_all_boxes_inside_frame_for_random_output():
    rng = np.random.default_rng(1)
    boxes = np.sort(rng.uniform(-200, 900, size=(500, 2, 2)), axis=1).transpose(0, 2, 1).reshape(-1, 4)[:, [0, 2, 1, 3]]
    dets = select_person_detections(boxes, [PERSON] * 500, rng.uniform(0, 1, 500), PERSON, 0.0, W, H)
    assert dets
    for d in dets:
        x, y, w, h = d.bbox
        assert 0 <= x and 0 <= y and w >= 1 and h >= 1 and x + w <= W and y + h <= H


def test_mismatched_model_output_rejected():
    with pytest.raises(ValueError, match="Mismatched"):
        select_person_detections([(0, 0, 1, 1)], [PERSON, PERSON], [0.9], PERSON, 0.5, W, H)


# --------------------------------------------------------------------------- determinism / ordering


def test_sorted_by_score_then_box_and_deterministic(make_detector):
    rows = [
        (100, 0, 110, 10, PERSON, 0.6),
        (50, 0, 60, 10, PERSON, 0.9),
        (30, 0, 40, 10, PERSON, 0.6),  # same score as the first: ordered by box
        (0, 0, 10, 10, PERSON, 0.75),
    ]
    detector, _ = make_detector(rows, 0.5)
    first = detector.detect(frame())
    assert [d.bbox[0] for d in first] == [50, 0, 30, 100]
    assert detector.detect(frame()) == first

    shuffled, _ = make_detector(list(reversed(rows)), 0.5)
    assert shuffled.detect(frame()) == first


# --------------------------------------------------------------------------- output shape / scope


def test_detection_has_only_box_and_score():
    assert [f.name for f in dataclasses.fields(PersonDetection)] == ["bbox", "score"]


def test_detection_is_immutable():
    det = PersonDetection((0, 0, 1, 1), 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        det.score = 1.0


def test_no_state_between_frames(make_detector):
    # Detection is per frame: no track ids, and calling twice gives independent equal results.
    detector, model = make_detector([(0, 0, 10, 10, PERSON, 0.9)], 0.5)
    a, b = detector.detect(frame()), detector.detect(frame())
    assert a == b and a is not b
    assert len(model.inputs) == 2


def test_satisfies_protocol(make_detector):
    detector, _ = make_detector([], 0.5)

    def run(d: PersonDetector):
        return d.detect(frame())

    assert run(detector) == []


def test_provenance_recorded(make_detector):
    detector, _ = make_detector([], 0.5)
    p = detector.provenance
    assert p["weights"] == "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"
    assert p["torch_version"] == torch.__version__
    assert p["weights_url"].startswith("https://download.pytorch.org/models/")
    assert "not commercially cleared" in p["training_data"]
    assert p["weights_file"] == "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    assert "weights_cache_path" not in p


def test_provenance_has_no_local_path(make_detector):
    detector, _ = make_detector([], 0.5)
    text = repr(detector.provenance)
    for local in (str(Path.home()), torch.hub.get_dir(), Path(torch.hub.get_dir()).as_posix()):
        assert local not in text
    for value in detector.provenance.values():
        if isinstance(value, str):
            assert not Path(value).is_absolute() and not re.match(r"[A-Za-z]:[\/]", value)


def test_weights_sha256_of_cached_file(make_detector):
    detector, _ = make_detector([], 0.5)
    digest = detector.provenance["weights_sha256"]
    cached = Path(torch.hub.get_dir()) / "checkpoints" / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    if cached.is_file():
        assert digest == hashlib.sha256(cached.read_bytes()).hexdigest()
        assert digest.startswith("dd69338a")  # torchvision's filename hash prefix
    else:
        assert digest is None  # only possible with an injected model; the real model loads this file


# --------------------------------------------------------------------------- real model (cached weights only)

_WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
_CACHED = Path(torch.hub.get_dir()) / "checkpoints" / Path(_WEIGHTS.url).name


@pytest.mark.skipif(not _CACHED.is_file(), reason=f"pretrained weights not cached at {_CACHED}; not downloading in tests")
def test_real_model_smoke():
    detector = TorchvisionPersonDetector(0.5)
    img = np.full((240, 320, 3), 127, np.uint8)
    dets = detector.detect(img)
    assert isinstance(dets, list)
    for d in dets:
        x, y, w, h = d.bbox
        assert 0 <= x and 0 <= y and w >= 1 and h >= 1 and x + w <= 320 and y + h <= 240
        assert 0.5 <= d.score <= 1
