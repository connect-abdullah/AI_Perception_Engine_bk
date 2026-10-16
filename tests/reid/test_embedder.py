"""Embedder tests. Use the real pretrained weights (downloaded once to the cache, ~17 MB)."""

import numpy as np
import pytest
from PIL import Image

from reid.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder(device="cpu")


@pytest.fixture
def bgr_crop():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(300, 120, 3), dtype=np.uint8)


def test_records_model(embedder):
    assert embedder.model_name == "osnet_x1_0_msmt17"
    assert embedder.weights_path.is_file()
    assert embedder.embedding_dim == 512


def test_embedding_shape_and_dtype(embedder, bgr_crop):
    emb = embedder.embed(bgr_crop)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32


def test_embedding_is_finite(embedder, bgr_crop):
    assert np.all(np.isfinite(embedder.embed(bgr_crop)))


def test_embedding_has_unit_norm(embedder, bgr_crop):
    assert np.linalg.norm(embedder.embed(bgr_crop)) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic(embedder, bgr_crop):
    np.testing.assert_array_equal(embedder.embed(bgr_crop), embedder.embed(bgr_crop))


def test_path_and_bgr_array_give_same_embedding(embedder, bgr_crop, tmp_path):
    path = tmp_path / "crop.png"
    Image.fromarray(bgr_crop[:, :, ::-1]).save(path)  # PNG is lossless; saved as RGB
    np.testing.assert_allclose(embedder.embed(path), embedder.embed(bgr_crop), atol=1e-5)


def test_missing_file_raises(embedder, tmp_path):
    with pytest.raises(FileNotFoundError):
        embedder.embed(tmp_path / "does_not_exist.jpg")


def test_non_image_file_raises(embedder, tmp_path):
    path = tmp_path / "not_an_image.jpg"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Not a readable image"):
        embedder.embed(path)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((100, 50), dtype=np.uint8),  # grayscale
        np.zeros((100, 50, 4), dtype=np.uint8),  # 4 channels
        np.zeros((0, 50, 3), dtype=np.uint8),  # empty
        np.zeros((100, 50, 3), dtype=np.float32),  # wrong dtype
    ],
)
def test_bad_array_raises(embedder, bad):
    with pytest.raises(ValueError):
        embedder.embed(bad)


def test_unsupported_type_raises(embedder):
    with pytest.raises(TypeError):
        embedder.embed(12345)
