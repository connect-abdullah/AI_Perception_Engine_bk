"""Tracklet aggregation tests. Use synthetic vectors and a fake embedder; no model weights needed."""

import zlib
from pathlib import Path

import numpy as np
import pytest

from reid.similarity import cosine_similarity
from reid.tracklet import aggregate_embeddings, embed_tracklet


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@pytest.fixture
def crops():
    rng = np.random.default_rng(0)
    return [rng.normal(size=512) for _ in range(7)]


class FakeEmbedder:
    """Per-identity base vector plus per-crop noise, keyed on the parent folder and file name."""

    def __init__(self, noise=0.3, dim=32):
        self.noise = noise
        self.dim = dim
        self.calls = 0

    def embed(self, image):
        self.calls += 1
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        base = np.random.default_rng(zlib.crc32(path.parent.name.encode())).normal(size=self.dim)
        jitter = np.random.default_rng(zlib.crc32(path.name.encode())).normal(size=self.dim)
        return unit(base + self.noise * jitter).astype(np.float32)


def make_track(root: Path, identity: str, n: int) -> list[Path]:
    folder = root / identity
    folder.mkdir(parents=True)
    paths = [folder / f"{identity}_crop{k}.jpg" for k in range(n)]
    for p in paths:
        p.write_bytes(b"")  # contents unused by FakeEmbedder
    return paths


# --------------------------------------------------------------------------- aggregation


def test_single_embedding_returns_its_normalised_form():
    v = np.array([3.0, 4.0, 0.0])
    np.testing.assert_allclose(aggregate_embeddings([v]), [0.6, 0.8, 0.0], atol=1e-7)


def test_multiple_embeddings_aggregate_to_normalised_mean():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    np.testing.assert_allclose(aggregate_embeddings([a, b]), [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-7)


def test_matches_reference_formula(crops):
    expected = unit(np.mean([unit(c) for c in crops], axis=0))
    np.testing.assert_allclose(aggregate_embeddings(crops), expected, atol=1e-6)


def test_each_crop_has_equal_weight_regardless_of_input_scale():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    np.testing.assert_allclose(aggregate_embeddings([100.0 * a, b]), aggregate_embeddings([a, b]), atol=1e-7)


def test_output_is_finite_unit_norm_float32_1d(crops):
    out = aggregate_embeddings(crops)
    assert out.shape == (512,) and out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("dim", [2, 128, 768, 2048])
def test_works_for_any_dimension(dim):
    rng = np.random.default_rng(dim)
    out = aggregate_embeddings([rng.normal(size=dim) for _ in range(3)])
    assert out.shape == (dim,)


def test_accepts_2d_array(crops):
    np.testing.assert_array_equal(aggregate_embeddings(np.stack(crops)), aggregate_embeddings(crops))


def test_input_order_does_not_change_result(crops):
    reference = aggregate_embeddings(crops)
    rng = np.random.default_rng(1)
    for _ in range(10):
        shuffled = [crops[k] for k in rng.permutation(len(crops))]
        np.testing.assert_array_equal(aggregate_embeddings(shuffled), reference)  # bit-identical


def test_deterministic(crops):
    np.testing.assert_array_equal(aggregate_embeddings(crops), aggregate_embeddings(crops))


def test_does_not_modify_inputs(crops):
    before = [c.copy() for c in crops]
    aggregate_embeddings(crops)
    for c, b in zip(crops, before):
        np.testing.assert_array_equal(c, b)


def test_empty_input_fails():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_embeddings([])


def test_mismatched_dimensions_fail():
    with pytest.raises(ValueError, match="mismatched dimensions"):
        aggregate_embeddings([np.ones(512), np.ones(256)])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_fails(bad):
    v = np.ones(4)
    v[2] = bad
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_embeddings([np.ones(4), v])


def test_zero_vector_fails():
    with pytest.raises(ValueError, match="zero vector"):
        aggregate_embeddings([np.ones(4), np.zeros(4)])


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.float64(1.0), np.array([])])
def test_non_vector_embedding_fails(bad):
    with pytest.raises(ValueError, match="1-D"):
        aggregate_embeddings([bad])


def test_cancelling_embeddings_fail():
    with pytest.raises(ValueError, match="cancel out"):
        aggregate_embeddings([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])


# --------------------------------------------------------------------------- embed_tracklet


def test_embed_tracklet_with_fake_embedder(tmp_path):
    track = make_track(tmp_path, "P001", 5)
    embedder = FakeEmbedder()
    out = embed_tracklet(track, embedder)

    assert embedder.calls == 5
    assert out.shape == (32,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(out, aggregate_embeddings([FakeEmbedder().embed(p) for p in track]), atol=1e-7)
    np.testing.assert_array_equal(embed_tracklet(list(reversed(track)), FakeEmbedder()), out)


def test_tracklet_embedding_is_closer_to_identity_than_typical_single_crop(tmp_path):
    # Plumbing sanity check on synthetic data: averaging reduces per-crop noise.
    track = make_track(tmp_path, "P001", 8)
    embedder = FakeEmbedder(noise=1.0)
    identity_direction = np.random.default_rng(zlib.crc32(b"P001")).normal(size=32)
    single = [cosine_similarity(embedder.embed(p), identity_direction) for p in track]
    assert cosine_similarity(embed_tracklet(track, embedder), identity_direction) > np.median(single)


def test_embed_tracklet_accepts_generator(tmp_path):
    track = make_track(tmp_path, "P002", 3)
    out = embed_tracklet((p for p in track), FakeEmbedder())
    assert out.shape == (32,)


def test_embed_tracklet_empty_fails():
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="no images"):
        embed_tracklet([], embedder)
    assert embedder.calls == 0


def test_embed_tracklet_propagates_bad_crop_error(tmp_path):
    track = make_track(tmp_path, "P003", 2) + [tmp_path / "missing.jpg"]
    with pytest.raises(FileNotFoundError):
        embed_tracklet(track, FakeEmbedder())
