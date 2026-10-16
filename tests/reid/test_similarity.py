import numpy as np
import pytest

from reid.similarity import cosine_similarity


@pytest.fixture
def vectors():
    rng = np.random.default_rng(0)
    return rng.normal(size=512), rng.normal(size=512)


def test_self_similarity_is_one(vectors):
    a, _ = vectors
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_symmetric(vectors):
    a, b = vectors
    assert cosine_similarity(a, b) == cosine_similarity(b, a)


def test_deterministic(vectors):
    a, b = vectors
    assert cosine_similarity(a, b) == cosine_similarity(a.copy(), b.copy())


def test_known_values():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_similarity(np.array([1.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(1 / np.sqrt(2))


def test_scale_invariant(vectors):
    a, b = vectors
    assert cosine_similarity(3.0 * a, b) == pytest.approx(cosine_similarity(a, b))


def test_returns_python_float(vectors):
    assert isinstance(cosine_similarity(*vectors), float)


@pytest.mark.parametrize(
    "a, b",
    [
        (np.zeros(4), np.ones(4)),  # zero vector
        (np.ones(4), np.ones(5)),  # shape mismatch
        (np.ones((2, 2)), np.ones((2, 2))),  # not 1-D
        (np.array([1.0, np.nan]), np.ones(2)),  # non-finite
    ],
)
def test_invalid_inputs_raise(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)
