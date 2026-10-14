"""Identity registry tests. Synthetic embeddings only; no model weights or images needed."""

import numpy as np
import pytest

from reid.identity import Candidate, IdentityRegistry
from reid.tracklet import aggregate_embeddings


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def axis(k, dim=4):
    v = np.zeros(dim)
    v[k] = 1.0
    return v


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------------------- codes


def test_first_identity_is_p001():
    assert IdentityRegistry().create_identity(axis(0)) == "P001"


def test_codes_are_sequential():
    reg = IdentityRegistry()
    codes = [reg.create_identity(axis(k % 4)) for k in range(12)]
    assert codes == [f"P{n:03d}" for n in range(1, 13)]
    assert reg.person_codes == codes
    assert len(reg) == 12


def test_codes_continue_past_999(rng):
    reg = IdentityRegistry()
    for _ in range(1000):
        last = reg.create_identity(rng.normal(size=3))
    assert last == "P1000"
    assert reg.get_identity("P1000").tracklet_count == 1


def test_two_registries_are_independent():
    a, b = IdentityRegistry(), IdentityRegistry()
    a.create_identity(axis(0))
    assert b.create_identity(axis(1)) == "P001"


@pytest.mark.parametrize("bad", ["p001", "P01", "P", "001", "PX01", "P001 ", "", "Alice", 1, None])
def test_invalid_person_code_fails_clearly(bad):
    reg = IdentityRegistry()
    reg.create_identity(axis(0))
    with pytest.raises(ValueError, match="person code"):
        reg.get_identity(bad)
    with pytest.raises(ValueError, match="person code"):
        reg.add_tracklet(bad, axis(0))


def test_unknown_person_code_fails_clearly():
    reg = IdentityRegistry()
    reg.create_identity(axis(0))
    with pytest.raises(KeyError, match="P002"):
        reg.get_identity("P002")
    with pytest.raises(KeyError, match="P002"):
        reg.add_tracklet("P002", axis(0))


def test_contains():
    reg = IdentityRegistry()
    reg.create_identity(axis(0))
    assert "P001" in reg and "P002" not in reg


# --------------------------------------------------------------------------- embeddings


def test_stored_embedding_is_normalised_readonly_float32():
    reg = IdentityRegistry()
    code = reg.create_identity(np.array([3.0, 4.0]))
    record = reg.get_identity(code)
    np.testing.assert_allclose(record.embedding, [0.6, 0.8], atol=1e-7)
    assert record.embedding.dtype == np.float32
    assert record.tracklet_count == 1
    with pytest.raises(ValueError):
        record.embedding[0] = 1.0


def test_caller_mutation_does_not_affect_registry():
    reg = IdentityRegistry()
    v = np.array([1.0, 0.0])
    reg.create_identity(v)
    v[:] = [0.0, 1.0]
    np.testing.assert_allclose(reg.get_identity("P001").embedding, [1.0, 0.0])


@pytest.mark.parametrize("dim", [2, 256, 512, 768])
def test_any_dimension(dim, rng):
    reg = IdentityRegistry()
    reg.create_identity(rng.normal(size=dim))
    assert reg.embedding_dim == dim
    assert reg.get_identity("P001").embedding.shape == (dim,)


def test_incompatible_dimensions_fail():
    reg = IdentityRegistry()
    reg.create_identity(np.ones(4))
    for action in (
        lambda: reg.create_identity(np.ones(5)),
        lambda: reg.add_tracklet("P001", np.ones(3)),
        lambda: reg.rank_candidates(np.ones(8)),
    ):
        with pytest.raises(ValueError, match="dimension"):
            action()
    assert len(reg) == 1 and reg.get_identity("P001").tracklet_count == 1


def test_fixed_dimension_up_front():
    reg = IdentityRegistry(embedding_dim=3)
    with pytest.raises(ValueError, match="dimension"):
        reg.create_identity(np.ones(4))
    with pytest.raises(ValueError, match="dimension"):
        reg.rank_candidates(np.ones(4))  # checked even while empty
    with pytest.raises(ValueError):
        IdentityRegistry(embedding_dim=0)


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros(4), "zero vector"),
        (np.array([1.0, np.nan, 0.0, 0.0]), "non-finite"),
        (np.array([1.0, np.inf, 0.0, 0.0]), "non-finite"),
        (np.ones((2, 2)), "1-D"),
        (np.array([]), "1-D"),
    ],
)
def test_invalid_embeddings_fail(bad, message):
    reg = IdentityRegistry()
    with pytest.raises(ValueError, match=message):
        reg.create_identity(bad)
    reg.create_identity(np.ones(4))
    with pytest.raises(ValueError, match=message):
        reg.add_tracklet("P001", bad)
    with pytest.raises(ValueError, match=message):
        reg.rank_candidates(bad)
    assert len(reg) == 1 and reg.get_identity("P001").tracklet_count == 1


# --------------------------------------------------------------------------- ranking


def test_ranking_order_and_similarity_values():
    reg = IdentityRegistry()
    reg.create_identity(np.array([1.0, 0.0, 0.0]))  # P001
    reg.create_identity(np.array([0.0, 1.0, 0.0]))  # P002
    reg.create_identity(np.array([1.0, 1.0, 0.0]))  # P003
    query = np.array([2.0, 0.5, 0.0])

    ranked = reg.rank_candidates(query)
    assert [c.person_code for c in ranked] == ["P001", "P003", "P002"]
    q = unit(query)
    expected = {"P001": q @ axis(0, 3), "P002": q @ axis(1, 3), "P003": q @ unit([1, 1, 0])}
    for c in ranked:
        assert isinstance(c, Candidate)
        assert c.similarity == pytest.approx(expected[c.person_code], abs=1e-6)


def test_ranking_is_sorted_descending_on_random_data(rng):
    reg = IdentityRegistry()
    for _ in range(20):
        reg.create_identity(rng.normal(size=64))
    sims = [c.similarity for c in reg.rank_candidates(rng.normal(size=64))]
    assert sims == sorted(sims, reverse=True)
    assert len(sims) == 20


def test_ties_broken_by_creation_order():
    reg = IdentityRegistry()
    for _ in range(3):
        reg.create_identity(axis(1))  # P001..P003 identical
    reg.create_identity(axis(0))  # P004 best match
    ranked = reg.rank_candidates(axis(0))
    assert [c.person_code for c in ranked] == ["P004", "P001", "P002", "P003"]
    ranked_tied = reg.rank_candidates(axis(1))
    assert [c.person_code for c in ranked_tied[:3]] == ["P001", "P002", "P003"]


def test_ties_use_numeric_not_string_order(rng):
    reg = IdentityRegistry()
    for _ in range(1000):
        reg.create_identity(axis(1))  # P001..P1000, all identical
    codes = [c.person_code for c in reg.rank_candidates(axis(1))]
    assert codes[:2] == ["P001", "P002"] and codes[-1] == "P1000"


def test_ranking_is_deterministic(rng):
    reg = IdentityRegistry()
    for _ in range(10):
        reg.create_identity(rng.normal(size=16))
    q = rng.normal(size=16)
    assert reg.rank_candidates(q) == reg.rank_candidates(q)


def test_top_k():
    reg = IdentityRegistry()
    for k in range(4):
        reg.create_identity(axis(k))
    query = np.array([4.0, 3.0, 2.0, 1.0])
    assert [c.person_code for c in reg.rank_candidates(query, top_k=2)] == ["P001", "P002"]
    assert len(reg.rank_candidates(query, top_k=10)) == 4
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="top_k"):
            reg.rank_candidates(query, top_k=bad)


def test_empty_registry_search_returns_empty():
    reg = IdentityRegistry()
    assert reg.rank_candidates(np.ones(4)) == []
    assert len(reg) == 0


def test_search_never_creates_or_updates_identities(rng):
    reg = IdentityRegistry()
    assert reg.rank_candidates(rng.normal(size=8)) == []
    assert len(reg) == 0 and reg.embedding_dim is None  # empty search does not even fix the dimension

    reg.create_identity(axis(0, 8))
    before = reg.get_identity("P001").embedding.copy()
    for _ in range(5):
        reg.rank_candidates(-axis(0, 8))  # very dissimilar query
        reg.rank_candidates(axis(0, 8))  # perfect match
    assert reg.person_codes == ["P001"]
    assert reg.get_identity("P001").tracklet_count == 1
    np.testing.assert_array_equal(reg.get_identity("P001").embedding, before)


# --------------------------------------------------------------------------- updates


def test_explicit_update_changes_representative_and_count():
    reg = IdentityRegistry()
    reg.create_identity(axis(0, 2))
    record = reg.add_tracklet("P001", axis(1, 2))
    assert record.tracklet_count == 2
    np.testing.assert_allclose(record.embedding, unit([1, 1]), atol=1e-7)
    assert reg.get_identity("P001").tracklet_count == 2


def test_update_only_affects_target_identity():
    reg = IdentityRegistry()
    reg.create_identity(axis(0))
    reg.create_identity(axis(1))
    reg.add_tracklet("P001", axis(2))
    np.testing.assert_allclose(reg.get_identity("P002").embedding, axis(1))
    assert reg.get_identity("P002").tracklet_count == 1


def test_multiple_updates_equal_normalised_mean_of_all_tracklets(rng):
    tracklets = [rng.normal(size=128) * rng.uniform(0.1, 10) for _ in range(6)]  # varied scales
    reg = IdentityRegistry()
    reg.create_identity(tracklets[0])
    for t in tracklets[1:]:
        reg.add_tracklet("P001", t)

    record = reg.get_identity("P001")
    assert record.tracklet_count == 6
    expected = unit(np.mean([unit(t) for t in tracklets], axis=0))
    np.testing.assert_allclose(record.embedding, expected, atol=1e-6)
    # same definition as the tracklet aggregator: every tracklet counts equally
    np.testing.assert_allclose(record.embedding, aggregate_embeddings(tracklets), atol=1e-6)


def test_update_order_does_not_matter(rng):
    tracklets = [rng.normal(size=32) for _ in range(5)]
    results = []
    for order in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 4, 1, 3]):
        reg = IdentityRegistry()
        reg.create_identity(tracklets[order[0]])
        for k in order[1:]:
            reg.add_tracklet("P001", tracklets[k])
        results.append(reg.get_identity("P001").embedding)
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], atol=1e-6)


def test_old_identity_is_not_dominated_by_new_tracklet():
    # After 3 tracklets along x, one tracklet along y must carry weight 1/4, not 1/2.
    reg = IdentityRegistry()
    reg.create_identity(axis(0, 2))
    reg.add_tracklet("P001", axis(0, 2))
    reg.add_tracklet("P001", axis(0, 2))
    record = reg.add_tracklet("P001", axis(1, 2))
    np.testing.assert_allclose(record.embedding, unit([3, 1]), atol=1e-7)


def test_cancelling_update_is_rejected_without_side_effects():
    reg = IdentityRegistry()
    reg.create_identity(axis(0, 2))
    with pytest.raises(ValueError, match="cancel"):
        reg.add_tracklet("P001", -axis(0, 2))
    record = reg.get_identity("P001")
    assert record.tracklet_count == 1
    np.testing.assert_allclose(record.embedding, axis(0, 2))
