"""Evaluator tests. Use a fake embedder, so no model weights or real images are needed."""

import csv
import zlib
from pathlib import Path

import numpy as np
import pytest

from reid.evaluate import (
    count_diff_pairs,
    discover_dataset,
    evaluate,
    format_report,
    leave_one_out_rank1,
    roc_auc,
    sample_diff_pairs,
    same_pair_indices,
    summarize,
    write_pairs_csv,
)


class FakeEmbedder:
    """Identity-specific base vector plus small per-image noise, keyed on the parent folder name."""

    def __init__(self, noise=0.1, dim=16):
        self.noise = noise
        self.dim = dim

    def embed(self, image):
        path = Path(image)
        base = np.random.default_rng(zlib.crc32(path.parent.name.encode())).normal(size=self.dim)
        jitter = np.random.default_rng(zlib.crc32(path.name.encode())).normal(size=self.dim)
        return (base + self.noise * jitter).astype(np.float32)


def make_dataset(root: Path, layout: dict[str, int]) -> Path:
    for identity, n in layout.items():
        folder = root / identity
        folder.mkdir(parents=True)
        for k in range(n):
            (folder / f"img{k}.jpg").write_bytes(b"")  # contents unused by FakeEmbedder
    return root


# --------------------------------------------------------------------------- discovery


def test_discovery_sorted_and_labeled(tmp_path):
    make_dataset(tmp_path, {"P002": 2, "P001": 3})
    samples, skipped = discover_dataset(tmp_path)
    assert [(s.identity, s.path.name) for s in samples] == [
        ("P001", "img0.jpg"), ("P001", "img1.jpg"), ("P001", "img2.jpg"),
        ("P002", "img0.jpg"), ("P002", "img1.jpg"),
    ]
    assert skipped == []


def test_discovery_ignores_non_images_hidden_loose_and_nested(tmp_path):
    make_dataset(tmp_path, {"P001": 2, "P002": 1})
    (tmp_path / "P001" / "notes.txt").write_text("x")
    (tmp_path / "P001" / ".hidden.jpg").write_bytes(b"")
    (tmp_path / "P001" / "UPPER.PNG").write_bytes(b"")
    (tmp_path / "P001" / "nested").mkdir()
    (tmp_path / "P001" / "nested" / "deep.jpg").write_bytes(b"")
    (tmp_path / "loose.jpg").write_bytes(b"")
    (tmp_path / ".git").mkdir()
    samples, _ = discover_dataset(tmp_path)
    assert sorted(s.path.name for s in samples if s.identity == "P001") == ["UPPER.PNG", "img0.jpg", "img1.jpg"]
    assert {s.identity for s in samples} == {"P001", "P002"}


def test_discovery_skips_empty_identity_dirs(tmp_path):
    make_dataset(tmp_path, {"P001": 2, "P002": 1})
    (tmp_path / "P003").mkdir()
    _, skipped = discover_dataset(tmp_path)
    assert skipped == ["P003"]


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_dataset(tmp_path / "nope")


def test_root_is_file_raises(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        discover_dataset(f)


def test_empty_root_raises(tmp_path):
    with pytest.raises(ValueError, match="at least 2 identity"):
        discover_dataset(tmp_path)


def test_single_identity_raises(tmp_path):
    make_dataset(tmp_path, {"P001": 3})
    with pytest.raises(ValueError, match="at least 2 identity"):
        discover_dataset(tmp_path)


def test_no_identity_with_two_images_raises(tmp_path):
    make_dataset(tmp_path, {"P001": 1, "P002": 1, "P003": 1})
    with pytest.raises(ValueError, match="2\\+ images"):
        discover_dataset(tmp_path)


# --------------------------------------------------------------------------- pairs


LABELS = ["A", "A", "A", "B", "B", "C"]


def test_same_pairs_are_exactly_within_identity():
    assert same_pair_indices(LABELS) == [(0, 1), (0, 2), (1, 2), (3, 4)]


def test_count_diff_pairs():
    # 15 total pairs - 4 same pairs
    assert count_diff_pairs(LABELS) == 11


def test_all_diff_pairs_are_cross_identity_and_complete():
    pairs = sample_diff_pairs(LABELS, None)
    assert len(pairs) == 11
    assert all(LABELS[i] != LABELS[j] and i < j for i, j in pairs)


@pytest.mark.parametrize("n", [1, 3, 5, 8, 10])  # covers both sparse and dense sampling paths
def test_sampling_is_deterministic_distinct_and_cross_identity(n):
    first = sample_diff_pairs(LABELS, n, seed=42)
    assert first == sample_diff_pairs(LABELS, n, seed=42)
    assert len(first) == n == len(set(first))
    assert all(LABELS[i] != LABELS[j] and i < j for i, j in first)


def test_sampling_seed_changes_selection():
    labels = [str(k // 3) for k in range(60)]  # 20 identities x 3 images
    assert sample_diff_pairs(labels, 50, seed=0) != sample_diff_pairs(labels, 50, seed=1)


def test_sampling_caps_at_total():
    assert len(sample_diff_pairs(LABELS, 1000)) == 11


# --------------------------------------------------------------------------- metrics


def test_roc_auc_known_values():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    # positives 0.8, 0.4 vs negatives 0.6, 0.2: 3 of 4 orderings correct
    assert roc_auc([0.8, 0.4], [0.6, 0.2]) == pytest.approx(0.75)
    # one tie (0.5 vs 0.5) counts as half: (1 + 0.5) / 2
    assert roc_auc([0.9, 0.5], [0.5]) == pytest.approx(0.75)


def test_roc_auc_matches_brute_force():
    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(0.6, 0.1, 50), 2)  # rounding forces ties
    neg = np.round(rng.normal(0.4, 0.1, 80), 2)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert roc_auc(pos, neg) == pytest.approx(brute)


def test_roc_auc_empty_raises():
    with pytest.raises(ValueError):
        roc_auc([], [0.1])


def test_summarize():
    stats = summarize(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["median"] == pytest.approx(0.5)
    assert stats["min"] == 0.0 and stats["max"] == 1.0
    assert stats["p25"] == pytest.approx(0.25)
    assert stats["std"] == pytest.approx(np.std([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_rank1():
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.7, 0.7]])
    labels = ["A", "A", "B", "B", "C"]  # C has a single image -> not a query
    assert leave_one_out_rank1(emb, labels) == (1.0, 4)
    acc, queries = leave_one_out_rank1(emb, ["A", "B", "A", "B", "C"])
    assert (acc, queries) == (0.0, 4)


# --------------------------------------------------------------------------- end to end


def test_evaluate_end_to_end_separable(tmp_path):
    make_dataset(tmp_path, {"P001": 4, "P002": 3, "P003": 3, "P004": 1})
    result = evaluate(tmp_path, FakeEmbedder(noise=0.1), diff_ratio=2.0, seed=0)

    assert len(result.samples) == 11
    assert len(result.same_pairs) == 6 + 3 + 3
    assert len(result.diff_pairs) == 24
    assert result.total_diff_pairs == 55 - 12
    assert all(p.same for p in result.same_pairs) and not any(p.same for p in result.diff_pairs)
    assert all(result.samples[p.a].identity == result.samples[p.b].identity for p in result.same_pairs)
    assert all(result.samples[p.a].identity != result.samples[p.b].identity for p in result.diff_pairs)
    assert result.roc_auc == 1.0
    assert result.rank1 == 1.0 and result.rank1_queries == 10
    assert result.same_stats["min"] > result.diff_stats["max"]


def test_evaluate_is_deterministic(tmp_path):
    make_dataset(tmp_path, {"P001": 3, "P002": 3, "P003": 3})
    r1 = evaluate(tmp_path, FakeEmbedder(noise=2.0), seed=7)
    r2 = evaluate(tmp_path, FakeEmbedder(noise=2.0), seed=7)
    assert r1.diff_pairs == r2.diff_pairs and r1.same_pairs == r2.same_pairs
    assert r1.roc_auc == r2.roc_auc


def test_evaluate_all_diff(tmp_path):
    make_dataset(tmp_path, {"P001": 2, "P002": 2, "P003": 2})
    result = evaluate(tmp_path, FakeEmbedder(), diff_ratio=None)
    assert len(result.diff_pairs) == result.total_diff_pairs == 12


def test_report_and_csv(tmp_path):
    data = make_dataset(tmp_path / "data", {"P001": 3, "P002": 2})
    result = evaluate(data, FakeEmbedder())
    report = format_report(result, top_k=2, model_name="fake")
    assert "ROC-AUC" in report and "Highest-scoring DIFFERENT" in report and "Lowest-scoring SAME" in report

    out = tmp_path / "pairs.csv"
    write_pairs_csv(result, out)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.same_pairs) + len(result.diff_pairs)
    assert all((r["identity_a"] == r["identity_b"]) == (r["same_person"] == "1") for r in rows)


def test_embedder_returning_wrong_shape_fails(tmp_path):
    class Bad:
        def embed(self, image):
            return np.zeros((2, 2))

    make_dataset(tmp_path, {"P001": 2, "P002": 1})
    with pytest.raises(ValueError, match="1-D"):
        evaluate(tmp_path, Bad())
