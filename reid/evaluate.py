"""Evaluate a ReID embedder on a labeled folder of person crops.

Folder layout (folder name = ground-truth identity, used for evaluation only):

    eval_data/
      P001/ img1.jpg img2.jpg ...
      P002/ ...

Usage:

    python -m reid.evaluate eval_data [--csv pairs.csv] [--diff-ratio 1.0 | --all-diff] [--seed 0]

Reports same-person vs different-person cosine score distributions, ROC-AUC,
leave-one-out rank-1 and the hardest pairs. It reports observed scores only and
does not choose or apply any identity threshold.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Protocol

import numpy as np

from reid.embedder import Embedder
from reid.similarity import cosine_similarity

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERCENTILES = (1, 5, 25, 75, 95, 99)


class SupportsEmbed(Protocol):
    def embed(self, image) -> np.ndarray: ...


@dataclass(frozen=True)
class Sample:
    identity: str
    path: Path


@dataclass(frozen=True)
class Pair:
    a: int  # index into samples
    b: int
    same: bool
    score: float


@dataclass
class EvalResult:
    samples: list[Sample]
    same_pairs: list[Pair]
    diff_pairs: list[Pair]
    total_diff_pairs: int
    same_stats: dict[str, float]
    diff_stats: dict[str, float]
    roc_auc: float
    rank1: float | None
    rank1_queries: int
    skipped_dirs: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- discovery


def discover_dataset(root: str | Path) -> tuple[list[Sample], list[str]]:
    """Return (samples sorted by identity then filename, identity dirs skipped for having no images).

    Only image files directly inside each identity folder are used; hidden entries,
    non-image files, loose files in the root and nested subfolders are ignored.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Evaluation folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Evaluation path is not a folder: {root}")

    samples: list[Sample] = []
    skipped: list[str] = []
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        images = sorted(
            p for p in identity_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            skipped.append(identity_dir.name)
            continue
        samples.extend(Sample(identity_dir.name, p) for p in images)

    identities = {s.identity for s in samples}
    if len(identities) < 2:
        raise ValueError(
            f"Need at least 2 identity folders containing images under {root}, found {len(identities)}"
        )
    if not any(sum(s.identity == i for s in samples) >= 2 for i in identities):
        raise ValueError(f"Need at least one identity with 2+ images under {root} to form same-person pairs")
    return samples, skipped


# --------------------------------------------------------------------------- pairs


def same_pair_indices(labels: list[str]) -> list[tuple[int, int]]:
    """All (i, j), i < j, where labels match."""
    return [(i, j) for i, j in combinations(range(len(labels)), 2) if labels[i] == labels[j]]


def count_diff_pairs(labels: list[str]) -> int:
    _, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    return int((n * (n - 1) - np.sum(counts * (counts - 1))) // 2)


def sample_diff_pairs(labels: list[str], n_pairs: int | None, seed: int = 0) -> list[tuple[int, int]]:
    """Uniformly sample distinct (i, j), i < j, with different labels. `None` returns all of them.

    Deterministic for a given (labels, n_pairs, seed). Result is sorted.
    """
    total = count_diff_pairs(labels)
    if n_pairs is None or n_pairs >= total:
        return [(i, j) for i, j in combinations(range(len(labels)), 2) if labels[i] != labels[j]]
    if n_pairs <= 0:
        return []

    rng = np.random.default_rng(seed)
    if n_pairs > total // 2:  # dense request: enumerate and choose, rather than rejection-sample
        all_pairs = [(i, j) for i, j in combinations(range(len(labels)), 2) if labels[i] != labels[j]]
        picked = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        return sorted(all_pairs[k] for k in picked)

    n = len(labels)
    chosen: set[tuple[int, int]] = set()
    while len(chosen) < n_pairs:
        i, j = rng.integers(0, n, size=2)
        if labels[i] == labels[j]:
            continue
        chosen.add((min(i, j), max(i, j)))
    return sorted((int(i), int(j)) for i, j in chosen)


# --------------------------------------------------------------------------- metrics


def summarize(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot summarise an empty score list")
    stats = {
        "count": float(scores.size),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(scores, p))
    return stats


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values share the mean of their ranks."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_vals[end + 1] == sorted_vals[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def roc_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """ROC-AUC = P(random positive scores higher than random negative), ties counted as 1/2.

    Computed via the Mann-Whitney U statistic; independent of class balance.
    """
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("ROC-AUC needs at least one positive and one negative score")
    ranks = _average_ranks(np.concatenate([pos, neg]))
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def leave_one_out_rank1(embeddings: np.ndarray, labels: list[str]) -> tuple[float | None, int]:
    """For every image whose identity has another image, is its nearest other image the same identity?

    Returns (accuracy, number of queries). Uses all images, not the sampled pairs.
    """
    labels_arr = np.asarray(labels)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / norms
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)

    hits = queries = 0
    for i in range(len(labels)):
        if np.sum(labels_arr == labels_arr[i]) < 2:
            continue
        queries += 1
        hits += int(labels_arr[int(np.argmax(sims[i]))] == labels_arr[i])
    return (hits / queries if queries else None), queries


# --------------------------------------------------------------------------- evaluation


def embed_samples(embedder: SupportsEmbed, samples: list[Sample]) -> np.ndarray:
    vectors = []
    for s in samples:
        vec = np.asarray(embedder.embed(s.path), dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Embedder returned shape {vec.shape} for {s.path}; expected 1-D")
        vectors.append(vec)
    return np.stack(vectors)


def evaluate(
    root: str | Path,
    embedder: SupportsEmbed,
    diff_ratio: float | None = 1.0,
    seed: int = 0,
) -> EvalResult:
    """Embed every image once, then score all same-person pairs and sampled different-person pairs.

    diff_ratio: number of different-person pairs sampled per same-person pair
                (None = use every different-person pair).
    """
    samples, skipped = discover_dataset(root)
    labels = [s.identity for s in samples]
    embeddings = embed_samples(embedder, samples)

    same_idx = same_pair_indices(labels)
    n_diff = None if diff_ratio is None else max(1, round(len(same_idx) * diff_ratio))
    diff_idx = sample_diff_pairs(labels, n_diff, seed=seed)

    def score(pairs, same):
        return [Pair(i, j, same, cosine_similarity(embeddings[i], embeddings[j])) for i, j in pairs]

    same_pairs = score(same_idx, True)
    diff_pairs = score(diff_idx, False)
    same_scores = np.array([p.score for p in same_pairs])
    diff_scores = np.array([p.score for p in diff_pairs])
    rank1, rank1_queries = leave_one_out_rank1(embeddings, labels)

    return EvalResult(
        samples=samples,
        same_pairs=same_pairs,
        diff_pairs=diff_pairs,
        total_diff_pairs=count_diff_pairs(labels),
        same_stats=summarize(same_scores),
        diff_stats=summarize(diff_scores),
        roc_auc=roc_auc(same_scores, diff_scores),
        rank1=rank1,
        rank1_queries=rank1_queries,
        skipped_dirs=skipped,
    )


# --------------------------------------------------------------------------- output


def write_pairs_csv(result: EvalResult, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["identity_a", "image_a", "identity_b", "image_b", "same_person", "cosine"])
        for p in result.same_pairs + result.diff_pairs:
            a, b = result.samples[p.a], result.samples[p.b]
            writer.writerow([a.identity, str(a.path), b.identity, str(b.path), int(p.same), f"{p.score:.6f}"])


def format_report(result: EvalResult, top_k: int = 10, model_name: str | None = None) -> str:
    s, d = result.same_stats, result.diff_stats
    n_ids = len({x.identity for x in result.samples})
    lines = []
    if model_name:
        lines.append(f"model: {model_name}")
    lines += [
        f"identities:            {n_ids}",
        f"images:                {len(result.samples)}",
        f"same-person pairs:     {len(result.same_pairs)} (all)",
        f"different-person pairs:{len(result.diff_pairs):>6} sampled of {result.total_diff_pairs} possible",
    ]
    if result.skipped_dirs:
        lines.append(f"skipped (no images):   {', '.join(result.skipped_dirs)}")

    lines += ["", f"{'cosine':<8}{'same':>10}{'different':>12}"]
    for key in ["mean", "median", "std", "min", *[f"p{p}" for p in PERCENTILES], "max"]:
        lines.append(f"{key:<8}{s[key]:>10.4f}{d[key]:>12.4f}")

    same_scores = np.array([p.score for p in result.same_pairs])
    diff_scores = np.array([p.score for p in result.diff_pairs])
    lines += [
        "",
        f"ROC-AUC:                                  {result.roc_auc:.4f}",
        f"same pairs scoring <= max different:      {np.mean(same_scores <= d['max']):.1%}",
        f"different pairs scoring >= min same:      {np.mean(diff_scores >= s['min']):.1%}",
    ]
    if result.rank1 is not None:
        lines.append(f"leave-one-out rank-1 (all images):        {result.rank1:.1%} over {result.rank1_queries} queries")

    def describe(pairs, title):
        out = ["", title]
        for p in pairs[:top_k]:
            a, b = result.samples[p.a], result.samples[p.b]
            out.append(f"  {p.score:.4f}  {a.identity}/{a.path.name}  <->  {b.identity}/{b.path.name}")
        return out

    lines += describe(sorted(result.diff_pairs, key=lambda p: -p.score), "Highest-scoring DIFFERENT-person pairs:")
    lines += describe(sorted(result.same_pairs, key=lambda p: p.score), "Lowest-scoring SAME-person pairs:")
    lines += ["", "Scores are descriptive only; no identity threshold is implied."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="folder with one subfolder of images per identity")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--diff-ratio", type=float, default=1.0,
                       help="different-person pairs sampled per same-person pair (default 1.0)")
    group.add_argument("--all-diff", action="store_true", help="use every different-person pair")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    parser.add_argument("--top-k", type=int, default=10, help="hard examples to show (default 10)")
    parser.add_argument("--csv", help="write pair-level results to this CSV file")
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda (default: auto)")
    args = parser.parse_args(argv)

    try:
        embedder = Embedder(device=args.device)
        result = evaluate(args.root, embedder, diff_ratio=None if args.all_diff else args.diff_ratio, seed=args.seed)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    print(format_report(result, top_k=args.top_k, model_name=embedder.model_name))
    if args.csv:
        write_pairs_csv(result, args.csv)
        print(f"\npair results written to {args.csv}")


if __name__ == "__main__":
    main()
