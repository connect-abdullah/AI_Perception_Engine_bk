"""Manual check: print the cosine similarity between two person crops.

    python -m reid.compare crop_a.jpg crop_b.jpg

The output is a raw score only; it does not decide whether the crops show the same person.
"""

from __future__ import annotations

import argparse

import numpy as np

from reid.embedder import Embedder
from reid.similarity import cosine_similarity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda (default: auto)")
    args = parser.parse_args(argv)

    embedder = Embedder(device=args.device)
    emb_a = embedder.embed(args.image_a)
    emb_b = embedder.embed(args.image_b)

    print(f"model:             {embedder.model_name}")
    print(f"weights:           {embedder.weights_path}")
    print(f"device:            {embedder.device}")
    print(f"image_a:           {args.image_a}")
    print(f"image_b:           {args.image_b}")
    print(f"embedding shape:   {emb_a.shape}")
    print(f"norms (a, b):      {np.linalg.norm(emb_a):.4f}, {np.linalg.norm(emb_b):.4f}")
    print(f"cosine similarity: {cosine_similarity(emb_a, emb_b):.4f}")


if __name__ == "__main__":
    main()
