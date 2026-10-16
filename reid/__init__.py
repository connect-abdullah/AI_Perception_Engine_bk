"""Person re-identification: crop embeddings and similarity. Independent of app/ and the database."""

from reid.embedder import Embedder
from reid.similarity import cosine_similarity

__all__ = ["Embedder", "cosine_similarity"]
