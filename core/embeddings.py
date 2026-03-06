"""Embedding generation using sentence-transformers."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """Generates text embeddings using a local sentence-transformer model.

    Uses all-MiniLM-L6-v2 by default: fast, free, 384-dimensional embeddings
    with strong semantic quality for RAG.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts into vectors."""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query])[0]
