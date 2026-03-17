"""Sentence-transformers embedding helpers."""

from __future__ import annotations

from typing import Iterable, List

from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingService:
    """Lazy wrapper around the sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_text(self, text: str) -> List[float]:
        """Encode a single text string into a dense embedding."""

        if not text or not text.strip():
            raise ValueError("Text for embedding cannot be empty.")
        vector = self._get_model().encode(text.strip(), normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: Iterable[str]) -> List[List[float]]:
        """Encode a batch of text chunks into dense embeddings."""

        payload = [text.strip() for text in texts if text and text.strip()]
        if not payload:
            return []
        vectors = self._get_model().encode(payload, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


embedding_service = EmbeddingService()
