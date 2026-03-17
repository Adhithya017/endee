"""Endee-style vector store wrapper with local persistence.

This module intentionally keeps the interface simple. In production you can
replace the JSON persistence layer with direct HTTP calls to a running Endee
server while keeping the same `store_text` and `search` methods.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import numpy as np

from src.config import settings


class EndeeVectorStore:
    """Small persistent vector index that mirrors how Endee would be used."""

    def __init__(self, store_path: Path | None = None, index_name: str | None = None) -> None:
        self.store_path = Path(store_path or settings.endee_store_path)
        self.index_name = index_name or settings.endee_index_name
        self._lock = threading.Lock()
        self._records: List[Dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            self._records = []
            return

        with self.store_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._records = payload.get("records", [])

    def _persist(self) -> None:
        data = {"index_name": self.index_name, "records": self._records}
        with self.store_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)

    def count(self) -> int:
        """Return the number of stored chunks."""

        return len(self._records)

    def store_text(self, text: str, embedding: List[float], metadata: Dict[str, object] | None = None) -> Dict[str, object]:
        """Store one chunk plus metadata and embedding."""

        if not text or not text.strip():
            raise ValueError("Chunk text cannot be empty.")
        if not embedding:
            raise ValueError("Embedding cannot be empty.")

        record = {
            "id": str(uuid4()),
            "text": text.strip(),
            "embedding": embedding,
            "metadata": metadata or {},
        }

        with self._lock:
            self._records.append(record)
            self._persist()

        return record

    def store_many(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, object]] | None = None,
    ) -> List[Dict[str, object]]:
        """Store multiple chunks in one operation."""

        if len(texts) != len(embeddings):
            raise ValueError("Texts and embeddings must have the same length.")

        metadatas = metadatas or [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("Metadatas and texts must have the same length.")

        stored_records: List[Dict[str, object]] = []
        with self._lock:
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                record = {
                    "id": str(uuid4()),
                    "text": text.strip(),
                    "embedding": embedding,
                    "metadata": metadata,
                }
                self._records.append(record)
                stored_records.append(record)
            self._persist()

        return stored_records

    def search(self, query_embedding: List[float], top_k: int | None = None) -> List[Dict[str, object]]:
        """Return the top-k most similar chunks using cosine similarity."""

        if not self._records:
            return []
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        limit = top_k or settings.top_k
        query_vector = np.array(query_embedding, dtype=np.float32)

        scored: List[Dict[str, object]] = []
        for record in self._records:
            record_vector = np.array(record["embedding"], dtype=np.float32)
            score = float(np.dot(query_vector, record_vector))
            scored.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record.get("metadata", {}),
                    "score": round(score, 4),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


vector_store = EndeeVectorStore()
