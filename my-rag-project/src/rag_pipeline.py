"""End-to-end ingestion, retrieval, and generation pipeline."""

from __future__ import annotations

from typing import Dict, List

from src.config import settings
from src.document_processor import split_text
from src.embeddings import embedding_service
from src.groq_client import groq_client
from src.vector_store import vector_store


class RAGPipeline:
    """Coordinates chunking, embeddings, retrieval, and LLM generation."""

    def ingest(self, text: str, source_name: str = "manual-input") -> Dict[str, object]:
        """Split and index a document into the local Endee-style store."""

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        chunks = split_text(text)
        if not chunks:
            raise ValueError("No chunks were generated from the supplied text.")

        embeddings = embedding_service.encode_batch(chunks)
        metadatas = [
            {"source": source_name, "chunk_index": index + 1, "total_chunks": len(chunks)}
            for index, _ in enumerate(chunks)
        ]
        stored_records = vector_store.store_many(chunks, embeddings, metadatas)

        return {
            "message": "Document ingested successfully.",
            "chunks_created": len(stored_records),
            "index_name": settings.endee_index_name,
            "source": source_name,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, object]]:
        """Retrieve the most relevant chunks for a question."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = embedding_service.encode_text(query)
        return vector_store.search(query_embedding, top_k=top_k or settings.top_k)

    def generate(self, query: str, top_k: int | None = None) -> Dict[str, object]:
        """Retrieve supporting chunks and ask Groq for a grounded answer."""

        matches = self.retrieve(query, top_k=top_k)
        if not matches:
            return {
                "answer": "The knowledge base is empty. Upload a document before asking questions.",
                "sources": [],
            }

        context_chunks = matches[: settings.max_context_chunks]
        context = "\n\n".join(
            f"Source {index + 1}:\n{item['text']}" for index, item in enumerate(context_chunks)
        )
        answer = groq_client.generate_answer(context=context, question=query)

        return {"answer": answer, "sources": matches}


rag_pipeline = RAGPipeline()
