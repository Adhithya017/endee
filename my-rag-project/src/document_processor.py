"""Utilities for turning raw documents into manageable chunks."""

from __future__ import annotations

from typing import List

from src.config import settings


def normalize_text(text: str) -> str:
    """Trim noisy whitespace while preserving paragraph boundaries."""

    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


def split_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
    """Split text into overlapping character-based chunks."""

    cleaned_text = normalize_text(text)
    if not cleaned_text:
        return []

    size = chunk_size or settings.chunk_size
    chunk_overlap = overlap if overlap is not None else settings.chunk_overlap

    if size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    step = size - chunk_overlap
    chunks: List[str] = []

    for start in range(0, len(cleaned_text), step):
        chunk = cleaned_text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks
