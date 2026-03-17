"""Application configuration and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ENV_PATH = BASE_DIR / ".env"

# Override any existing environment variables so edits to `my-rag-project/.env`
# take effect immediately on process restart.
load_dotenv(ENV_PATH, override=True)


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings for the RAG application."""

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "llama3-8b-8192")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "3"))
    endee_index_name: str = os.getenv("ENDEE_INDEX_NAME", "research-assistant")
    endee_store_path: Path = Path(
        os.getenv("ENDEE_STORE_PATH", str(DATA_DIR / "endee_store.json"))
    )
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    # Keep backend on 8000 by default (matches frontend expectation).
    api_port: int = int(os.getenv("API_PORT", "8000"))
    streamlit_api_url: str = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

    def ensure_directories(self) -> None:
        """Create local folders needed for persistence."""

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.endee_store_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
