# AI Research Assistant using Endee Vector Database

## Overview

This project is a complete Retrieval Augmented Generation (RAG) application built inside the Endee repository without modifying Endee's core folders. It accepts documents, splits them into chunks, generates embeddings with `sentence-transformers`, stores them in an Endee-style vector store, retrieves the most relevant chunks for a question, and sends the grounded context to Groq for final answer generation.

The vector layer is implemented as a clean wrapper in `src/vector_store.py`. It uses local JSON persistence plus NumPy similarity search so the app works immediately, while keeping the interface structured around Endee usage. If you later connect to a live Endee server, you can swap the persistence logic for Endee API calls without changing the rest of the application.

## Features

- Semantic search over uploaded text
- Endee-style vector storage wrapper with persistent local index
- Groq-powered grounded responses using `llama-3.1-8b-instant`
- Flask API for ingestion and querying
- Lightweight Streamlit UI for uploads and questions
- Error handling for missing input, empty knowledge base, and missing API key

## Architecture

`User -> Embedding -> Endee -> Retrieval -> Groq -> Response`

1. A user uploads text or a TXT file.
2. The document processor splits the content into overlapping chunks.
3. The embedding service converts those chunks into dense vectors.
4. The Endee wrapper stores the chunks and vectors in a persistent local index.
5. A question is embedded and matched against the stored vectors.
6. Top-k relevant chunks are sent to Groq as retrieval context.
7. Groq returns a grounded answer plus the retrieved sources.

## Project Structure

```text
my-rag-project/
├── src/
│   ├── config.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── groq_client.py
│   ├── rag_pipeline.py
│   └── document_processor.py
├── api/
│   └── app.py
├── frontend/
│   └── app.py
├── data/
│   └── sample.txt
├── requirements.txt
├── .env.example
└── README.md
```

## How Endee Is Used

This repository contains the real Endee vector database, but the app intentionally avoids changing Endee's internal source tree. Instead, `src/vector_store.py` provides a small Endee integration wrapper that models the same role Endee would play in production:

- `store_text(...)` stores chunk text, embedding vectors, and metadata into the vector index.
- `search(...)` performs top-k similarity retrieval against stored vectors.
- The wrapper persists indexed data in `data/endee_store.json`, which keeps the demo self-contained and easy to run.

To connect a real Endee deployment later, replace the local persistence/search internals with HTTP client calls to the Endee server, while preserving the public methods used by `rag_pipeline.py`.

## Setup

### 1. Move into the project

```bash
cd my-rag-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and set your Groq API key:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Optional settings:

- `MODEL_NAME` defaults to `llama-3.1-8b-instant`
- `TOP_K` controls retrieval depth
- `CHUNK_SIZE` and `CHUNK_OVERLAP` control chunking

### 5. Start the Flask API

```bash
python api/app.py
```

The API runs by default at `http://127.0.0.1:5000`.

### 6. Start the Streamlit UI

In a second terminal:

```bash
streamlit run frontend/app.py
```

## API Endpoints

### Health check

```http
GET /
```

Example response:

```json
{
  "index_name": "research-assistant",
  "service": "endee-rag-api",
  "status": "ok",
  "stored_chunks": 0
}
```

### Upload document

```http
POST /upload
Content-Type: application/json
```

Request body:

```json
{
  "text": "Endee is a vector database for RAG systems.",
  "source_name": "sample-note"
}
```

Response:

```json
{
  "chunks_created": 1,
  "index_name": "research-assistant",
  "message": "Document ingested successfully.",
  "source": "sample-note"
}
```

### Query the knowledge base

```http
POST /query
Content-Type: application/json
```

Request body:

```json
{
  "question": "What is Endee used for?"
}
```

Response:

```json
{
  "answer": "Endee is used for retrieval-heavy AI workloads such as RAG and semantic search.",
  "sources": [
    {
      "id": "chunk-id",
      "metadata": {
        "chunk_index": 1,
        "source": "sample-note",
        "total_chunks": 1
      },
      "score": 0.8231,
      "text": "Endee is a vector database for RAG systems."
    }
  ]
}
```

## Running the Demo

1. Start the Flask API with `python api/app.py`.
2. Open the Streamlit UI with `streamlit run frontend/app.py`.
3. Paste text or upload `data/sample.txt`.
4. Click `Ingest Document`.
5. Ask a question and review the answer plus retrieved context.

## Notes

- The first embedding request may take a bit longer because the sentence-transformers model loads on demand.
- If no data has been ingested, `/query` returns a helpful empty-knowledge-base message.
- If `GROQ_API_KEY` is missing, query generation will fail with a clear error until the key is configured.
