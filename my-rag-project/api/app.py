from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.rag_pipeline import rag_pipeline  # noqa: E402
from src.vector_store import vector_store  # noqa: E402


app = Flask(__name__)


@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "my-rag-project",
            "stored_chunks": vector_store.count(),
            "index_name": settings.endee_index_name,
        }
    )


@app.route("/upload", methods=["POST"])
def upload():
    payload = request.get_json(silent=True) or {}
    text_val = payload.get("text")
    if text_val is None:
        text = ""
    elif isinstance(text_val, str):
        text = text_val.strip()
    else:
        return jsonify({"error": "Field 'text' must be a string."}), 400
    if not text:
        return jsonify({"error": "Missing 'text' in request body."}), 400

    try:
        source_name = (payload.get("source_name") or "api-upload").strip()
        result = rag_pipeline.ingest(text=text, source_name=source_name)
        return jsonify(result), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    q_val = payload.get("question")
    if q_val is None:
        question = ""
    elif isinstance(q_val, str):
        question = q_val.strip()
    else:
        return jsonify({"error": "Field 'question' must be a string."}), 400
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    try:
        result = rag_pipeline.generate(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use a stable server configuration to avoid connection resets from the auto-reloader,
    # which can restart the process while requests are in-flight.
    app.run(host=settings.api_host, port=settings.api_port, debug=False, use_reloader=False)
