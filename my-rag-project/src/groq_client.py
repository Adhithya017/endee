"""Groq client wrapper used by the RAG pipeline."""

from __future__ import annotations

from groq import Groq

from src.config import settings


class GroqClient:
    """Thin wrapper that keeps prompt construction in one place."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.model_name
        self._client: Groq | None = None

    def _get_client(self) -> Groq:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your environment or .env file.")
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer grounded in the retrieved context."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        system_prompt = (
            "You are an AI research assistant. Answer using only the provided context. "
            "If the answer is not in the context, say that the available documents do not contain enough information."
        )
        user_prompt = f"Context:\n{context or 'No context available.'}\n\nQuestion:\n{question.strip()}"

        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()


groq_client = GroqClient()
