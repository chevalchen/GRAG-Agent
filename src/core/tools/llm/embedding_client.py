from __future__ import annotations

from openai import OpenAI


class EmbeddingClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_sync(texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        rsp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in rsp.data]
