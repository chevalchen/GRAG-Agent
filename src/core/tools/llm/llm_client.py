from __future__ import annotations

from typing import Iterator

from openai import OpenAI


class LLMClient:
    """OpenAI LLM 客户端"""
    def __init__(self, api_key: str, base_url: str, model: str):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: list[dict], stream: bool = False) -> str | Iterator[str]:
        if not stream:
            return self._chat_sync(messages)
        return self._stream_sync(messages)

    def _chat_sync(self, messages: list[dict]) -> str:
        rsp = self._client.chat.completions.create(model=self._model, messages=messages)
        return rsp.choices[0].message.content or ""

    def _stream_sync(self, messages: list[dict]) -> Iterator[str]:
        rsp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
        for chunk in rsp:
            content = chunk.choices[0].delta.content
            if content:
                yield content
