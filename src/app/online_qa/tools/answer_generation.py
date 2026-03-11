import asyncio
from typing import List, Optional

from langchain_core.documents import Document


class AnswerGenerationTool:
    def __init__(self, generation_module):
        self._generation = generation_module

    async def generate(self, question: str, documents: List[Document], stream: bool = False) -> str:
        if not stream:
            return await asyncio.to_thread(self._generation.generate_adaptive_answer, question, documents)
        return await asyncio.to_thread(self._stream_to_string, question, documents)

    def _stream_to_string(self, question: str, documents: List[Document]) -> str:
        parts: List[str] = []
        for chunk in self._generation.generate_adaptive_answer_stream(question, documents):
            parts.append(chunk)
        return "".join(parts)

