import logging
from typing import List, Optional
import time

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class AnswerGenerationTool:
    def __init__(self, generation_module, max_retries: int = 3, retry_backoff_seconds: float = 1.5):
        self._generation = generation_module
        self._max_retries = max(1, max_retries)
        self._retry_backoff_seconds = max(0.1, retry_backoff_seconds)

    def generate(self, question: str, documents: List[Document], stream: bool = False, history: Optional[list[dict]] = None) -> str:
        question = self._inject_history(question, history)
        if not stream:
            return self._generate_with_retry(question, documents)
        return self._stream_to_string(question, documents)

    def _inject_history(self, question: str, history: Optional[list[dict]]) -> str:
        if not history:
            return question
        recent = history[-10:]
        lines: List[str] = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if not content:
                continue
            if role == "user":
                prefix = "用户"
            elif role == "assistant":
                prefix = "助手"
            else:
                prefix = str(role) if role else "未知"
            lines.append(f"{prefix}: {content}")
        if not lines:
            return question
        history_text = "\n".join(lines)
        return f"对话历史（最近{len(lines)}条）:\n{history_text}\n\n当前问题: {question}"

    def _generate_with_retry(self, question: str, documents: List[Document]) -> str:
        last_response = ""
        for attempt in range(self._max_retries):
            response = self._generation.generate_adaptive_answer(question, documents)
            last_response = response or ""
            if not self._is_overloaded_error(last_response):
                return last_response
            if attempt < self._max_retries - 1:
                wait_s = self._retry_backoff_seconds * (attempt + 1)
                logger.warning(f"LLM引擎过载，{wait_s:.1f}秒后进行第{attempt + 2}次重试")
                time.sleep(wait_s)
        return last_response

    def _is_overloaded_error(self, text: str) -> bool:
        lowered = (text or "").lower()
        return (
            "error code: 429" in lowered
            or "engine_overloaded_error" in lowered
            or "currently overloaded" in lowered
        )

    def _stream_to_string(self, question: str, documents: List[Document]) -> str:
        parts: List[str] = []
        for chunk in self._generation.generate_adaptive_answer_stream(question, documents):
            parts.append(chunk)
        return "".join(parts)
