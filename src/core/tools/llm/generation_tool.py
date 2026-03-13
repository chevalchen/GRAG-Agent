from __future__ import annotations

from src.core.schemas.document import Document
from src.core.tools.llm.llm_client import LLMClient


class GenerationTool:
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    def generate(
        self,
        query: str,
        docs: list[Document],
        stream: bool = False,
        history: list[dict] | None = None,
    ) -> str | None:
        context = "\n\n".join(d.content for d in docs[:10])
        history_prefix = ""
        if history:
            recent = history[-10:]
            lines: list[str] = []
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
            if lines:
                history_prefix = "对话历史（最近几轮）：\n" + "\n".join(lines) + "\n\n"
        prompt = f"{history_prefix}请基于下列资料回答问题。\n\n资料:\n{context}\n\n问题:{query}\n\n回答:"
        messages = [{"role": "user", "content": prompt}]
        if not stream:
            return self._llm_client.chat(messages, stream=False)
        parts = []
        for chunk in self._llm_client.chat(messages, stream=True):
            parts.append(chunk)
        return "".join(parts)
