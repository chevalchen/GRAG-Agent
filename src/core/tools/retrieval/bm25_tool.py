from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class _BM25ToolInput(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=100)


class BM25Tool(BaseTool):
    name: str = "bm25_search"
    description: str = "基于 BM25 的文本检索（初始化时构建索引），返回 LangChain Documents"
    args_schema: type[BaseModel] = _BM25ToolInput

    def __init__(self, documents: list[Document]):
        super().__init__()
        from langchain_community.retrievers import BM25Retriever

        self._retriever = BM25Retriever.from_documents(documents) if documents else None

    def _run(self, query: str, top_k: int = 10) -> list[Document]:
        q = (query or "").strip()
        if not q or self._retriever is None:
            return []
        try:
            self._retriever.k = int(top_k)
        except Exception:
            pass
        try:
            return list(self._retriever.get_relevant_documents(q))
        except Exception:
            return []
