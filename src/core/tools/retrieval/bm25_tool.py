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

        self._documents = list(documents or [])
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


def build_bm25_from_milvus(milvus_client, top_k: int = 10, collection_name: str = "tcm_knowledge") -> BM25Tool:
    if milvus_client is None:
        return BM25Tool(documents=[])
    try:
        rows = milvus_client.query(
            collection_name=collection_name,
            filter='doc_type == "drug"',
            output_fields=[
                "text",
                "doc_type",
                "chunk_type",
                "source",
                "drug_name",
                "canonical_name",
                "normalized_name",
                "aliases_text",
                "alias_norms_text",
                "answer",
                "node_id",
                "chunk_id",
            ],
            limit=16384,
        )
    except Exception:
        rows = []
    documents: list[Document] = []
    for row in rows or []:
        documents.append(
            Document(
                page_content=str(row.get("text") or ""),
                metadata={
                    "doc_type": row.get("doc_type") or "drug",
                    "chunk_type": row.get("chunk_type") or "",
                    "source": row.get("source") or "",
                    "drug_name": row.get("drug_name") or "",
                    "canonical_name": row.get("canonical_name") or "",
                    "normalized_name": row.get("normalized_name") or "",
                    "aliases_text": row.get("aliases_text") or "",
                    "alias_norms_text": row.get("alias_norms_text") or "",
                    "answer": row.get("answer") or "",
                    "node_id": row.get("node_id") or "",
                    "chunk_id": row.get("chunk_id") or "",
                },
            )
        )
    tool = BM25Tool(documents=documents)
    return tool
