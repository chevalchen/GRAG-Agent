from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class _BM25ToolInput(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=100)


class BM25Tool(BaseTool):
    """BM25检索工具，支持懒加载和磁盘缓存"""

    name: str = "bm25_search"
    description: str = "基于BM25的稀疏检索，支持从Milvus懒加载构建索引"
    args_schema: type[BaseModel] = _BM25ToolInput

    def __init__(
        self,
        milvus_tool: Any | None = None,
        *,
        documents: list[Document] | None = None,
        cache_path: str = ".cache/bm25_index.pkl",
        filter_expr: str | None = None,
    ):
        """
        Initialize the BM25 tool.

        Args:
            milvus_tool:  MilvusVectorTool instance for lazy corpus loading.
            documents:    Pre-built document list (takes priority over milvus_tool).
            cache_path:   Disk cache path for the serialised BM25 index.
            filter_expr:  Optional Milvus filter expression for corpus selection.
                          ``None`` (default) fetches the full corpus.
                          Example: ``'doc_type in ["drug", "tcm_literature"]'``
        """
        super().__init__()
        self._milvus_tool = milvus_tool
        self._documents = documents
        self._cache_path = Path(cache_path)
        self._filter_expr = filter_expr  # None = full corpus
        self._retriever = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """确保索引已加载（懒加载）"""
        if self._loaded:
            return self._retriever is not None

        # 1. 尝试从缓存加载
        if self._load_from_cache():
            self._loaded = True
            return True

        # 2. 使用传入的documents
        if self._documents:
            self._build_from_documents(self._documents)
            self._loaded = True
            return self._retriever is not None

        # 3. Lazy-load from Milvus using the configured filter (or full corpus)
        if self._milvus_tool is not None:
            docs = self._fetch_from_milvus(self._filter_expr)
            if docs:
                self._build_from_documents(docs)
                self._save_to_cache()
                self._loaded = True
                return True

        self._loaded = True
        return False

    def _load_from_cache(self) -> bool:
        """从磁盘缓存加载索引"""
        if not self._cache_path.exists():
            return False
        try:
            with open(self._cache_path, "rb") as f:
                cache = pickle.load(f)
            self._retriever = cache.get("retriever")
            self._documents = cache.get("documents", [])
            return self._retriever is not None
        except Exception:
            return False

    def _save_to_cache(self) -> None:
        """保存索引到磁盘缓存"""
        if self._retriever is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache = {
                "retriever": self._retriever,
                "documents": self._documents,
            }
            with open(self._cache_path, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass

    def _fetch_from_milvus(self, filter_expr: str | None = None) -> list[Document]:
        """Fetch documents from Milvus to build the BM25 index corpus.

        Args:
            filter_expr: Milvus filter expression.  ``None`` (or empty string)
                         fetches the full corpus without a doc_type constraint.
                         Pass e.g. ``'doc_type == "drug"'`` to restrict the corpus.
                         When not supplied, falls back to the instance-level
                         ``filter_expr`` set at construction time (default: None).
        """
        if self._milvus_tool is None:
            return []
        try:
            client = getattr(self._milvus_tool, "client", None)
            collection_name = getattr(self._milvus_tool, "collection_name", "tcm_knowledge")
            if client is None:
                return []

            # Resolve effective filter: caller arg > instance default > full corpus ("")
            effective_filter: str = (
                filter_expr if filter_expr is not None
                else (self._filter_expr if self._filter_expr is not None else "")
            )

            rows = client.query(
                collection_name=collection_name,
                filter=effective_filter,
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
            ) or []

            documents: list[Document] = []
            for row in rows:
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
            return documents
        except Exception:
            return []

    def _build_from_documents(self, documents: list[Document]) -> None:
        """从文档构建BM25索引"""
        if not documents:
            self._retriever = None
            return
        try:
            from langchain_community.retrievers import BM25Retriever

            self._documents = list(documents)
            self._retriever = BM25Retriever.from_documents(documents)
        except Exception:
            self._retriever = None

    def _run(self, query: str, top_k: int = 10) -> list[Document]:
        """执行BM25检索"""
        q = (query or "").strip()
        if not q:
            return []

        if not self._ensure_loaded():
            return []

        try:
            self._retriever.k = int(top_k)
        except Exception:
            pass

        try:
            return list(self._retriever.get_relevant_documents(q))
        except Exception:
            return []

    def clear_cache(self) -> None:
        """清除磁盘缓存"""
        try:
            if self._cache_path.exists():
                self._cache_path.unlink()
        except Exception:
            pass

    def rebuild(self) -> bool:
        """强制重建索引"""
        self._loaded = False
        self._retriever = None
        self.clear_cache()
        return self._ensure_loaded()


# 兼容旧代码的工厂函数
def build_bm25_from_milvus(
    milvus_client,
    top_k: int = 10,
    collection_name: str = "tcm_knowledge",
) -> BM25Tool:
    """
    从Milvus构建BM25工具（兼容旧接口）

    注意：新版建议直接使用 BM25Tool(milvus_tool=...) 实现懒加载
    """
    from src.core.tools.vector.milvus_tool import MilvusVectorTool

    # 构造一个最小的milvus_tool对象用于懒加载
    class _MinimalMilvusTool:
        def __init__(self, client, collection_name):
            self.client = client
            self.collection_name = collection_name

    minimal_tool = _MinimalMilvusTool(milvus_client, collection_name)
    tool = BM25Tool(milvus_tool=minimal_tool)
    # 立即触发加载（保持旧行为兼容性）
    tool._ensure_loaded()
    return tool
