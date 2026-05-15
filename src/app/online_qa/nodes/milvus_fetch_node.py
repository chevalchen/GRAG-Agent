from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_milvus_fetch_node(
    milvus: MilvusVectorTool,
    *,
    fallback_top_k: int = 6,
    point_get_limit: int = 20,
) -> Callable[[OnlineQAState], dict]:
    """
    Reverse-path Milvus point-get node.

    Reads milvus_doc_ids (node_ids set by neo4j_search_node) and fetches
    the corresponding text slices from Milvus via scalar filter.

    Fallback: if milvus_doc_ids is empty, falls back to dense vector search
    using the rewritten query (one-level fallback per design contract).
    """

    def milvus_fetch_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        doc_ids = list(state.get("milvus_doc_ids") or [])
        query = (state.get("rewritten_query") or state.get("query") or "").strip()

        docs: list[Document] = []
        strategy = "point_get"

        if doc_ids:
            # Build scalar filter: node_id in [...]
            ids_csv = ", ".join(f'"{nid}"' for nid in doc_ids[:point_get_limit])
            expr = f"node_id in [{ids_csv}]"
            try:
                client = getattr(milvus, "client", None)
                collection = getattr(milvus, "_collection_name", None) or getattr(milvus, "collection_name", None)
                if client and collection:
                    rows = client.query(
                        collection_name=collection,
                        filter=expr,
                        output_fields=["chunk_id", "node_id", "doc_type", "text", "content"],
                        limit=point_get_limit,
                    ) or []
                    for row in rows:
                        text = str(row.get("text") or row.get("content") or "").strip()
                        if not text:
                            continue
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "chunk_id": row.get("chunk_id") or "",
                                "node_id": row.get("node_id") or "",
                                "doc_type": row.get("doc_type") or "",
                                "search_source": "milvus_fetch",
                            },
                        ))
                else:
                    # client not available: fall back to dense search
                    strategy = "fallback_dense"
            except Exception:
                strategy = "fallback_dense"
        else:
            strategy = "fallback_dense"

        if strategy == "fallback_dense" and query:
            # One-level fallback: dense top-k search with no filter
            try:
                raw = milvus.invoke({"query": query, "top_k": fallback_top_k}) or []
                for doc in raw:
                    md = doc.metadata or {}
                    doc.metadata = {**md, "search_source": "milvus_fetch_fallback"}
                docs = raw
            except Exception:
                docs = []

        return {
            "milvus_docs": docs,
            "metrics": {
                **(state.get("metrics") or {}),
                "milvus_fetch_seconds": time.time() - t0,
                "milvus_fetch_count": len(docs),
                "milvus_fetch_strategy": strategy,
            },
        }

    return milvus_fetch_node
