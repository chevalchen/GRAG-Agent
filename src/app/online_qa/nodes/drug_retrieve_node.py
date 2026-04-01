from __future__ import annotations

import concurrent.futures
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_drug_retrieve_node(
    bm25: BM25Tool,
    milvus: MilvusVectorTool,
    *,
    top_k: int,
) -> Callable[[OnlineQAState], dict]:
    def drug_retrieve_node(state: OnlineQAState) -> dict:
        routing = state.get("routing") or {}
        if not routing.get("use_drug_vec"):
            return {"drug_docs": []}
        query = (state.get("query") or "").strip()
        if not query:
            return {"drug_docs": []}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_bm25 = ex.submit(bm25.invoke, {"query": query, "top_k": int(top_k)})
            f_milvus = ex.submit(milvus.invoke, {"query": query, "top_k": int(top_k), "expr": 'doc_type=="drug"'})
            bm25_docs = f_bm25.result() or []
            milvus_docs = f_milvus.result() or []

        merged: list[Document] = []
        seen: set[str] = set()
        for doc in [*bm25_docs, *milvus_docs]:
            md = doc.metadata or {}
            key = str(md.get("chunk_id") or md.get("node_id") or doc.page_content[:60])
            if key in seen:
                continue
            seen.add(key)
            doc.metadata = {**md, "search_source": md.get("search_source") or "drug_retrieve"}
            merged.append(doc)
        return {"drug_docs": merged}

    return drug_retrieve_node
