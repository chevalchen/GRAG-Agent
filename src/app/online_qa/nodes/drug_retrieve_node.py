from __future__ import annotations

import concurrent.futures
import time
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
    expand_factor: float = 2.0,
    max_retrieval_top_k: int = 12,
    merge_strategy: str = "balanced",
    timeout_seconds: float = 1.2,
) -> Callable[[OnlineQAState], dict]:
    def _merge_docs(bm25_docs: list[Document], milvus_docs: list[Document]) -> list[Document]:
        if merge_strategy != "balanced":
            return [*bm25_docs, *milvus_docs]
        merged: list[Document] = []
        i = 0
        max_len = max(len(bm25_docs), len(milvus_docs))
        while i < max_len:
            if i < len(bm25_docs):
                merged.append(bm25_docs[i])
            if i < len(milvus_docs):
                merged.append(milvus_docs[i])
            i += 1
        return merged

    def drug_retrieve_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        routing = state.get("routing") or {}
        if not routing.get("use_drug_vec"):
            return {"drug_docs": [], "metrics": {**(state.get("metrics") or {}), "drug_retrieve_seconds": 0.0}}
        query = (state.get("query") or "").strip()
        if not query:
            return {"drug_docs": [], "metrics": {**(state.get("metrics") or {}), "drug_retrieve_seconds": 0.0}}

        complexity = str(routing.get("complexity_level") or "complex")
        expanded_k = int(top_k * max(expand_factor, 1.0))
        retrieval_top_k = int(top_k) if complexity == "simple" else max(int(top_k), expanded_k)
        retrieval_top_k = min(retrieval_top_k, int(max_retrieval_top_k))
        resolved_drug = state.get("resolved_drug") or {}
        canonical_name = str(resolved_drug.get("canonical_name") or "").strip()
        node_id = str(resolved_drug.get("node_id") or "").strip()
        retrieval_query = f"{canonical_name} {query}".strip() if canonical_name else query
        expr = f'doc_type=="drug" AND node_id=="{node_id}"' if node_id else 'doc_type=="drug"'
        skip_bm25 = bool(node_id and complexity == "simple")

        bm25_docs: list[Document] = []
        milvus_docs: list[Document] = []
        timed_out = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures = {}
            if not skip_bm25:
                futures[ex.submit(bm25.invoke, {"query": retrieval_query, "top_k": retrieval_top_k})] = "bm25"
            futures[ex.submit(milvus.invoke, {"query": retrieval_query, "top_k": retrieval_top_k, "expr": expr})] = "milvus"
            done, pending = concurrent.futures.wait(
                futures.keys(),
                timeout=max(float(timeout_seconds), 0.01),
            )
            timed_out = bool(pending)
            for fut in done:
                name = futures[fut]
                try:
                    docs = fut.result() or []
                except Exception:
                    docs = []
                if name == "bm25":
                    bm25_docs = docs
                else:
                    milvus_docs = docs
            for fut in pending:
                fut.cancel()

        merged: list[Document] = []
        seen: set[str] = set()
        for doc in _merge_docs(bm25_docs, milvus_docs):
            md = doc.metadata or {}
            key = str(md.get("chunk_id") or md.get("node_id") or doc.page_content[:60])
            if key in seen:
                continue
            seen.add(key)
            doc.metadata = {
                **md,
                "search_source": md.get("search_source") or "drug_retrieve",
                "canonical_name": md.get("canonical_name") or canonical_name,
                "resolved_drug": resolved_drug or None,
            }
            merged.append(doc)
        metrics = {
            **(state.get("metrics") or {}),
            "drug_retrieve_seconds": time.time() - t0,
            "drug_bm25_docs": len(bm25_docs),
            "drug_milvus_docs": len(milvus_docs),
            "drug_merge_strategy": merge_strategy,
            "drug_retrieve_timeout": timed_out,
            "drug_retrieve_top_k": retrieval_top_k,
            "drug_bm25_skipped": skip_bm25,
        }
        return {"drug_docs": merged, "metrics": metrics}

    return drug_retrieve_node
