from __future__ import annotations

import time
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_lit_retrieve_node(
    milvus: MilvusVectorTool,
    *,
    top_k: int,
    expand_factor: float = 1.5,
    max_retrieval_top_k: int = 10,
) -> Callable[[OnlineQAState], dict]:
    def lit_retrieve_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        routing = state.get("routing") or {}
        if not routing.get("use_lit_vec"):
            return {"lit_docs": [], "metrics": {**(state.get("metrics") or {}), "lit_retrieve_seconds": 0.0}}
        query = (state.get("query") or "").strip()
        if not query:
            return {"lit_docs": [], "metrics": {**(state.get("metrics") or {}), "lit_retrieve_seconds": 0.0}}
        complexity = str(routing.get("complexity_level") or "complex")
        retrieval_top_k = int(top_k) if complexity == "simple" else max(int(top_k), int(top_k * max(expand_factor, 1.0)))
        retrieval_top_k = min(retrieval_top_k, int(max_retrieval_top_k))
        source_hint = list(routing.get("source_hint") or [])
        docs = milvus.invoke(
            {
                "query": query,
                "top_k": retrieval_top_k,
                "expr": 'doc_type=="tcm_literature"',
                "source_hint": source_hint,
            }
        ) or []
        metrics = {
            **(state.get("metrics") or {}),
            "lit_retrieve_seconds": time.time() - t0,
            "lit_retrieve_top_k": retrieval_top_k,
        }
        return {"lit_docs": docs, "metrics": metrics}

    return lit_retrieve_node
