from __future__ import annotations

import time
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_health_retrieve_node(
    health_milvus_tool: MilvusVectorTool,
    top_k: int = 3,
    expand_factor: float = 1.5,
    max_retrieval_top_k: int = 12,
) -> Callable[[OnlineQAState], dict]:
    def health_retrieve_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        routing = state.get("routing") or {}
        if not routing.get("use_health_vec"):
            return {"health_docs": [], "metrics": {**(state.get("metrics") or {}), "health_retrieve_seconds": 0.0}}

        query = state.get("query", "")
        if not query:
            return {"health_docs": [], "metrics": {**(state.get("metrics") or {}), "health_retrieve_seconds": time.time() - t0}}

        # Progressive degradation: skip retrieval when health collection is absent.
        if not health_milvus_tool.has_collection():
            metrics = {
                **(state.get("metrics") or {}),
                "health_retrieve_seconds": time.time() - t0,
                "health_retrieved": 0,
                "health_collection_ready": False,
            }
            return {"health_docs": [], "metrics": metrics}

        retrieval_top_k = min(max(int(top_k * expand_factor), int(top_k)), max_retrieval_top_k)
        try:
            docs = health_milvus_tool.invoke({"query": query, "top_k": retrieval_top_k}) or []
        except Exception:
            docs = []

        metrics = {
            **(state.get("metrics") or {}),
            "health_retrieve_seconds": time.time() - t0,
            "health_retrieved": len(docs),
            "health_retrieve_top_k": retrieval_top_k,
            "health_collection_ready": True,
        }
        return {"health_docs": docs, "metrics": metrics}

    return health_retrieve_node
