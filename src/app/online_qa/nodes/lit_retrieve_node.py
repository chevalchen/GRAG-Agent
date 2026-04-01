from __future__ import annotations

from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_lit_retrieve_node(milvus: MilvusVectorTool, *, top_k: int) -> Callable[[OnlineQAState], dict]:
    def lit_retrieve_node(state: OnlineQAState) -> dict:
        routing = state.get("routing") or {}
        if not routing.get("use_lit_vec"):
            return {"lit_docs": []}
        query = (state.get("query") or "").strip()
        if not query:
            return {"lit_docs": []}
        source_hint = list(routing.get("source_hint") or [])
        docs = milvus.invoke(
            {
                "query": query,
                "top_k": int(top_k),
                "expr": 'doc_type=="tcm_literature"',
                "source_hint": source_hint,
            }
        ) or []
        return {"lit_docs": docs}

    return lit_retrieve_node
