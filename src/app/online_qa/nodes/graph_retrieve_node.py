from __future__ import annotations

from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool


def make_graph_retrieve_node(neo4j: Neo4jGraphTool, *, top_k: int) -> Callable[[OnlineQAState], dict]:
    def graph_retrieve_node(state: OnlineQAState) -> dict:
        analysis = state.get("analysis")
        keywords: list[str] = []
        relationship_intensity = 0.0
        if analysis:
            keywords = list(getattr(analysis, "keywords", []) or [])
            try:
                relationship_intensity = float(getattr(analysis, "relationship_intensity", 0.0) or 0.0)
            except Exception:
                relationship_intensity = 0.0
        query = (state.get("query") or "").strip()
        q = " ".join([*keywords]) if keywords else query
        max_depth = 1 if relationship_intensity < 0.3 else 2
        docs = neo4j.invoke({"query": q, "max_depth": int(max_depth), "max_nodes": int(top_k * 20)}) or []
        for d in docs:
            d.metadata = {**(d.metadata or {}), "search_source": "neo4j"}
        return {"graph_docs": docs}

    return graph_retrieve_node
