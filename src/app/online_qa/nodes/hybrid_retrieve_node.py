from __future__ import annotations

import concurrent.futures
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_hybrid_retrieve_node(
    bm25: BM25Tool,
    milvus: MilvusVectorTool,
    neo4j: Neo4jGraphTool,
    *,
    top_k: int,
) -> Callable[[OnlineQAState], dict]:
    """
    构建混合检索节点
    
    Args:
        bm25: BM25 检索工具
        milvus: Milvus 向量工具
        neo4j: Neo4j 图工具
        top_k: 检索文档数量
    Returns:
        混合检索节点
    """
    def hybrid_retrieve_node(state: OnlineQAState) -> dict:
        """
        混合检索节点
        
        Args:
            state: 在线问答状态
            
        Returns:
            混合检索结果
        """
        query = (state.get("query") or "").strip()
        if not query:
            return {"hybrid_docs": []}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_bm25 = ex.submit(bm25.invoke, {"query": query, "top_k": int(top_k)})
            f_milvus = ex.submit(milvus.invoke, {"query": query, "top_k": int(top_k), "filter_expr": ""})
            bm25_docs = f_bm25.result() or []
            milvus_docs = f_milvus.result() or []

        combined: list[Document] = []
        for d in bm25_docs:
            d.metadata = {**(d.metadata or {}), "search_source": "bm25"}
            combined.append(d)
        combined.extend(milvus_docs)

        node_ids: list[str] = []
        for d in milvus_docs:
            nid = (d.metadata or {}).get("node_id")
            if nid:
                node_ids.append(str(nid))
        node_ids = list(dict.fromkeys(node_ids))[:10]
        expanded: list[Document] = []
        for nid in node_ids:
            expanded.extend(neo4j.invoke({"query": f"node_id:{nid}", "max_depth": 1, "max_nodes": 30}) or [])
        for d in expanded:
            d.metadata = {**(d.metadata or {}), "search_source": "neo4j_expand"}
        combined.extend(expanded)

        return {"hybrid_docs": combined}

    return hybrid_retrieve_node
