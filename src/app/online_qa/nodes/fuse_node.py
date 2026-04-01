from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState


def _dedup_key(doc: Document) -> str:
    """
    文档去重键
    """
    md = doc.metadata or {}
    node_id = str(md.get("node_id") or "")
    chunk_id = str(md.get("chunk_id") or "")
    return node_id + chunk_id


def _rrf(docs_lists: list[list[Document]], *, k: int, top_k: int) -> list[Document]:
    """
    递归倒数排名融合（RRF）
    
    Args:
        docs_lists: 文档列表
        k: 融合参数
        top_k: 融合文档数量

    Returns:
        融合后的文档列表
    """
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in docs_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _dedup_key(doc) or str(id(doc))
            by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[Document] = []
    for key, s in ranked[:top_k]:
        d = by_key[key]
        d.metadata = {**(d.metadata or {}), "final_score": float(s)}
        out.append(d)
    return out


def make_fuse_node(*, top_k: int) -> Callable[[OnlineQAState], dict]:
    """
    构建融合节点

    Args:
        top_k: 融合文档数量

    Returns:
        融合节点
    """
    def fuse_node(state: OnlineQAState) -> dict:
        """
        融合节点
        
        Args:
            state: 在线问答状态
            
        Returns:
            融合结果
        """
        t0 = time.time()
        drug_docs = state.get("drug_docs") or []
        graph_docs = state.get("graph_docs") or []
        lit_docs = state.get("lit_docs") or []
        health_docs = state.get("health_docs") or []
        all_lists = [graph_docs, drug_docs, lit_docs, health_docs]
        non_empty = [x for x in all_lists if x]
        if not non_empty:
            docs_rrf = []
        elif len(non_empty) == 1:
            docs_rrf = non_empty[0][: int(top_k * 3)]
        else:
            docs_rrf = _rrf(all_lists, k=60, top_k=int(top_k * 3))
        metrics = {**(state.get("metrics") or {}), "fuse_seconds": time.time() - t0, "fuse_ready": True}
        return {"docs_rrf": docs_rrf, "metrics": metrics}

    return fuse_node
