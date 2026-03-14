from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState


def _dedup_key(doc: Document) -> str:
    md = doc.metadata or {}
    node_id = str(md.get("node_id") or "")
    chunk_id = str(md.get("chunk_id") or "")
    return node_id + chunk_id


def _rrf(docs_lists: list[list[Document]], *, k: int, top_k: int) -> list[Document]:
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
    def fuse_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        analysis = state.get("analysis")
        strategy = getattr(analysis, "recommended_strategy", "hybrid") if analysis else "hybrid"
        hybrid_docs = state.get("hybrid_docs") or []
        graph_docs = state.get("graph_docs") or []

        ready = True
        if strategy == "combined" and (not hybrid_docs or not graph_docs):
            ready = False

        docs_final = _rrf([hybrid_docs, graph_docs], k=60, top_k=int(top_k)) if strategy == "combined" else (hybrid_docs or graph_docs)
        metrics = {**(state.get("metrics") or {}), "fuse_seconds": time.time() - t0, "fuse_ready": ready}
        return {"docs_final": docs_final, "metrics": metrics}

    return fuse_node
