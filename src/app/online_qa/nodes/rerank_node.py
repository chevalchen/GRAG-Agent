from __future__ import annotations

from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState


def make_rerank_node(reranker_model: str, *, top_k: int) -> Callable[[OnlineQAState], dict]:
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(reranker_model)
    except Exception:
        model = None

    def rerank_node(state: OnlineQAState) -> dict:
        docs = list(state.get("docs_rrf") or [])
        if not docs:
            return {"docs_final": []}
        query = (state.get("query") or "").strip()
        if not query or model is None:
            return {"docs_final": docs[: int(top_k)]}
        pairs = [[query, d.page_content] for d in docs]
        try:
            scores = model.predict(pairs)
        except Exception:
            return {"docs_final": docs[: int(top_k)]}
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        docs_final = []
        for doc, score in ranked[: int(top_k)]:
            doc.metadata = {**(doc.metadata or {}), "rerank_score": float(score)}
            docs_final.append(doc)
        return {"docs_final": docs_final}

    return rerank_node
