from __future__ import annotations

import time
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState


def _doc_rerank_text(doc) -> str:
    metadata = doc.metadata or {}
    question = str(doc.page_content or "").strip()
    answer = str(metadata.get("answer") or "").strip()
    chunk_type = str(metadata.get("chunk_type") or "").strip()
    doc_type = str(metadata.get("doc_type") or "").strip()
    if answer and (chunk_type == "qa" or doc_type == "health_popularization"):
        return f"问题：{question}\n回答：{answer}" if question else f"回答：{answer}"
    return question


def make_rerank_node(
    reranker_model: str,
    *,
    top_k: int,
    simple_skip_threshold: int = 6,
    simple_candidate_limit: int = 8,
    complex_candidate_limit: int = 14,
    cross_encoder_model=None,
) -> Callable[[OnlineQAState], dict]:
    if cross_encoder_model is not None:
        model = cross_encoder_model
    else:
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(reranker_model)
        except Exception:
            model = None

    def rerank_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        docs = list(state.get("docs_rrf") or [])
        if not docs:
            return {
                "docs_final": [],
                "metrics": {
                    **(state.get("metrics") or {}),
                    "rerank_seconds": 0.0,
                    "rerank_mode": "skipped",
                    "rerank_candidates": 0,
                },
            }
        query = (state.get("rewritten_query") or state.get("query") or "").strip()
        routing = state.get("routing") or {}
        complexity = str(routing.get("complexity_level") or "complex")
        if len(docs) <= int(top_k):
            metrics = {
                **(state.get("metrics") or {}),
                "rerank_seconds": time.time() - t0,
                "rerank_fallback": False,
                "rerank_mode": "skipped",
                "rerank_candidates": len(docs),
            }
            return {"docs_final": docs[: int(top_k)], "metrics": metrics}
        if complexity == "simple" and len(docs) <= int(simple_skip_threshold):
            metrics = {
                **(state.get("metrics") or {}),
                "rerank_seconds": time.time() - t0,
                "rerank_fallback": False,
                "rerank_mode": "skipped",
                "rerank_candidates": len(docs),
            }
            return {"docs_final": docs[: int(top_k)], "metrics": metrics}
        candidate_limit = int(complex_candidate_limit if complexity == "complex" else simple_candidate_limit)
        candidate_limit = max(candidate_limit, int(top_k))
        candidates = docs[:candidate_limit]
        rerank_mode = "full" if len(candidates) == len(docs) else "reduced"
        if not query or model is None:
            metrics = {
                **(state.get("metrics") or {}),
                "rerank_seconds": time.time() - t0,
                "rerank_fallback": True,
                "rerank_mode": "fallback",
                "rerank_candidates": len(candidates),
            }
            return {"docs_final": candidates[: int(top_k)], "metrics": metrics}
        pairs = [[query, _doc_rerank_text(d)] for d in candidates]
        try:
            scores = model.predict(pairs)
        except Exception:
            metrics = {
                **(state.get("metrics") or {}),
                "rerank_seconds": time.time() - t0,
                "rerank_fallback": True,
                "rerank_mode": "fallback",
                "rerank_candidates": len(candidates),
            }
            return {"docs_final": candidates[: int(top_k)], "metrics": metrics}
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        docs_final = []
        for doc, score in ranked[: int(top_k)]:
            doc.metadata = {**(doc.metadata or {}), "rerank_score": float(score)}
            docs_final.append(doc)
        metrics = {
            **(state.get("metrics") or {}),
            "rerank_seconds": time.time() - t0,
            "rerank_fallback": False,
            "rerank_mode": rerank_mode,
            "rerank_candidates": len(candidates),
        }
        return {"docs_final": docs_final, "metrics": metrics}

    return rerank_node
