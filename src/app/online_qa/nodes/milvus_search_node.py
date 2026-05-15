from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool
from src.core.utils.drug_name_normalizer import normalize_drug_name


def _resolve_kg_anchor(neo4j: Neo4jGraphTool, candidates: list[str]) -> str | None:
    """
    Given a list of candidate drug/entity names extracted from the query,
    return the first node_id that resolves in Neo4j.
    """
    cfg = getattr(neo4j, "_config", None)
    if cfg is None:
        return None

    for candidate in candidates:
        normalized = normalize_drug_name(candidate)
        if not normalized:
            continue
        cypher = """
        MATCH (d:Concept {conceptType:'Drug'})
        WHERE d.normalizedName = $normalized
           OR $normalized IN coalesce(d.aliasNorms, [])
        RETURN d.nodeId AS node_id LIMIT 1
        """
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
            with driver.session(database=cfg.neo4j_database) as session:
                rows = session.run(cypher, normalized=normalized).data() or []
            driver.close()
        except Exception:
            continue
        if rows and rows[0].get("node_id"):
            return str(rows[0]["node_id"])
    return None


def _merge_balanced(bm25_docs: list[Document], milvus_docs: list[Document]) -> list[Document]:
    merged: list[Document] = []
    for i in range(max(len(bm25_docs), len(milvus_docs))):
        if i < len(bm25_docs):
            merged.append(bm25_docs[i])
        if i < len(milvus_docs):
            merged.append(milvus_docs[i])
    return merged


def make_milvus_search_node(
    bm25: BM25Tool,
    milvus: MilvusVectorTool,
    neo4j: Neo4jGraphTool,
    *,
    top_k: int = 6,
    expand_factor: float = 2.0,
    max_top_k: int = 12,
    timeout_seconds: float = 1.2,
) -> Callable[[OnlineQAState], dict]:
    """
    Generalized hybrid retrieval node for the forward (symptom_to_drug / theory) path.

    Uses BM25 + dense Milvus without doc_type constraints.
    Sets kg_anchor_id by resolving extracted drug entities against Neo4j.
    """

    def milvus_search_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        routing = state.get("routing") or {}
        query = (state.get("rewritten_query") or state.get("query") or "").strip()
        if not query:
            return {"milvus_docs": [], "kg_anchor_id": None, "metrics": _m(state, t0)}

        complexity = str(routing.get("complexity_level") or "complex")
        expanded_k = int(top_k * max(expand_factor, 1.0))
        retrieval_top_k = top_k if complexity == "simple" else min(expanded_k, max_top_k)

        bm25_docs: list[Document] = []
        milvus_docs_raw: list[Document] = []
        timed_out = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            bm25_fut = ex.submit(bm25.invoke, {"query": query, "top_k": retrieval_top_k})
            milvus_fut = ex.submit(milvus.invoke, {"query": query, "top_k": retrieval_top_k})
            done, pending = concurrent.futures.wait(
                [bm25_fut, milvus_fut], timeout=max(float(timeout_seconds), 0.01)
            )
            timed_out = bool(pending)
            for fut in done:
                try:
                    docs = fut.result() or []
                except Exception:
                    docs = []
                if fut is bm25_fut:
                    bm25_docs = docs
                else:
                    milvus_docs_raw = docs
            for fut in pending:
                fut.cancel()

        merged: list[Document] = []
        seen: set[str] = set()
        for doc in _merge_balanced(bm25_docs, milvus_docs_raw):
            md = doc.metadata or {}
            key = str(md.get("chunk_id") or md.get("node_id") or doc.page_content[:60])
            if key in seen:
                continue
            seen.add(key)
            doc.metadata = {**md, "search_source": md.get("search_source") or "milvus_search"}
            merged.append(doc)

        # resolve kg_anchor_id from extracted entities → top-1 dense hit as fallback
        kg_anchor_id: str | None = None
        entities = state.get("extracted_entities") or {}
        drug_names = entities.get("drugs") or []

        if drug_names:
            kg_anchor_id = _resolve_kg_anchor(neo4j, drug_names)

        if not kg_anchor_id:
            # try top-1 milvus hit's node_id as anchor
            for doc in milvus_docs_raw[:1]:
                nid = str((doc.metadata or {}).get("node_id") or "").strip()
                if nid:
                    kg_anchor_id = nid
                    break

        return {
            "milvus_docs": merged,
            "kg_anchor_id": kg_anchor_id,
            "metrics": {
                **_m(state, t0),
                "milvus_search_bm25": len(bm25_docs),
                "milvus_search_dense": len(milvus_docs_raw),
                "milvus_search_merged": len(merged),
                "milvus_search_timeout": timed_out,
                "kg_anchor_id": kg_anchor_id or "",
            },
        }

    return milvus_search_node


def _m(state: OnlineQAState, t0: float) -> dict:
    base = state.get("metrics") or {}
    return {**base, "milvus_search_seconds": time.time() - t0}
