from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.utils.drug_name_normalizer import normalize_drug_name

# Reverse path: find drugs related to a given population / symptom / adverse reaction
_CYPHER_CAUTION_FOR = """
MATCH (p:Concept {conceptType:'Population'})
WHERE p.name CONTAINS $name OR $name CONTAINS p.name
MATCH (d:Concept {conceptType:'Drug'})-[:CAUTION_FOR]->(p)
RETURN d.nodeId AS node_id,
       d.name AS drug_name,
       'CAUTION_FOR' AS reason_type,
       p.name AS reason_label
LIMIT $limit
"""

_CYPHER_ADVERSE = """
MATCH (a:Concept {conceptType:'AdverseReaction'})
WHERE a.name CONTAINS $name OR $name CONTAINS a.name
MATCH (d:Concept {conceptType:'Drug'})-[:HAS_ADVERSE_REACTION]->(a)
RETURN d.nodeId AS node_id,
       d.name AS drug_name,
       'HAS_ADVERSE_REACTION' AS reason_type,
       a.name AS reason_label
LIMIT $limit
"""

_CYPHER_INDICATES = """
MATCH (x:Concept)
WHERE (x.conceptType IN ['Disease','Syndrome','Symptom'])
  AND (x.name CONTAINS $name OR $name CONTAINS x.name)
MATCH (d:Concept {conceptType:'Drug'})-[:INDICATES]->(x)
RETURN d.nodeId AS node_id,
       d.name AS drug_name,
       'INDICATES' AS reason_type,
       x.name AS reason_label
LIMIT $limit
"""

# Resolve a drug name to node_id for building milvus_doc_ids
_CYPHER_DRUG_RESOLVE = """
MATCH (d:Concept {conceptType:'Drug'})
WHERE d.normalizedName = $normalized OR $normalized IN coalesce(d.aliasNorms, [])
RETURN d.nodeId AS node_id LIMIT 1
"""


def _query_rows(neo4j: Neo4jGraphTool, cypher: str, params: dict) -> list[dict]:
    cfg = getattr(neo4j, "_config", None)
    if cfg is not None:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
            with driver.session(database=cfg.neo4j_database) as session:
                rows = session.run(cypher, **params).data() or []
            driver.close()
            return rows
        except Exception:
            pass
    graph = getattr(neo4j, "_graph", None)
    if graph is None:
        return []
    try:
        return graph.query(cypher, params) or []
    except Exception:
        return []


def _resolve_drug_node_id(neo4j: Neo4jGraphTool, name: str) -> str | None:
    normalized = normalize_drug_name(name)
    if not normalized:
        return None
    rows = _query_rows(neo4j, _CYPHER_DRUG_RESOLVE, {"normalized": normalized})
    if rows and rows[0].get("node_id"):
        return str(rows[0]["node_id"])
    return None


def make_neo4j_search_node(
    neo4j: Neo4jGraphTool,
    *,
    limit: int = 10,
) -> Callable[[OnlineQAState], dict]:
    """
    Reverse-path Neo4j search node for conditions_exact intent.

    Uses extracted_entities (diseases, symptoms, drugs) to find matching
    Drug nodes via CAUTION_FOR, HAS_ADVERSE_REACTION, and INDICATES relations.

    Outputs:
    - graph_docs: structured contraindication/indication documents
    - milvus_doc_ids: node_ids of matched drugs for MilvusFetch to point-get
    """

    def neo4j_search_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        query = (state.get("rewritten_query") or state.get("query") or "").strip()
        entities = state.get("extracted_entities") or {}

        # Collect candidate names: diseases + symptoms + drugs + keywords
        routing = state.get("routing") or {}
        candidates: list[str] = []
        for key in ("diseases", "symptoms", "drugs"):
            candidates.extend([str(x).strip() for x in (entities.get(key) or []) if str(x).strip()])
        for kw in (routing.get("keywords") or []):
            s = str(kw).strip()
            if s and s not in candidates:
                candidates.append(s)

        all_rows: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()

        for name in candidates:
            if not name:
                continue
            for cypher in [_CYPHER_CAUTION_FOR, _CYPHER_ADVERSE, _CYPHER_INDICATES]:
                rows = _query_rows(neo4j, cypher, {"name": name, "limit": limit})
                for row in rows:
                    key = (str(row.get("node_id") or ""), str(row.get("reason_type") or ""))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    all_rows.append(row)

        # If no Neo4j hits — milvus_fetch_node will fall back to dense top-k
        docs: list[Document] = []
        milvus_doc_ids: list[str] = []

        for row in all_rows:
            node_id = str(row.get("node_id") or "").strip()
            drug_name = str(row.get("drug_name") or "").strip()
            reason_type = str(row.get("reason_type") or "").strip()
            reason_label = str(row.get("reason_label") or "").strip()
            page = f"药品：{drug_name}\n关系类型：{reason_type}\n相关节点：{reason_label}"
            docs.append(Document(
                page_content=page,
                metadata={
                    "node_id": node_id,
                    "doc_type": "graph_reverse",
                    "search_source": "neo4j_search",
                    "drug_name": drug_name,
                    "reason_type": reason_type,
                    "reason_label": reason_label,
                },
            ))
            if node_id and node_id not in milvus_doc_ids:
                milvus_doc_ids.append(node_id)

        # Also resolve drug names in extracted_entities directly
        for drug_name in (entities.get("drugs") or []):
            nid = _resolve_drug_node_id(neo4j, drug_name)
            if nid and nid not in milvus_doc_ids:
                milvus_doc_ids.append(nid)

        return {
            "graph_docs": docs,
            "milvus_doc_ids": milvus_doc_ids,
            "metrics": {
                **(state.get("metrics") or {}),
                "neo4j_search_seconds": time.time() - t0,
                "neo4j_search_rows": len(all_rows),
                "neo4j_search_ids": len(milvus_doc_ids),
            },
        }

    return neo4j_search_node
