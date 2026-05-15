from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool

# Reused from graph_retrieve_node — fetch full drug subgraph by node_id
_TEMPLATE_BY_NODE_ID = """
MATCH (d:Concept {nodeId: $node_id, conceptType:'Drug'})
OPTIONAL MATCH (d)-[:CONTAINS]->(i:Concept {conceptType:'Ingredient'})
OPTIONAL MATCH (d)-[:HAS_EFFECT]->(e:Concept {conceptType:'Effect'})
OPTIONAL MATCH (d)-[:INDICATES]->(x:Concept)
OPTIONAL MATCH (d)-[:CAUTION_FOR]->(p:Concept {conceptType:'Population'})
OPTIONAL MATCH (d)-[ra]->(a:Concept {conceptType:'AdverseReaction'})
WHERE type(ra) = 'HAS_ADVERSE_REACTION'
OPTIONAL MATCH (i)-[:CAUTION_FOR]->(ip:Concept {conceptType:'Population'})
RETURN d.nodeId AS node_id,
       d.name AS drug_name,
       collect(DISTINCT i.name)[..50] AS ingredients,
       collect(DISTINCT e.name)[..50] AS effects,
       collect(DISTINCT x.name)[..50] AS indications,
       collect(DISTINCT p.name)[..50] AS cautions,
       collect(DISTINCT a.name)[..50] AS adverse_reactions,
       collect(DISTINCT {ingredient: i.name, caution: ip.name})[..80] AS ingredient_cautions
LIMIT $limit
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


def _clean(values: list) -> list[str]:
    return [str(x).strip() for x in (values or []) if str(x).strip()]


def _row_to_doc(row: dict) -> Document:
    ingredients = _clean(row.get("ingredients") or [])
    effects = _clean(row.get("effects") or [])
    indications = _clean(row.get("indications") or [])
    cautions = _clean(row.get("cautions") or [])
    adverse_reactions = _clean(row.get("adverse_reactions") or [])
    ingredient_cautions = []
    for pair in row.get("ingredient_cautions") or []:
        if not isinstance(pair, dict):
            continue
        ing = str(pair.get("ingredient") or "").strip()
        cau = str(pair.get("caution") or "").strip()
        if ing and cau:
            ingredient_cautions.append(f"{ing}→{cau}")

    sections = [f"药品：{row.get('drug_name') or ''}"]
    if ingredients:
        sections.append(f"组方成分：{'、'.join(ingredients)}")
    if effects:
        sections.append(f"相关功效：{'、'.join(effects)}")
    if indications:
        sections.append(f"相关症状/证候/疾病：{'、'.join(indications)}")
    if cautions:
        sections.append(f"禁忌/慎用人群：{'、'.join(cautions)}")
    if adverse_reactions:
        sections.append(f"不良反应：{'、'.join(adverse_reactions)}")
    if ingredient_cautions:
        sections.append(f"成分禁忌线索：{'；'.join(ingredient_cautions[:20])}")

    return Document(
        page_content="\n".join(sections),
        metadata={
            "node_id": row.get("node_id") or "",
            "doc_type": "graph_enrich",
            "search_source": "graph_enrich",
            "drug_name": row.get("drug_name") or "",
        },
    )


def make_graph_enrich_node(
    neo4j: Neo4jGraphTool,
    *,
    limit: int = 8,
) -> Callable[[OnlineQAState], dict]:
    """
    Forward-path graph enrichment node.

    Reads kg_anchor_id from state and fetches the full drug subgraph
    (ingredients / effects / indications / contraindications / adverse reactions).
    Returns empty graph_docs if kg_anchor_id is None (one-level fallback).
    """

    def graph_enrich_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        node_id = (state.get("kg_anchor_id") or "").strip()
        if not node_id:
            return {
                "graph_docs": [],
                "metrics": {**(state.get("metrics") or {}), "graph_enrich_seconds": time.time() - t0, "graph_enrich_rows": 0},
            }

        rows = _query_rows(neo4j, _TEMPLATE_BY_NODE_ID, {"node_id": node_id, "limit": limit})
        docs = [_row_to_doc(r) for r in rows] if rows else []

        return {
            "graph_docs": docs,
            "metrics": {
                **(state.get("metrics") or {}),
                "graph_enrich_seconds": time.time() - t0,
                "graph_enrich_rows": len(rows),
            },
        }

    return graph_enrich_node
