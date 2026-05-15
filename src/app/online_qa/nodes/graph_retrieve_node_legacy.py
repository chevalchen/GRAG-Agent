from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool

TEMPLATE_DRUG_CHAIN = """
MATCH (d:Concept {conceptType:'Drug'})
WHERE any(kw IN $drug_keywords WHERE d.name CONTAINS kw)
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

TEMPLATE_DRUG_FALLBACK = """
MATCH (d:Concept {conceptType:'Drug'})
WHERE any(kw IN $all_keywords WHERE d.name CONTAINS kw)
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

TEMPLATE_DRUG_BY_NODE_ID = """
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


def _clean(values: list) -> list[str]:
    out = []
    for x in values or []:
        s = str(x or "").strip()
        if not s:
            continue
        out.append(s)
    return out


def _split_keywords(routing: dict, query: str) -> tuple[list[str], list[str], str]:
    intent_words = {"适应症", "主治", "功效", "禁忌", "不良反应", "副作用", "注意事项", "是什么", "什么"}
    keywords = _clean(list(routing.get("keywords") or []))
    all_keywords = list(dict.fromkeys([*keywords, query.strip()]))
    drug_keywords = [k for k in keywords if k not in intent_words and len(k) >= 2]
    if not drug_keywords:
        candidate = query
        for w in intent_words:
            candidate = candidate.replace(w, "")
        candidate = candidate.replace("的", "").replace("？", "").replace("?", "").strip()
        if candidate:
            drug_keywords = [candidate]
    strategy = "fallback_indication" if any(x in query for x in ["适应症", "主治"]) else "exact_drug"
    return drug_keywords, _clean(all_keywords), strategy


def _query_rows(neo4j: Neo4jGraphTool, cypher: str, params: dict) -> list[dict]:
    cfg = getattr(neo4j, "_config", None)
    if cfg is not None:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                cfg.neo4j_uri,
                auth=(cfg.neo4j_user, cfg.neo4j_password),
            )
            session = driver.session(database=cfg.neo4j_database)
            rows = session.run(cypher, **params).data() or []
            session.close()
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


def make_graph_retrieve_node(
    neo4j: Neo4jGraphTool,
    *,
    top_k: int,
    expand_factor: float = 3.0,
    max_graph_rows: int = 24,
    graph_fallback_max_nodes: int = 48,
) -> Callable[[OnlineQAState], dict]:
    """
    构建图检索节点
    
    Args:
        neo4j: Neo4j 图工具
        top_k: 检索文档数量
    Returns:
        图检索节点
    """
    def graph_retrieve_node(state: OnlineQAState) -> dict:
        """
        图检索节点
        
        Args:
            state: 在线问答状态
            
        Returns:
            图检索结果
        """
        t0 = time.time()
        routing = state.get("routing") or {}
        if not routing.get("use_graph"):
            return {"graph_docs": [], "metrics": {**(state.get("metrics") or {}), "graph_retrieve_seconds": 0.0}}
        keywords = list(routing.get("keywords") or [])
        query = (state.get("query") or "").strip()
        complexity = str(routing.get("complexity_level") or "complex")
        resolved_drug = state.get("resolved_drug") or {}
        drug_keywords, all_keywords, strategy = _split_keywords(routing, query)
        expanded_k = max(int(top_k * max(expand_factor, 1.0)), int(top_k))
        row_limit = min(expanded_k, int(max_graph_rows))
        rows = []
        template_used = "exact_drug"
        if resolved_drug.get("node_id"):
            rows = _query_rows(
                neo4j,
                TEMPLATE_DRUG_BY_NODE_ID,
                {"node_id": str(resolved_drug.get("node_id") or ""), "limit": row_limit},
            )
            template_used = "resolved_node_id"
        if not rows:
            rows = _query_rows(neo4j, TEMPLATE_DRUG_CHAIN, {"drug_keywords": drug_keywords, "limit": row_limit})
            template_used = "exact_drug"
        if not rows:
            rows = _query_rows(neo4j, TEMPLATE_DRUG_FALLBACK, {"all_keywords": all_keywords, "limit": row_limit})
            template_used = strategy
        if not rows:
            # 简单问句或已进行关键词回退后，不再继续触发高成本全图回退查询。
            if complexity != "complex":
                metrics = {
                    **(state.get("metrics") or {}),
                    "graph_rows": 0,
                    "graph_docs": 0,
                    "graph_template": "skipped_fallback",
                    "graph_fallback_invoked": False,
                    "graph_retrieve_seconds": time.time() - t0,
                }
                return {"graph_docs": [], "metrics": metrics}
            docs = neo4j.invoke({"query": query, "max_depth": 2, "max_nodes": int(graph_fallback_max_nodes)}) or []
            for d in docs:
                d.metadata = {**(d.metadata or {}), "search_source": "neo4j", "template_used": "neo4j_tool_fallback"}
            metrics = {
                **(state.get("metrics") or {}),
                "graph_rows": len(docs),
                "graph_docs": len(docs),
                "graph_template": "neo4j_tool_fallback",
                "graph_fallback_invoked": True,
                "graph_retrieve_seconds": time.time() - t0,
            }
            return {"graph_docs": docs, "metrics": metrics}
        docs: list[Document] = []
        for row in rows or []:
            ingredients = _clean(row.get("ingredients") or [])
            effects = _clean(row.get("effects") or [])
            indications = _clean(row.get("indications") or [])
            cautions = _clean(row.get("cautions") or [])
            adverse_reactions = _clean(row.get("adverse_reactions") or [])
            ingredient_cautions = []
            for pair in row.get("ingredient_cautions") or []:
                if not isinstance(pair, dict):
                    continue
                ingredient = str(pair.get("ingredient") or "").strip()
                caution = str(pair.get("caution") or "").strip()
                if ingredient and caution:
                    ingredient_cautions.append(f"{ingredient}→{caution}")
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
            page = "\n".join(sections)
            docs.append(
                Document(
                    page_content=page,
                    metadata={
                        "node_id": row.get("node_id") or "",
                        "doc_type": "graph_chain",
                        "search_source": "neo4j",
                        "drug_name": row.get("drug_name") or "",
                        "canonical_name": resolved_drug.get("canonical_name") or row.get("drug_name") or "",
                        "match_type": resolved_drug.get("match_type") or "",
                        "resolved_drug": resolved_drug or None,
                        "template_used": template_used,
                    },
                )
            )
        metrics = {
            **(state.get("metrics") or {}),
            "graph_rows": len(rows or []),
            "graph_docs": len(docs),
            "graph_template": template_used,
            "graph_fallback_invoked": False,
            "graph_retrieve_seconds": time.time() - t0,
        }
        return {"graph_docs": docs, "metrics": metrics}

    return graph_retrieve_node
