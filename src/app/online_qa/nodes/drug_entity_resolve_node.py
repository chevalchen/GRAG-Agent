from __future__ import annotations

from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.utils.drug_name_normalizer import normalize_drug_name


def _candidate_mentions(query: str, keywords: list[str]) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    raw_candidates = [query, *list(keywords or [])]
    intent_words = {"适应症", "主治", "功效", "禁忌", "不良反应", "副作用", "注意事项", "是什么", "什么", "的"}
    for item in raw_candidates:
        text = str(item or "").strip()
        if not text:
            continue
        reduced = text
        for word in intent_words:
            reduced = reduced.replace(word, "")
        reduced = reduced.replace("？", "").replace("?", "").strip()
        for candidate in [text, reduced]:
            s = str(candidate or "").strip()
            if len(s) < 2 or s in seen:
                continue
            seen.add(s)
            mentions.append(s)
    return mentions


def _resolve_with_graph(neo4j: Neo4jGraphTool, candidate: str) -> dict | None:
    cfg = getattr(neo4j, "_config", None)
    if cfg is None:
        return None
    normalized = normalize_drug_name(candidate)
    if not normalized:
        return None
    cypher = """
    MATCH (d:Concept {conceptType:'Drug'})
    WHERE d.normalizedName = $normalized
       OR $normalized IN coalesce(d.aliasNorms, [])
    RETURN d.nodeId AS node_id,
           coalesce(d.canonicalName, d.name, "") AS canonical_name,
           coalesce(d.normalizedName, "") AS normalized_name,
           coalesce(d.aliases, []) AS aliases,
           coalesce(d.aliasNorms, []) AS alias_norms,
           CASE
               WHEN d.normalizedName = $normalized THEN 'normalized_name'
               ELSE 'alias_norm'
           END AS match_type
    LIMIT 5
    """
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
        session = driver.session(database=cfg.neo4j_database)
        rows = session.run(cypher, normalized=normalized).data() or []
        session.close()
        driver.close()
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0]
    aliases = [str(x).strip() for x in (row.get("aliases") or []) if str(x).strip()]
    matched_alias = candidate
    for alias in aliases:
        if normalize_drug_name(alias) == normalized:
            matched_alias = alias
            break
    return {
        "canonical_name": str(row.get("canonical_name") or "").strip(),
        "normalized_name": str(row.get("normalized_name") or normalized).strip(),
        "node_id": str(row.get("node_id") or "").strip(),
        "matched_alias": matched_alias,
        "match_type": str(row.get("match_type") or "alias_norm").strip(),
    }


def make_drug_entity_resolve_node(neo4j: Neo4jGraphTool) -> Callable[[OnlineQAState], dict]:
    def drug_entity_resolve_node(state: OnlineQAState) -> dict:
        routing = state.get("routing") or {}
        if not (routing.get("use_graph") or routing.get("use_drug_vec")):
            return {"resolved_drug": None}
        query = str(state.get("query") or "").strip()
        if not query:
            return {"resolved_drug": None}
        mentions = _candidate_mentions(query, list(routing.get("keywords") or []))
        for candidate in mentions:
            resolved = _resolve_with_graph(neo4j, candidate)
            if resolved and resolved.get("node_id"):
                return {"resolved_drug": resolved}
        return {"resolved_drug": None}

    return drug_entity_resolve_node
