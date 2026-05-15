from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.app.config import GraphRAGConfig

# Allowed depth values for expand_by_node_id.
# Neo4j 5 does not support parameterized variable-length path depth,
# so depth is injected via safe f-string after whitelist validation.
_DEPTH_WHITELIST: frozenset[int] = frozenset({1, 2, 3})

# --- Stage 3 Cypher templates ---

_CYPHER_CAUTION_FOR = """
MATCH (pop:Concept {conceptType: 'Population'})-[:CAUTION_FOR]-(drug:Concept {conceptType: 'Drug'})
WHERE pop.name CONTAINS $name
RETURN drug.nodeId AS drug_node_id, drug.name AS drug_name, pop.name AS reason_label
LIMIT $limit
"""

_CYPHER_HAS_ADVERSE = """
MATCH (adv:Concept {conceptType: 'AdverseReaction'})-[:HAS_ADVERSE_REACTION]-(drug:Concept {conceptType: 'Drug'})
WHERE adv.name CONTAINS $name
RETURN drug.nodeId AS drug_node_id, drug.name AS drug_name, adv.name AS reason_label
LIMIT $limit
"""

_CYPHER_INDICATES_SYM = """
MATCH (sym:Concept {conceptType: 'Symptom'})-[:INDICATES]-(drug:Concept {conceptType: 'Drug'})
WHERE sym.name CONTAINS $name
RETURN drug.nodeId AS drug_node_id, drug.name AS drug_name, sym.name AS reason_label
LIMIT $limit
"""


class _Neo4jGraphToolInput(BaseModel):
    """Neo4j 图工具输入"""
    query: str = Field(..., description="自然语言查询或 node_id:<id> 形式的节点扩展请求")
    max_depth: int = Field(2, ge=1, le=4)
    max_nodes: int = Field(50, ge=1, le=500)


@lru_cache(maxsize=8)
def _get_graph(uri: str, user: str, password: str, database: str) -> Neo4jGraph:
    """获取 Neo4j 图数据库连接"""
    from langchain_neo4j import Neo4jGraph

    return Neo4jGraph(url=uri, username=user, password=password, database=database)


def _infer_mode(query: str) -> Literal["subgraph", "multi_hop", "entity_relation"]:
    """根据查询推断检索模式"""
    q = query.strip()
    if any(x in q for x in ["路径", "怎么到", "如何到", "连到", "多跳"]):
        return "multi_hop"
    if any(x in q for x in ["关系", "有什么关系", "关联"]):
        return "entity_relation"
    return "subgraph"


def _extract_entities(query: str) -> tuple[str | None, str | None]:
    """从查询中提取实体"""
    q = query.strip()
    q = re.sub(r"\s+", "", q)
    if "到" in q:
        left, right = q.split("到", 1)
        return left or None, right or None
    if "和" in q:
        left, right = q.split("和", 1)
        return left or None, right or None
    return None, None


class Neo4jGraphTool(BaseTool):
    """Neo4j 图工具"""
    name: str = "neo4j_search"
    description: str = "在 Neo4j 中做子图/多跳/实体关系检索，返回 LangChain Documents"
    args_schema: type[BaseModel] = _Neo4jGraphToolInput

    def __init__(self, config: GraphRAGConfig):
        super().__init__()
        self._config = config
        try:
            self._graph = _get_graph(config.neo4j_uri, config.neo4j_user, config.neo4j_password, config.neo4j_database)
        except Exception:
            self._graph = None

    def _run(self, query: str, max_depth: int = 2, max_nodes: int = 50) -> list[Document]:
        if self._graph is None:
            return []
        q = (query or "").strip()
        if not q:
            return []

        if q.startswith("node_id:"):
            node_id = q.split("node_id:", 1)[1].strip()
            return self._expand_one_hop(node_id=node_id, max_nodes=max_nodes)

        mode = _infer_mode(q)
        if mode == "entity_relation":
            a, b = _extract_entities(q)
            if a and b:
                return self._entity_relation(a=a, b=b, max_nodes=max_nodes)
            return self._subgraph_keywords(keywords=q, max_depth=1, max_nodes=max_nodes)

        if mode == "multi_hop":
            a, b = _extract_entities(q)
            if a and b:
                return self._multi_hop(a=a, b=b, max_depth=max_depth, max_nodes=max_nodes)
            return self._subgraph_keywords(keywords=q, max_depth=max_depth, max_nodes=max_nodes)

        return self._subgraph_keywords(keywords=q, max_depth=max_depth, max_nodes=max_nodes)

    def _expand_one_hop(self, node_id: str, max_nodes: int) -> list[Document]:
        cypher = """
        MATCH (n {nodeId: $node_id})-[r]-(m)
        RETURN n.nodeId AS source_id,
               coalesce(n.conceptType, head(labels(n))) AS source_type,
               type(r) AS rel_type,
               m.nodeId AS target_id,
               coalesce(m.conceptType, head(labels(m))) AS target_type,
               coalesce(n.name, n.description, "") AS source_name,
               coalesce(m.name, m.description, "") AS target_name
        LIMIT $limit
        """
        rows = self._graph.query(cypher, {"node_id": node_id, "limit": int(max_nodes)})
        docs: list[Document] = []
        for row in rows or []:
            page = f"{row.get('source_name') or row.get('source_id')} -[{row.get('rel_type')}]- {row.get('target_name') or row.get('target_id')}"
            docs.append(
                Document(
                    page_content=page,
                    metadata={
                        "node_id": row.get("target_id") or "",
                        "node_type": row.get("target_type") or "Unknown",
                        "path_length": 1,
                        "source_node_id": row.get("source_id") or "",
                    },
                )
            )
        return docs

    def _subgraph_keywords(self, keywords: str, max_depth: int, max_nodes: int) -> list[Document]:
        cypher = """
        MATCH (r)
        WHERE coalesce(r.conceptType, head(labels(r)), "") = "Recipe"
          AND (
            r.name CONTAINS $kw
            OR coalesce(r.description, "") CONTAINS $kw
            OR coalesce(r.preferredTerm, "") CONTAINS $kw
          )
        WITH r
        OPTIONAL MATCH (r)-[]->(i)
        WHERE coalesce(i.conceptType, head(labels(i)), "") = "Ingredient"
        OPTIONAL MATCH (r)-[]->(s)
        WHERE coalesce(s.conceptType, head(labels(s)), "") = "CookingStep"
        WITH r, collect(DISTINCT i.name)[..30] AS ingredients, collect(DISTINCT s.description)[..30] AS steps
        RETURN r.nodeId AS node_id,
               r.name AS recipe_name,
               r.category AS category,
               r.cuisineType AS cuisine_type,
               r.difficulty AS difficulty,
               ingredients AS ingredients,
               steps AS steps
        LIMIT $limit
        """
        rows = self._graph.query(cypher, {"kw": keywords, "limit": int(max_nodes)})
        docs: list[Document] = []
        for row in rows or []:
            recipe_name = row.get("recipe_name") or "未知菜谱"
            ingredients = row.get("ingredients") or []
            steps = row.get("steps") or []
            content = "\n".join(
                [
                    f"# {recipe_name}",
                    "",
                    "## 所需食材",
                    *[f"- {x}" for x in ingredients if x],
                    "",
                    "## 关键步骤",
                    *[f"- {x}" for x in steps if x],
                ]
            ).strip()
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "node_id": row.get("node_id") or "",
                        "node_type": "Recipe",
                        "path_length": int(max_depth),
                        "recipe_name": recipe_name,
                        "category": row.get("category") or "",
                        "cuisine_type": row.get("cuisine_type") or "",
                        "difficulty": row.get("difficulty") or 0,
                        "doc_type": "recipe",
                    },
                )
            )
        return docs

    def _entity_relation(self, a: str, b: str, max_nodes: int) -> list[Document]:
        cypher = """
        MATCH (x) WHERE coalesce(x.name, x.description, "") CONTAINS $a
        MATCH (y) WHERE coalesce(y.name, y.description, "") CONTAINS $b
        WITH x, y
        MATCH p=(x)-[r*1..2]-(y)
        RETURN [n IN nodes(p) | {id: n.nodeId, node_type: coalesce(n.conceptType, head(labels(n))), name: coalesce(n.name, n.description, "")}] AS ns,
               [rel IN relationships(p) | type(rel)] AS rs,
               length(p) AS path_length
        LIMIT $limit
        """
        rows = self._graph.query(cypher, {"a": a, "b": b, "limit": int(max_nodes)})
        docs: list[Document] = []
        for row in rows or []:
            nodes = row.get("ns") or []
            rels = row.get("rs") or []
            parts: list[str] = []
            for i, n in enumerate(nodes):
                parts.append(n.get("name") or n.get("id") or "Unknown")
                if i < len(rels):
                    parts.append(f"-[{rels[i]}]->")
            docs.append(
                Document(
                    page_content=" ".join(parts),
                    metadata={
                        "node_id": (nodes[-1].get("id") if nodes else "") or "",
                        "node_type": (nodes[-1].get("node_type") if nodes else "Unknown"),
                        "path_length": int(row.get("path_length") or 0),
                    },
                )
            )
        return docs

    # ------------------------------------------------------------------
    # Stage 3 public helpers
    # ------------------------------------------------------------------

    def expand_by_node_id(
        self,
        node_id: str,
        depth: int = 2,
        rel_types: list[str] | None = None,
    ) -> list[dict]:
        """Fetch the TCM drug subgraph by nodeId, traversing up to `depth` hops.

        Depth is validated against a whitelist {1, 2, 3} to prevent injection.
        Neo4j 5 does not support parameterized variable-length path depth, so
        the value is safely injected via f-string after whitelist check.

        Args:
            node_id:   The Concept.nodeId to expand from.
            depth:     Number of hops (1–3). Values outside whitelist default to 2.
            rel_types: Optional list of relationship type names to keep
                       (Python-side filter; all rels are fetched then filtered).

        Returns:
            List of dicts with keys:
                source_id, source_name, source_type,
                rel_path (list of rel type strings along path),
                target_id, target_name, target_type
        """
        if self._graph is None:
            return []
        safe_depth = int(depth) if int(depth) in _DEPTH_WHITELIST else 2
        cypher = f"""
        MATCH (n:Concept {{nodeId: $node_id}})-[r*1..{safe_depth}]-(m:Concept)
        RETURN
            n.nodeId AS source_id,
            coalesce(n.name, n.description, '') AS source_name,
            coalesce(n.conceptType, head(labels(n))) AS source_type,
            [rel IN r | type(rel)] AS rel_path,
            m.nodeId AS target_id,
            coalesce(m.name, m.description, '') AS target_name,
            coalesce(m.conceptType, head(labels(m))) AS target_type
        LIMIT $limit
        """
        try:
            rows = self._graph.query(cypher, {"node_id": str(node_id), "limit": 100})
        except Exception:
            return []

        result: list[dict] = []
        for row in rows or []:
            rel_path: list[str] = row.get("rel_path") or []
            # Python-side rel_types filter when caller requests specific edge types
            if rel_types and not any(r in rel_types for r in rel_path):
                continue
            result.append({
                "source_id": row.get("source_id") or "",
                "source_name": row.get("source_name") or "",
                "source_type": row.get("source_type") or "",
                "rel_path": rel_path,
                "target_id": row.get("target_id") or "",
                "target_name": row.get("target_name") or "",
                "target_type": row.get("target_type") or "",
            })
        return result

    def find_drugs_by_caution(
        self,
        population_name: str | None = None,
        symptom_name: str | None = None,
        adverse_name: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find drugs linked to a population caution, symptom indication, or adverse reaction.

        Each non-None argument triggers an independent Cypher query.
        Results across queries are merged and deduplicated by drug_node_id.

        Args:
            population_name: Population group to check CAUTION_FOR (e.g. "孕妇").
            symptom_name:    Symptom/disease for INDICATES direction (e.g. "月经不调").
            adverse_name:    Adverse reaction for HAS_ADVERSE_REACTION (e.g. "头晕").
            limit:           Per-query row limit.

        Returns:
            List of dicts: {drug_node_id, drug_name, reason_path: [label, rel_type]}
        """
        if self._graph is None:
            return []

        raw: list[dict] = []
        params = {"limit": int(limit)}

        if population_name:
            try:
                for row in (self._graph.query(_CYPHER_CAUTION_FOR, {**params, "name": population_name}) or []):
                    raw.append({
                        "drug_node_id": row.get("drug_node_id") or "",
                        "drug_name": row.get("drug_name") or "",
                        "reason_path": [row.get("reason_label") or "", "CAUTION_FOR"],
                    })
            except Exception:
                pass

        if adverse_name:
            try:
                for row in (self._graph.query(_CYPHER_HAS_ADVERSE, {**params, "name": adverse_name}) or []):
                    raw.append({
                        "drug_node_id": row.get("drug_node_id") or "",
                        "drug_name": row.get("drug_name") or "",
                        "reason_path": [row.get("reason_label") or "", "HAS_ADVERSE_REACTION"],
                    })
            except Exception:
                pass

        if symptom_name:
            try:
                for row in (self._graph.query(_CYPHER_INDICATES_SYM, {**params, "name": symptom_name}) or []):
                    raw.append({
                        "drug_node_id": row.get("drug_node_id") or "",
                        "drug_name": row.get("drug_name") or "",
                        "reason_path": [row.get("reason_label") or "", "INDICATES"],
                    })
            except Exception:
                pass

        # Deduplicate by drug_node_id, preserve first occurrence
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in raw:
            key = item["drug_node_id"]
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def _multi_hop(self, a: str, b: str, max_depth: int, max_nodes: int) -> list[Document]:
        cypher = """
        MATCH (x) WHERE coalesce(x.name, x.description, "") CONTAINS $a
        MATCH (y) WHERE coalesce(y.name, y.description, "") CONTAINS $b
        WITH x, y
        MATCH p=(x)-[r*1..$depth]-(y)
        RETURN [n IN nodes(p) | {id: n.nodeId, node_type: coalesce(n.conceptType, head(labels(n))), name: coalesce(n.name, n.description, "")}] AS ns,
               length(p) AS path_length
        ORDER BY length(p) ASC
        LIMIT $limit
        """
        rows = self._graph.query(cypher, {"a": a, "b": b, "depth": int(max_depth), "limit": int(max_nodes)})
        docs: list[Document] = []
        for row in rows or []:
            nodes = row.get("ns") or []
            chain = " -> ".join([(n.get("name") or n.get("id") or "Unknown") for n in nodes])
            docs.append(
                Document(
                    page_content=chain,
                    metadata={
                        "node_id": (nodes[-1].get("id") if nodes else "") or "",
                        "node_type": (nodes[-1].get("node_type") if nodes else "Unknown"),
                        "path_length": int(row.get("path_length") or 0),
                    },
                )
            )
        return docs
