from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.app.config import GraphRAGConfig


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
