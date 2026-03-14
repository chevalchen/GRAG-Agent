from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.config import GraphRAGConfig
from src.app.offline_ingestion.state import OfflineIngestionState
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def _load_recipe_docs(config: GraphRAGConfig, *, limit: int = 5000) -> list[Document]:
    try:
        from langchain_community.graphs import Neo4jGraph

        graph = Neo4jGraph(
            url=config.neo4j_uri,
            username=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        rows = graph.query(
            """
            MATCH (r:Concept {conceptType: "Recipe"})
            OPTIONAL MATCH (r)-[]->(i:Concept {conceptType: "Ingredient"})
            OPTIONAL MATCH (r)-[]->(s:Concept {conceptType: "CookingStep"})
            WITH r,
                 collect(DISTINCT i.name)[..50] AS ingredients,
                 collect(DISTINCT s.description)[..50] AS steps
            RETURN r.nodeId AS node_id,
                   r.name AS recipe_name,
                   r.category AS category,
                   r.cuisineType AS cuisine_type,
                   r.difficulty AS difficulty,
                   ingredients AS ingredients,
                   steps AS steps
            LIMIT $limit
            """,
            {"limit": int(limit)},
        )
    except Exception:
        return []

    docs: list[Document] = []
    chunk_size = int(config.chunk_size)
    chunk_overlap = int(config.chunk_overlap)
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = 0
    step = max(1, chunk_size - chunk_overlap)

    for row in rows or []:
        recipe_name = row.get("recipe_name") or "未知菜谱"
        ingredients = row.get("ingredients") or []
        steps_list = row.get("steps") or []
        full_text = "\n".join(
            [
                f"# {recipe_name}",
                "",
                "## 所需食材",
                *[f"- {x}" for x in ingredients if x],
                "",
                "## 关键步骤",
                *[f"- {x}" for x in steps_list if x],
            ]
        ).strip()
        base_meta = {
            "node_id": row.get("node_id") or "",
            "node_type": "Recipe",
            "recipe_name": recipe_name,
            "category": row.get("category") or "",
            "cuisine_type": row.get("cuisine_type") or "",
            "difficulty": row.get("difficulty") or 0,
            "doc_type": "recipe",
        }
        chunks = [full_text[i : i + chunk_size] for i in range(0, len(full_text), step)] or [full_text]
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{base_meta['node_id']}_{idx}"
            docs.append(Document(page_content=chunk, metadata={**base_meta, "chunk_id": chunk_id, "parent_id": base_meta["node_id"]}))
    return docs


def make_milvus_index_node(config: GraphRAGConfig) -> Callable[[OfflineIngestionState], dict]:
    def milvus_index_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}

        start = time.time()
        milvus = MilvusVectorTool(config)
        if not milvus.ensure_collection(force_recreate=True):
            metrics = {**(state.get("metrics") or {}), "milvus_index_seconds": time.time() - start, "milvus_indexed": False}
            return {"indexed_count": 0, "metrics": metrics}

        docs = _load_recipe_docs(config)
        try:
            inserted = milvus.upsert_documents(docs)
            _ = milvus.create_index()
            _ = milvus.load_collection()
        except Exception:
            inserted = 0

        metrics = {**(state.get("metrics") or {}), "milvus_index_seconds": time.time() - start, "milvus_indexed": True}
        return {"indexed_count": int(inserted), "metrics": metrics}

    return milvus_index_node

