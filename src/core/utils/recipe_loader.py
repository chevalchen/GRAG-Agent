from __future__ import annotations

from langchain_core.documents import Document


def load_recipe_docs(config, *, limit: int = 5000) -> list[Document]:
    try:
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=config.neo4j_uri,
            username=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        rows = graph.query(
            """
            MATCH (r)
            WHERE coalesce(r.conceptType, head(labels(r)), "") = "Recipe"
            OPTIONAL MATCH (r)-[]->(i)
            WHERE coalesce(i.conceptType, head(labels(i)), "") = "Ingredient"
            OPTIONAL MATCH (r)-[]->(s)
            WHERE coalesce(s.conceptType, head(labels(s)), "") = "CookingStep"
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

    chunk_size = int(config.chunk_size)
    chunk_overlap = int(config.chunk_overlap)
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = 0
    step = max(1, chunk_size - chunk_overlap)

    docs: list[Document] = []
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
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={**base_meta, "chunk_id": chunk_id, "parent_id": base_meta["node_id"]},
                )
            )
    return docs
