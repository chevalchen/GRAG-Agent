from __future__ import annotations

import csv
import os
import time
from collections.abc import Callable

from src.app.config import GraphRAGConfig
from src.app.offline_ingestion.state import OfflineIngestionState


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def make_neo4j_write_node(config: GraphRAGConfig, *, builder) -> Callable[[OfflineIngestionState], dict]:
    def neo4j_write_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}

        start = time.time()
        failed_writes = list(state.get("failed_writes") or [])

        try:
            builder.merge_all_batches()
        except Exception as e:
            failed_writes.append({"stage": "merge_batches", "error": str(e)})
            return {"failed_writes": failed_writes, "metrics": {**(state.get("metrics") or {}), "neo4j_write_seconds": time.time() - start}}

        concepts_path = os.path.join(state["output_dir"], "concepts.csv")
        relationships_path = os.path.join(state["output_dir"], "relationships.csv")
        concepts = _read_csv(concepts_path)
        relationships = _read_csv(relationships_path)

        try:
            from langchain_community.graphs import Neo4jGraph

            graph = Neo4jGraph(
                url=config.neo4j_uri,
                username=config.neo4j_user,
                password=config.neo4j_password,
                database=config.neo4j_database,
            )
        except Exception as e:
            failed_writes.append({"stage": "connect_neo4j", "error": str(e)})
            return {"failed_writes": failed_writes, "metrics": {**(state.get("metrics") or {}), "neo4j_write_seconds": time.time() - start}}

        try:
            graph.query(
                """
                UNWIND $rows AS row
                MERGE (n:Concept {nodeId: row.concept_id})
                SET n.conceptType = row.concept_type,
                    n.name = row.name,
                    n.preferredTerm = coalesce(row.preferred_term, ""),
                    n.category = coalesce(row.category, ""),
                    n.synonyms = coalesce(row.synonyms, ""),
                    n.difficulty = toInteger(coalesce(row.difficulty, "0")),
                    n.cuisineType = coalesce(row.cuisine_type, ""),
                    n.prepTime = coalesce(row.prep_time, ""),
                    n.cookTime = coalesce(row.cook_time, ""),
                    n.servings = coalesce(row.servings, ""),
                    n.tags = coalesce(row.tags, ""),
                    n.filePath = coalesce(row.file_path, ""),
                    n.amount = coalesce(row.amount, ""),
                    n.unit = coalesce(row.unit, ""),
                    n.isMain = coalesce(row.is_main, ""),
                    n.description = coalesce(row.description, ""),
                    n.stepNumber = coalesce(row.step_number, ""),
                    n.methods = coalesce(row.methods, ""),
                    n.tools = coalesce(row.tools, ""),
                    n.timeEstimate = coalesce(row.time_estimate, "")
                """,
                {"rows": concepts},
            )
        except Exception as e:
            failed_writes.append({"stage": "write_nodes", "error": str(e)})

        try:
            graph.query(
                """
                UNWIND $rows AS row
                MATCH (start:Concept {nodeId: row.source_id})
                MATCH (end:Concept {nodeId: row.target_id})
                CALL apoc.create.relationship(
                    start,
                    row.relationship_type,
                    {relationshipId: row.relationship_id, amount: coalesce(row.amount, ""), unit: coalesce(row.unit, ""), stepOrder: toInteger(coalesce(row.step_order, "0"))},
                    end
                ) YIELD rel
                RETURN count(rel) AS written
                """,
                {"rows": relationships},
            )
        except Exception as e:
            try:
                graph.query(
                    """
                    UNWIND $rows AS row
                    MATCH (start:Concept {nodeId: row.source_id})
                    MATCH (end:Concept {nodeId: row.target_id})
                    MERGE (start)-[r:RELATED]->(end)
                    SET r.relationshipType = row.relationship_type,
                        r.relationshipId = row.relationship_id,
                        r.amount = coalesce(row.amount, ""),
                        r.unit = coalesce(row.unit, ""),
                        r.stepOrder = toInteger(coalesce(row.step_order, "0"))
                    """,
                    {"rows": relationships},
                )
            except Exception as e2:
                failed_writes.append({"stage": "write_relationships", "error": str(e), "fallback_error": str(e2)})

        metrics = {**(state.get("metrics") or {}), "neo4j_write_seconds": time.time() - start, "neo4j_written": True}
        return {"failed_writes": failed_writes, "metrics": metrics}

    return neo4j_write_node

