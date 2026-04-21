from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

from src.app.config import GraphRAGConfig
from src.app.offline_ingestion.state import OfflineIngestionState


def _to_node_id(prefix: str, value: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def make_neo4j_write_node(config: GraphRAGConfig) -> Callable[[OfflineIngestionState], dict]:
    def neo4j_write_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}

        start = time.time()
        failed_writes = list(state.get("failed_writes") or [])
        parsed_records = list(state.get("parsed_records") or [])

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password),
            )
            session = driver.session(database=config.neo4j_database)
        except Exception as e:
            failed_writes.append({"stage": "connect_neo4j", "error": str(e)})
            return {"failed_writes": failed_writes, "metrics": {**(state.get("metrics") or {}), "neo4j_write_seconds": time.time() - start}}

        drug_rows: list[dict] = []
        ingredient_rows: list[dict] = []
        effect_rows: list[dict] = []
        symptom_rows: list[dict] = []
        disease_rows: list[dict] = []
        syndrome_rows: list[dict] = []
        population_rows: list[dict] = []
        adverse_rows: list[dict] = []
        contains_rows: list[dict] = []
        has_effect_rows: list[dict] = []
        indicates_rows: list[dict] = []
        caution_rows: list[dict] = []
        adverse_rel_rows: list[dict] = []
        ingredient_caution_rows: list[dict] = []
        seen_ingredient: set[str] = set()
        seen_effect: set[str] = set()
        seen_symptom: set[str] = set()
        seen_disease: set[str] = set()
        seen_syndrome: set[str] = set()
        seen_population: set[str] = set()
        seen_adverse: set[str] = set()

        for row in parsed_records:
            canonical_name = str(row.get("canonical_name") or row.get("drug_name") or "").strip()
            if not canonical_name:
                continue
            drug_id = str(row.get("node_id") or _to_node_id("drug", canonical_name)).strip()
            drug_rows.append(
                {
                    "nodeId": drug_id,
                    "name": canonical_name,
                    "canonicalName": canonical_name,
                    "normalizedName": str(row.get("normalized_name") or "").strip(),
                    "aliases": [str(x).strip() for x in (row.get("aliases") or []) if str(x).strip()],
                    "aliasNorms": [str(x).strip() for x in (row.get("alias_norms") or []) if str(x).strip()],
                }
            )
            for ingredient in row.get("ingredients") or []:
                ingredient_name = str(ingredient or "").strip()
                if not ingredient_name:
                    continue
                ingredient_id = _to_node_id("ing", ingredient_name)
                if ingredient_id not in seen_ingredient:
                    ingredient_rows.append({"nodeId": ingredient_id, "name": ingredient_name})
                    seen_ingredient.add(ingredient_id)
                contains_rows.append({"drugId": drug_id, "ingredientId": ingredient_id})

            for effect in row.get("effects") or []:
                effect_name = str(effect or "").strip()
                if not effect_name:
                    continue
                effect_id = _to_node_id("eff", effect_name)
                if effect_id not in seen_effect:
                    effect_rows.append({"nodeId": effect_id, "name": effect_name})
                    seen_effect.add(effect_id)
                has_effect_rows.append({"drugId": drug_id, "effectId": effect_id})

            for symptom in row.get("symptoms") or []:
                symptom_name = str(symptom or "").strip()
                if not symptom_name:
                    continue
                symptom_id = _to_node_id("sym", symptom_name)
                if symptom_id not in seen_symptom:
                    symptom_rows.append({"nodeId": symptom_id, "name": symptom_name})
                    seen_symptom.add(symptom_id)
                indicates_rows.append({"drugId": drug_id, "targetId": symptom_id})

            for disease in row.get("diseases") or []:
                disease_name = str(disease or "").strip()
                if not disease_name:
                    continue
                disease_id = _to_node_id("dis", disease_name)
                if disease_id not in seen_disease:
                    disease_rows.append({"nodeId": disease_id, "name": disease_name})
                    seen_disease.add(disease_id)
                indicates_rows.append({"drugId": drug_id, "targetId": disease_id})

            for syndrome in row.get("syndromes") or []:
                syndrome_name = str(syndrome or "").strip()
                if not syndrome_name:
                    continue
                syndrome_id = _to_node_id("syn", syndrome_name)
                if syndrome_id not in seen_syndrome:
                    syndrome_rows.append({"nodeId": syndrome_id, "name": syndrome_name})
                    seen_syndrome.add(syndrome_id)
                indicates_rows.append({"drugId": drug_id, "targetId": syndrome_id})

            for population in row.get("populations") or []:
                population_name = str(population or "").strip()
                if not population_name:
                    continue
                population_id = _to_node_id("pop", population_name)
                if population_id not in seen_population:
                    population_rows.append({"nodeId": population_id, "name": population_name})
                    seen_population.add(population_id)
                caution_rows.append({"drugId": drug_id, "populationId": population_id})
                for ingredient in row.get("ingredients") or []:
                    ingredient_name = str(ingredient or "").strip()
                    if not ingredient_name:
                        continue
                    ingredient_id = _to_node_id("ing", ingredient_name)
                    ingredient_caution_rows.append({"ingredientId": ingredient_id, "populationId": population_id})

            for adverse in row.get("adverse_reactions") or []:
                adverse_name = str(adverse or "").strip()
                if not adverse_name:
                    continue
                adverse_id = _to_node_id("adr", adverse_name)
                if adverse_id not in seen_adverse:
                    adverse_rows.append({"nodeId": adverse_id, "name": adverse_name})
                    seen_adverse.add(adverse_id)
                adverse_rel_rows.append({"drugId": drug_id, "adverseId": adverse_id})

        try:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (d:Concept {nodeId: row.nodeId})
                SET d.conceptType = 'Drug',
                    d.name = row.name,
                    d.canonicalName = row.canonicalName,
                    d.normalizedName = row.normalizedName,
                    d.aliases = row.aliases,
                    d.aliasNorms = row.aliasNorms
                """,
                rows=drug_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (i:Concept {nodeId: row.nodeId})
                SET i.conceptType = 'Ingredient',
                    i.name = row.name
                """,
                rows=ingredient_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (e:Concept {nodeId: row.nodeId})
                SET e.conceptType = 'Effect',
                    e.name = row.name
                """,
                rows=effect_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (s:Concept {nodeId: row.nodeId})
                SET s.conceptType = 'Symptom',
                    s.name = row.name
                """,
                rows=symptom_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (d:Concept {nodeId: row.nodeId})
                SET d.conceptType = 'Disease',
                    d.name = row.name
                """,
                rows=disease_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (s:Concept {nodeId: row.nodeId})
                SET s.conceptType = 'Syndrome',
                    s.name = row.name
                """,
                rows=syndrome_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (p:Concept {nodeId: row.nodeId})
                SET p.conceptType = 'Population',
                    p.name = row.name
                """,
                rows=population_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (a:Concept {nodeId: row.nodeId})
                SET a.conceptType = 'AdverseReaction',
                    a.name = row.name
                """,
                rows=adverse_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (d:Concept {nodeId: row.drugId})
                MATCH (i:Concept {nodeId: row.ingredientId})
                MERGE (d)-[:CONTAINS]->(i)
                """,
                rows=contains_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (d:Concept {nodeId: row.drugId})
                MATCH (e:Concept {nodeId: row.effectId})
                MERGE (d)-[:HAS_EFFECT]->(e)
                """,
                rows=has_effect_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (d:Concept {nodeId: row.drugId})
                MATCH (t:Concept {nodeId: row.targetId})
                MERGE (d)-[:INDICATES]->(t)
                """,
                rows=indicates_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (d:Concept {nodeId: row.drugId})
                MATCH (p:Concept {nodeId: row.populationId})
                MERGE (d)-[:CAUTION_FOR]->(p)
                """,
                rows=caution_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (i:Concept {nodeId: row.ingredientId})
                MATCH (p:Concept {nodeId: row.populationId})
                MERGE (i)-[:CAUTION_FOR]->(p)
                """,
                rows=ingredient_caution_rows,
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (d:Concept {nodeId: row.drugId})
                MATCH (a:Concept {nodeId: row.adverseId})
                MERGE (d)-[:HAS_ADVERSE_REACTION]->(a)
                """,
                rows=adverse_rel_rows,
            )
        except Exception as e:
            failed_writes.append({"stage": "write_tcm_graph", "error": str(e)})
        finally:
            try:
                session.close()
                driver.close()
            except Exception:
                pass

        metrics = {
            **(state.get("metrics") or {}),
            "neo4j_write_seconds": time.time() - start,
            "neo4j_written": len(drug_rows),
        }
        return {"failed_writes": failed_writes, "metrics": metrics}

    return neo4j_write_node
