# Stage 3 — Data Layer Requirements

## Context

Stage 2 introduced the new dual-flow LangGraph architecture with `graph_enrich_node` and `milvus_fetch_node`, both of which require query-side helpers that did not exist in the original tool layer.

**Constraint**: no data re-ingestion, no schema changes.  Only query-side helpers are added/fixed.

---

## Requirements

### NEO4J-01 — Multi-hop Subgraph Expansion

`Neo4jGraphTool.expand_by_node_id(node_id, depth=2, rel_types=None) → list[dict]`

- Returns every `(Concept)-[r*1..depth]-(Concept)` neighbour reachable from `node_id`.
- `depth` must be validated against whitelist `{1, 2, 3}` before being interpolated into Cypher (Neo4j 5 limitation: parameterized variable-length depth is unsupported).
- Optional `rel_types` list for Python-side filtering of returned paths.
- Returns structured dicts (not `Document`), so callers can decide how to surface them.
- Fail-safe: returns `[]` on connection error or missing graph.

### NEO4J-02 — Caution / Condition Drug Lookup

`Neo4jGraphTool.find_drugs_by_caution(population_name, symptom_name, adverse_name, limit=20) → list[dict]`

- Accepts any combination of the three filter dimensions.
- Issues up to three independent Cypher queries against the existing TCM schema relations:
  - `CAUTION_FOR` (Population → Drug)
  - `HAS_ADVERSE_REACTION` (AdverseReaction → Drug)
  - `INDICATES` (Symptom → Drug)
- Results merged and deduplicated by `drug_node_id`.
- Returns `[{drug_node_id, drug_name, reason_path: [label, rel_type]}]`.
- No args → empty list returned without issuing any query.
- Per-query exceptions swallowed (partial results preferred over total failure).

### MILVUS-01 — Point-Get by ID

`MilvusVectorTool.point_get(ids, by="chunk_id", limit=100) → list[Document]`

- Builds scalar filter `chunk_id in [...]` or `node_id in [...]` and calls `client.query()`.
- Returns `Document` objects with `metadata["search_source"] = "point_get"`.
- Empty `ids` list → returns `[]` without querying.
- Rows with empty `text` are skipped.
- Exceptions swallowed, returns `[]`.

### BM25-01 — Configurable Corpus Filter

`BM25Tool(filter_expr=None)` constructor parameter.

- `None` (default) → `_fetch_from_milvus` queries Milvus with `filter=""` (full corpus, all doc_types).
- Non-None string → forwarded as the Milvus query filter.
- Replaces the hardcoded `filter='doc_type == "drug"'` that excluded TCM literature and health QA docs from BM25 retrieval.
- Existing callers that pass no `filter_expr` silently upgrade to full-corpus indexing.

### VERIFY-01 — Data Layer Smoke Check Script

`scripts/verify_data_layer.py`

- Pings Neo4j: `MATCH (n:Concept) RETURN n.conceptType, count(*)`.
- Pings Milvus: checks both `tcm_knowledge` and `health_knowledge` collections exist and reports `row_count`.
- Writes `benchmark/data_layer_report.json`.
- Exit code 0 = all OK, 1 = any failure.

---

## Out of Scope (Stage 3)

- Milvus sparse-vector (server-side hybrid) — schema change required; deferred.
- Re-ingestion or schema modification.
- Changing `drug_entity_resolve_node` (already used internally by `neo4j_search_node` in Stage 2).
