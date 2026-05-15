# Stage 3 — Data Layer Design

## Files Modified

| File | Change |
|------|--------|
| `src/core/tools/graph/neo4j_tool.py` | Added `expand_by_node_id()`, `find_drugs_by_caution()`, Cypher templates, `_DEPTH_WHITELIST` |
| `src/core/tools/vector/milvus_tool.py` | Added `point_get()` |
| `src/core/tools/retrieval/bm25_tool.py` | Added `filter_expr` constructor param; updated `_fetch_from_milvus()` |

## New Files

| File | Purpose |
|------|---------|
| `scripts/verify_data_layer.py` | One-shot data-layer smoke check + JSON report |
| `tests/core/test_neo4j_tool.py` | 11 unit tests for `expand_by_node_id` + `find_drugs_by_caution` |
| `tests/core/test_milvus_tool.py` | 8 unit tests for `point_get` |
| `tests/core/test_bm25_corpus.py` | 5 unit tests for configurable corpus filter |

---

## Design Decisions

### NEO4J-01: Depth Whitelist

Neo4j 5 removed support for parameterized variable-length path depth
(`MATCH (n)-[r*1..$depth]-(m)` is a syntax error). The only safe workaround is
f-string injection after strict whitelist validation.

```python
_DEPTH_WHITELIST = frozenset({1, 2, 3})
safe_depth = int(depth) if int(depth) in _DEPTH_WHITELIST else 2
cypher = f"MATCH (n:Concept {{nodeId: $node_id}})-[r*1..{safe_depth}]-(m:Concept) ..."
```

`node_id` is still passed as a Cypher parameter (`$node_id`) — no injection risk there.

### NEO4J-01: rel_types as Python-side filter

Filtering by relationship type in a variable-length path at the Cypher level requires
`APOC` or complex `WHERE ALL(r IN rels | type(r) IN [...])` predicates, and performance
is acceptable at the small row counts returned (≤100).  The simpler Python-side filter
keeps the Cypher readable and avoids APOC dependency.

### NEO4J-02: Per-query exception swallowing

Each of the three Cypher queries (`CAUTION_FOR`, `HAS_ADVERSE_REACTION`, `INDICATES`)
runs in its own try/except.  This means a broken index on one relation type still returns
results from the other two — partial results are preferable to total failure.

### MILVUS-01: `point_get` filter syntax

Milvus scalar filter for list membership:
```
chunk_id in ["id1", "id2", ...]
```
IDs are wrapped in double-quotes and comma-separated via Python f-string.  The field
name (`chunk_id` or `node_id`) is hardcoded per the `by` parameter — no injection risk.

### BM25-01: Full corpus default

The old hardcoded `filter='doc_type == "drug"'` meant BM25 only indexed drug profiles,
missing TCM literature and health QA documents.  This degraded recall for theory and
health-advice queries.

New default is `filter=""` (empty string = full corpus in pymilvus `MilvusClient.query`).
Callers that need the old behaviour can pass `filter_expr='doc_type == "drug"'`.

BM25 corpus is rebuilt lazily and cached to disk.  Changing `filter_expr` requires
`BM25Tool.rebuild()` or deleting the cache file to take effect.

### VERIFY-01: Script design

The script is intentionally standalone (`python -m scripts.verify_data_layer`) and
writes a machine-readable JSON report so it can be parsed by CI or benchmarking scripts.
Exit code 1 on any failure allows `&&`-chaining in shell scripts.

---

## Test Strategy

All three tool test files use `unittest.mock.MagicMock` for their respective clients
(`_get_graph`, `_get_milvus_client_by_uri`) — no live database connection required.

Key assertions:
- Cypher depth strings (`1..1`, `1..2`, `1..3`) present in the interpolated query.
- Invalid depth (0, 99) clamps to 2.
- Milvus `filter` kwarg contains the correct field name and quoted IDs.
- Empty inputs return `[]` without issuing any query.
- Exceptions are swallowed and return `[]` (fail-safe behavior).
