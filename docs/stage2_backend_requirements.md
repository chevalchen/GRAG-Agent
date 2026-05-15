# Stage 2 Backend Requirements: LangGraph Flow Refactor

## 1. Background

Current flow is linear: supervisor → 4-way concurrent retrieval → RRF → rerank → answer.
Problems:
- No intent-based routing; all queries run all 4 retrieval branches regardless of need
- Graph is used as one of 4 parallel branches, not as an enrichment layer
- No sufficiency check; poor recall goes directly to generation
- No streaming; answer arrives all at once

## 2. Target Architecture

```
Router
  ├─ symptom_to_drug → MilvusSearch → GraphEnrich ─┐
  └─ conditions_exact → Neo4jSearch → MilvusFetch  ─┤
                                                    ▼
                                              Fuse (RRF)
                                                    │
                                               Rerank
                                                    │
                                             Sufficiency
                                          ┌────┴────┐
                                      sufficient  insufficient
                                          │            │
                                        Answer      Router (loop, max 1)
                                          │
                                         END
```

## 3. Functional Requirements

### 3.1 Intent Routing
- 4 intent classes: `chat` | `theory` | `symptom_to_drug` | `conditions_exact`
- `chat`: non-medical small talk → direct fallback answer, skip retrieval
- `theory`: TCM theory / general education → symptom_to_drug path (use Milvus literature branch)
- `symptom_to_drug`: "what drug for this symptom?" → forward flow (Milvus + GraphEnrich)
- `conditions_exact`: "what drugs conflict / who should avoid X?" → reverse flow (Neo4j + MilvusFetch)
- Router also produces:
  - `rewritten_query`: coreference resolution + disambiguation (single LLM step)
  - `extracted_entities`: `{drugs:[], diseases:[], symptoms:[]}` extracted from query JSON

### 3.2 Forward Flow (symptom_to_drug / theory)
- `milvus_search_node`: client-side hybrid (dense BM25), no `doc_type` constraint, sets `kg_anchor_id` from top-1 hit's `node_id`
- `graph_enrich_node`: reads `kg_anchor_id`, multi-hop (depth 2) via `TEMPLATE_DRUG_BY_NODE_ID`, returns structured `graph_docs`

### 3.3 Reverse Flow (conditions_exact)
- `neo4j_search_node`: entity-based Cypher using `CAUTION_FOR` / `HAS_ADVERSE_REACTION` / `INDICATES` relations, returns structured docs and `milvus_doc_ids`
- `milvus_fetch_node`: point-get by `node_id in [...]` via `client.query()`, returns `milvus_docs`

### 3.4 Fusion & Ranking
- `fuse_node`: 2-bucket RRF (`milvus_docs`, `graph_docs`), k=60
- `rerank_node`: unchanged — CrossEncoder reranking, skippable for simple queries

### 3.5 Sufficiency Check
- `sufficiency_node`: JSON-LLM call, judges whether `docs_final` covers query's key facets
- Returns `sufficient` or `insufficient`
- `iteration_count` incremented on `insufficient`; if `iteration_count >= max_iters` (default 1), force `sufficient` → generate

### 3.6 Answer Generation (Streaming)
- `answer_node` becomes async
- Main answer via `llm._llm.astream()` — tokens yielded via SSE
- After main answer, a second small LLM call generates 3 `next_questions`
- Inline safety check (十八反/十九畏 keyword list) before yielding; appends disclaimer, does NOT block

### 3.7 SSE Endpoint
- `POST /api/v1/qa/stream` upgraded to use `graph.astream_events(version="v2")`
- Filters `on_chat_model_stream` events from `answer_node` for `token` SSE events
- Emits `suggest` event with `next_questions` after answer completes
- Emits `done` event with `session_id` and `trace_id`

## 4. One-Level Fallbacks Only

| Failure | Fallback |
|---------|---------|
| Router JSON parse error | Default to `symptom_to_drug` |
| `kg_anchor_id` is None | `graph_enrich_node` returns empty docs, skip |
| Neo4jSearch 0 hits | `milvus_fetch_node` falls back to dense top-k |
| GraphEnrich 0 hits | Empty `graph_docs`, continue to fuse |
| `iteration_count >= max_iters` | Force generate with available context |

## 5. State Contract

See `docs/stage2_backend_design.md` for full state definition.

## 6. Non-Functional

- All node code: English comments
- Existing rerank_node: no changes
- Old nodes archived as `_legacy.py` after refactor
- No new Milvus schema changes (no sparse field)
- No Neo4j schema changes
