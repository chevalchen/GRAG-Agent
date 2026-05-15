# Stage 2 Backend Design

## 1. State Definition (`src/app/online_qa/state.py`)

```python
from typing import TypedDict, Annotated, Literal
import operator

class RoutingSignal(TypedDict, total=False):
    intent: str            # chat | theory | symptom_to_drug | conditions_exact
    complexity_level: str  # simple | complex
    keywords: list[str]
    reason: str

class OnlineQAState(TypedDict, total=False):
    # --- input ---
    query: str
    rewritten_query: str                          # coreference-resolved query
    extracted_entities: dict[str, list[str]]      # {drugs:[], diseases:[], symptoms:[]}

    # --- routing ---
    routing: RoutingSignal | None

    # --- retrieval bridges ---
    kg_anchor_id: str | None          # forward: MilvusSearch → GraphEnrich
    milvus_doc_ids: list[str]         # reverse: Neo4jSearch → MilvusFetch

    # --- retrieval results ---
    milvus_docs: list[Document]
    graph_docs: list[Document]

    # --- fusion / rerank ---
    docs_rrf: list[Document]
    docs_final: list[Document]

    # --- control flow ---
    sufficiency: str | None           # "sufficient" | "insufficient"
    iteration_count: int              # starts 0; incremented on insufficient
    safety_blocked: bool
    safety_reason: str

    # --- output ---
    answer: str
    next_questions: list[str]

    # --- multi-turn ---
    history: Annotated[list[dict], operator.add]
    metrics: Annotated[dict, operator.or_]
    error: str | None
```

## 2. Node Inventory

| Node file | Role | Key inputs | Key outputs |
|-----------|------|-----------|------------|
| `router_node.py` | 4-way intent + rewrite | `query`, `history` | `routing`, `rewritten_query`, `extracted_entities` |
| `milvus_search_node.py` | Hybrid retrieval (dense+BM25) | `rewritten_query` | `milvus_docs`, `kg_anchor_id` |
| `graph_enrich_node.py` | Multi-hop graph enrichment | `kg_anchor_id` | `graph_docs` |
| `neo4j_search_node.py` | Conditions/contraindication Cypher | `rewritten_query`, `extracted_entities` | `graph_docs`, `milvus_doc_ids` |
| `milvus_fetch_node.py` | Point-get by node_id list | `milvus_doc_ids` | `milvus_docs` |
| `fuse_node.py` | 2-bucket RRF | `milvus_docs`, `graph_docs` | `docs_rrf` |
| `rerank_node.py` | CrossEncoder rerank (unchanged) | `docs_rrf`, `routing` | `docs_final` |
| `sufficiency_node.py` | JSON-LLM sufficiency judge | `docs_final`, `query`, `iteration_count` | `sufficiency`, `iteration_count` |
| `answer_node.py` | Async stream + next_questions | `docs_final`, `query`, `history` | `answer`, `next_questions` |

## 3. Graph Wiring (`online_qa_graph.py`)

```
START → router

router --conditional--> {
  "symptom_to_drug": milvus_search,
  "theory":          milvus_search,
  "conditions_exact": neo4j_search,
  "chat":            answer            # skip retrieval entirely
}

milvus_search → graph_enrich → fuse
neo4j_search  → milvus_fetch → fuse

fuse → rerank → sufficiency

sufficiency --conditional--> {
  "sufficient":   answer,
  "insufficient": router               # iteration_count already incremented
}

answer → END
```

Router entrance guard: if `iteration_count >= max_iters`, route directly to answer
regardless of intent.

## 4. SSE Protocol

```
event: token
data: {"chunk": "<text>"}

event: suggest
data: {"questions": ["q1", "q2", "q3"]}

event: done
data: {"session_id": "...", "trace_id": "..."}

event: error
data: {"code": "...", "message": "..."}
```

Implementation: `astream_events(version="v2")`, filter `on_chat_model_stream` where
`event["metadata"]["langgraph_node"] == "answer_node"`.

## 5. Key Reuse from Existing Code

| Existing pattern | Reused in |
|-----------------|-----------|
| `_extract_json()` | `router_node.py` |
| `_classify_complexity()` | `router_node.py` |
| `_merge_docs()` balanced strategy | `milvus_search_node.py` |
| `TEMPLATE_DRUG_BY_NODE_ID`, `_query_rows()` | `graph_enrich_node.py` |
| `_resolve_with_graph()` Cypher | `neo4j_search_node.py` |
| `_rrf()` | `fuse_node.py` (unchanged logic, new bucket names) |
| `_build_context_with_budget()` | `answer_node.py` |
| `LLMGenerationTool.stream_text()` | `answer_node.py` (async path) |

## 6. Inline Safety Check

十八反 (18 incompatibilities) + 十九畏 (19 mutual antagonisms) keyword list in
`answer_node.py`. If answer text contains any pair, set `safety_blocked=True`,
append `\n\n⚠️ 注意：本回答涉及配伍禁忌，请在中医医师指导下使用。` to answer.
No loop-back.

## 7. Config Additions (`src/app/config.py`)

- `frontend_origins: list[str]` — already added in Stage 1
- `sufficiency_max_iters: int = 1` — already added in Stage 1
- `enable_stream: bool = True` — already added in Stage 1
