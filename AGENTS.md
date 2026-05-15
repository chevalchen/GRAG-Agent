# AGENTS.md

This file provides project navigation for AI-assisted tools, reflecting the current state of the codebase.

---

## 1. Project Overview

The **TCM Intelligent Consultation System** is a GraphRAG-based Traditional Chinese Medicine knowledge Q&A platform.

### Core Capabilities

| Module | Function |
|--------|----------|
| Offline Ingestion | Parse drug NER entities and TCM literature, build Neo4j knowledge graph + Milvus vector index |
| Online QA | Multi-way retrieval (graph/drug/literature/health) and intelligent answer generation via LangGraph |
| MCP Server | Standardized interface for external clients (e.g., Trae) |

---

## 2. Technology Stack

| Category | Technology |
|----------|------------|
| Workflow Engine | LangGraph 1.x |
| Knowledge Graph | Neo4j |
| Vector Database | Milvus |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranking | CrossEncoder (BAAI/bge-reranker-base) |
| LLM | Moonshot/Kimi API |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Session Persistence | SQLite |
| MCP Framework | FastMCP |

---

## 3. Project Structure

```
src/
├── app/
│   ├── config.py                    # Configuration management
│   ├── mcp_server.py                # MCP server entry
│   ├── online_qa/                   # Online QA module
│   │   ├── cli.py                   # CLI entry
│   │   ├── checkpointer.py          # Session persistence
│   │   ├── state.py                 # State definitions
│   │   ├── graphs/
│   │   │   └── online_qa_graph.py   # QA workflow graph
│   │   └── nodes/                   # Node implementations
│   │       ├── supervisor_node.py           # Intent routing
│   │       ├── drug_entity_resolve_node.py  # Drug entity resolution
│   │       ├── graph_retrieve_node.py       # Graph retrieval
│   │       ├── drug_retrieve_node.py        # Drug vector retrieval
│   │       ├── lit_retrieve_node.py         # Literature retrieval
│   │       ├── health_retrieve_node.py      # Health knowledge retrieval
│   │       ├── fuse_node.py                 # RRF fusion
│   │       ├── rerank_node.py               # Reranking
│   │       └── answer_node.py               # Answer generation
│   └── offline_ingestion/           # Offline ingestion module
│       ├── cli.py                   # CLI entry
│       ├── state.py                 # State definitions
│       ├── graphs/
│       │   └── ingestion_graph.py   # Ingestion workflow graph
│       ├── nodes/                   # Node implementations
│       │   ├── parse_node.py        # NER parsing
│       │   ├── chunk_node.py        # Document chunking
│       │   ├── normalize_node.py    # Entity normalization
│       │   ├── neo4j_write_node.py  # Write to Neo4j
│       │   └── milvus_index_node.py # Index to Milvus
│       └── tools/                   # Utility functions
├── core/
│   ├── schemas/document.py          # Document models
│   └── tools/
│       ├── db/                      # Database clients
│       │   ├── neo4j_client.py
│       │   └── milvus_client.py
│       ├── graph/neo4j_tool.py      # Graph operations
│       ├── retrieval/bm25_tool.py   # BM25 retrieval
│       ├── vector/milvus_tool.py    # Vector retrieval
│       └── llm/                     # LLM utilities
│           ├── llm_client.py
│           ├── embedding_client.py
│           └── generation_tool.py
│   └── utils/
│       ├── drug_name_normalizer.py  # Drug name normalization
│       └── recipe_loader.py
└── legacy/                          # Legacy code (reference only)
    ├── agent/
    └── rag_modules/
```

---

## 4. Online QA Architecture

### 4.1 Execution Flow

```
START
  │
  ▼
supervisor_node ─────────────────┐
  │                              │
  ▼                              │
drug_entity_resolve_node         │
  │                              │
  ├──► graph_retrieve_node ──┐   │
  ├──► drug_retrieve_node ───┤   │
  ├──► lit_retrieve_node ────┼───┤
  └──► health_retrieve_node ─┘   │
                                 │
  ┌──────────────────────────────┘
  ▼
fuse_node (RRF fusion)
  │
  ▼
rerank_node (CrossEncoder reranking)
  │
  ▼
answer_node (Answer generation)
  │
  ▼
END
```

### 4.2 Node Responsibilities

| Node | Responsibility |
|------|----------------|
| supervisor_node | Analyze query intent, output routing signal (complexity/enabled sources/keywords) |
| drug_entity_resolve_node | Identify drug names, resolve to standard drug entities (canonical_name, node_id) |
| graph_retrieve_node | Fetch drug subgraph by node_id; fallback to keyword paths on failure |
| drug_retrieve_node | BM25 + Milvus dual retrieval; enhance query with canonical name, filter by node_id |
| lit_retrieve_node | Milvus literature retrieval (doc_type == "tcm_literature") |
| health_retrieve_node | Health knowledge retrieval (currently stub) |
| fuse_node | Four-way RRF fusion, ranked by reciprocal rank sum |
| rerank_node | CrossEncoder reranking; skippable for simple queries |
| answer_node | Select context strategy by complexity, generate answer |

### 4.3 State Definition

```python
OnlineQAState:
- query: str                    # User query
- routing: RoutingSignal        # Routing signal
- resolved_drug: ResolvedDrug   # Resolved drug entity
- graph_docs: list[Document]    # Graph retrieval results
- drug_docs: list[Document]     # Drug retrieval results
- lit_docs: list[Document]      # Literature retrieval results
- health_docs: list[Document]   # Health retrieval results
- docs_rrf: list[Document]      # RRF fusion results
- docs_final: list[Document]    # Final results after reranking
- metrics: dict                 # Execution metrics
- answer: str                   # Final answer
- history: list[dict]           # Conversation history
```

---

## 5. Offline Ingestion Architecture

### 5.1 Execution Flow

```
                    ┌─► parse ──► normalize ──► neo4j_write ──┐
                    │                                         │
scan ───────────────┤                                         ├──► milvus_index ──► END
  │                 │                                         │
  │                 └─► chunk ────────────────────────────────┘
  │
  └── Scan data files (data/drug/medical_ner_entities.json + data/lit/Traditional_Chinese_Medical_Literature_QA.json)
```

### 5.2 Node Responsibilities

| Node | Responsibility |
|------|----------------|
| scan | Scan data directory, locate input files |
| parse | Parse NER entities (drugs, ingredients, effects, symptoms, etc.) |
| normalize | Normalize conceptType, add canonical_name / aliases / node_id |
| chunk | Chunk literature QA data |
| neo4j_write | Write nodes and relationships to Neo4j |
| milvus_index | Build unified vector index |

### 5.3 Data Model

**Neo4j Node Types**: Drug, Ingredient, Effect, Symptom, Disease, Syndrome, Population, AdverseReaction

**Neo4j Relationship Types**: CONTAINS, HAS_EFFECT, INDICATES, CAUTION_FOR, HAS_ADVERSE_REACTION

**Milvus Document Types**: drug_profile, tcm_literature, health_popularization

---

## 6. MCP Server Interface

File: `src/app/mcp_server.py`

### 6.1 Tool List

| Tool | Parameters | Return Value |
|------|------------|--------------|
| tcm_query | question: str, session_id: str (optional) | str (plain text answer) |
| session_list | none | JSON (session list) |
| session_delete | session_id: str | JSON (operation result) |

### 6.2 Startup

```bash
uv run python -m src.app.mcp_server
```

### 6.3 Trae Configuration Example

```json
{
  "mcpServers": {
    "tcm-assistant": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.app.mcp_server"],
      "cwd": "D:\\CODE\\MedicalRAG\\GRAG-Agent"
    }
  }
}
```

---

## 7. Configuration

File: `src/app/config.py`

### 7.1 Environment Variables

```bash
# Database Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=tcm_knowledge

# Model Configuration
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
LLM_MODEL=kimi-k2-0711-preview
RERANKER_MODEL=BAAI/bge-reranker-base

# API Keys
MOONSHOT_API_KEY=...
KIMI_API_KEY=...

# Performance Tuning
FAST_MODE=false                    # Enable lightweight models
BM25_TOP_K=10
RETRIEVAL_TIMEOUT_SECONDS=1.2
TOP_K=3

# Session Persistence
CHECKPOINTER_PATH=.checkpoints/tcm.db
```

### 7.2 Fast Mode

Set `FAST_MODE=true` to switch to lightweight models for lower latency:
- FAST_EMBEDDING_MODEL
- FAST_LLM_MODEL
- FAST_RERANKER_MODEL

---

## 8. Usage

### 8.1 Online QA (CLI)

```bash
# Interactive mode
uv run python -m src.app.online_qa.cli

# Single query mode
uv run python -m src.app.online_qa.cli --query "What are the indications of Wuji Baifeng Wan?" --stream
```

### 8.2 Offline Ingestion

```bash
uv run python -m src.app.offline_ingestion.cli <data_dir> -o ./ai_output --resume
```

### 8.3 MCP Server

```bash
uv run python -m src.app.mcp_server
```

### 8.4 Testing

```bash
python -m pytest tests -v
```

---

## 9. Performance Benchmark

**Test Command**:

```bash
python -m scripts.benchmark_online_qa \
  --queries-file scripts/benchmark_queries_task5.txt \
  --runs 1 \
  --p95-target 6.0 \
  --output benchmark/task5_benchmark_result.json
```

**Latest Results** (2026-04-16):

| Metric | Value |
|--------|-------|
| Samples | 6 |
| mean | 6.082s |
| p50 | 5.507s |
| p95 | 8.715s |
| Target | 6.000s (not met) |
| Bottleneck | supervisor_node (intent analysis) |

**Optimization Direction**: Optimize LLM intent analysis latency (model selection, prompt length, timeout strategy)

---

## 10. Design Principles

1. **Node-based Extension**: Online capabilities via new nodes, avoiding global singletons
2. **Dependency Injection**: Tools injected via factory functions, facilitating testing and replacement
3. **Unified Normalization**: Drug name normalization shared between offline and online
4. **Progressive Degradation**: Automatic keyword fallback when graph retrieval fails
5. **Complexity Awareness**: Skip reranking for simple queries, limit context budget

---

## 11. Constraints & Notes

- Cypher query changes must be compatible with current type recognition strategy
- BM25 single query limit: 16384 (avoid Milvus query window errors)
- Drug entities use unified node_id without relying on alias node modeling
- Run only necessary tests, avoid meaningless full test runs

---

## 12. First Principles

- No compatibility patch solutions
- No over-engineering, prioritize shortest correct path
- Solutions must pass full-linkage logic validation
