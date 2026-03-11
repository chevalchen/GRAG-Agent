

## 11. 落地规划：将 `src/legacy/**` 重构为 LangGraph Nodes + Tools

本节是“工程形态统一 + LangGraph 编排 + 多智能体 + 混合图检索”的落地任务书，目标是在保持策略效果与 DB 接口语义不变的前提下，把 legacy 实现逐步下沉到统一工具层，并在 Graph 中以节点方式组合。

### 11.1 总体目标与边界

- 目标：`src/app/**` 只负责编排、权限控制与状态更新；外部系统调用全部通过 `src/core/tools/**`
- 允许的阶段性过渡：`src/app/**` 的 Tools 可以先“薄封装 legacy”，随后再逐步拆解 legacy 为更细粒度工具
- 禁止的行为：在重构过程中改写 Cypher 语义、改变 Milvus schema/search_params 的语义

### 11.2 legacy → Tool/Node 映射表（在线）

| legacy 文件 | 新 Tool 目标 | 新 Node/Agent 使用方式 | 迁移优先级 |
|---|---|---|---|
| `src/legacy/rag_modules/intelligent_query_router.py` | `QueryAnalysisTool.analyze(query)` | `Supervisor/RouterAgent` 调用并写入 `analysis/route` | P0 |
| `src/legacy/rag_modules/hybrid_retrieval.py` | `HybridRetrieveTool.search(query, top_k)` + 细分 `Neo4jTool/MilvusTool/BM25Tool` | `HybridRetrievalAgent` 调用 | P0 |
| `src/legacy/rag_modules/graph_rag_retrieval.py` | `GraphQueryPlanTool.plan(query)` + `Neo4jTraverseTool.traverse(plan)` + `GraphRAGRetrieveTool.search(...)` | `GraphRetrievalAgent` 调用 | P0 |
| `src/legacy/rag_modules/generation_integration.py` | `AnswerGenerationTool.generate(query, docs, stream=False)` | `AnswerAgent` 调用 | P0 |
| `src/legacy/rag_modules/graph_data_preparation.py` | `Neo4jDataLoaderTool` + `DocumentBuilderTool` + `ChunkerTool` | `KnowledgeBaseBuildGraph` 的节点 | P1 |
| `src/legacy/rag_modules/milvus_index_construction.py` | `MilvusIndexTool.build/load/search/has_collection` | `KnowledgeBaseBuildGraph` 与 `HybridRetrieveTool` | P1 |

### 11.3 legacy → Tool/Node 映射表（离线）

| legacy 文件 | 新 Tool 目标 | 新 Graph/Node 使用方式 | 迁移优先级 |
|---|---|---|---|
| `src/legacy/agent/recipe_ai_agent.py` | `RecipeParseTool`、`GraphRecordBuilderTool`、`ExporterTool`、`ProgressStoreTool`、`BatchStoreTool` | `OfflineIngestionGraph` 逐节点调用 | P0 |
| `src/legacy/agent/amount_normalizer.py` | `NormalizeTool.normalize(recipe_info)` | ingestion normalize node | P1 |

### 11.4 Tool API 规范（建议签名）

在线问答最小 Tool API：
- `QueryAnalysisTool.analyze(query: str) -> QueryAnalysis`
- `HybridRetrieveTool.search(query: str, top_k: int) -> list[Document]`
- `GraphRAGRetrieveTool.search(query: str, top_k: int) -> list[Document]`
- `FusionTool.fuse(graph_docs: list[Document], hybrid_docs: list[Document], top_k: int) -> list[Document]`
- `AnswerGenerationTool.generate(query: str, docs: list[Document], stream: bool = False) -> str | None`

基础设施 Tool（共享）：
- `Neo4jTool.session()`：统一 driver/session 生命周期
- `MilvusTool.client()`：统一 client 生命周期
- `RetryPolicy` / `RateLimitPolicy`：统一在工具层应用
- `MetricsTool.emit(name, value, tags)`：统一 metrics

### 11.5 统一在线 LangGraph（多智能体 + 混合图检索）

推荐 DAG（受控多智能体，不做自由工具调用）：
1) `SupervisorNode`：调用 `QueryAnalysisTool` → 写入 `analysis/route`
2) `HybridRetrieveNode`：调用 `HybridRetrieveTool`
3) `GraphRetrieveNode`：调用 `GraphRAGRetrieveTool`
4) `FuseNode`（仅 combined 路由进入）：调用 `FusionTool`
5) `AnswerNode`：调用 `AnswerGenerationTool`
6) （可选）`PostCheckNode`：空答/引用校验（不改变策略，仅记录 metrics）

并发策略：
- combined 时 `HybridRetrieve` 与 `GraphRetrieve` 并发执行（`asyncio.gather` + semaphore）
- LLM 并发、Neo4j 并发、Milvus 并发分别限制

### 11.6 知识库构建 LangGraph（建议新增）

目的：把“KB 构建/加载”的脚本逻辑从 CLI 中拆出，成为可复用 Graph。

推荐节点：
1) `CheckMilvusNode`：has_collection/load_collection
2) `LoadNeo4jGraphNode`：load_graph_data
3) `BuildDocumentsNode`：build_recipe_documents
4) `ChunkNode`：chunk_documents
5) `BuildMilvusIndexNode`：build_vector_index（必要时）

### 11.7 交付节奏（里程碑 M0-M4）

- M0（0.5-1 天）：冻结 Tool API + State 字段 + CLI 运行命令
- M1（1-2 天）：新增 KB Build Graph，并切换在线 CLI 走 Graph
- M2（2-4 天）：Neo4j/Milvus/LLM 全部下沉到 `src/core/tools/**`，online graph 只调用 tools
- M3（1-2 天）：离线 ingestion 全工具化（parse/normalize/build_records/flush/export）
- M4（1-2 天）：回归与验收（策略分布、docs 结构、E2E 冒烟、可观测）

### 11.8 验收标准（针对本规划）

- 在线：能输出 route/检索摘要/可选 metrics；combined 仍是 graph 优先 round-robin；无 driver/session NoneType 报错
- 离线：resume、batch flush、export 行为与 legacy 一致
- 工程：`src/app/**` 不再直接创建 Neo4j driver 或 Milvus client（统一工具层）
- 测试：工具层可 mock；至少覆盖路由/融合/工具权限；保留 1 条端到端冒烟脚本
