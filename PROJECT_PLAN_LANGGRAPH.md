# 项目重构规划：LangGraph 多智能体（Hybrid/Graph 检索）

本规划基于当前仓库 [README.md](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/README.md)、[CLAUDE.md](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/CLAUDE.md) 与现有源代码（`rag_modules/`、`agent/`）制定，目标是将系统重构为**标准 LangGraph 项目形态**的**异步、多智能体、混合/图检索**架构，同时在“策略效果不变”和“谨慎修改数据库接口”的约束下推进。

---

## 0. 目标与约束

### 目标
- 将在线问答链路重构为 **LangGraph 工作流 + 多智能体协作**：路由、检索、融合、生成、可选评估。
- 支持**异步执行**（LLM 与检索并发、可控并发度）。
- 将 `agent/` 离线图谱构建纳入统一 LangGraph 管理：解析→批处理→导出（Neo4j CSV 等）。
- 按“标准 LangGraph 项目”组织：清晰的 graph / agents / tools / state / config 分层。

### 硬约束
- **路由策略效果不变**：保持 `IntelligentQueryRouter.analyze_query` 的提示词/字段输出与降级规则一致，路由结果、统计口径一致。
- **谨慎修改数据库接口**：Neo4j / Milvus 连接、Cypher 查询、Milvus schema 与 search 参数不做语义性改写；优先“包一层适配器”。
- `.env` 继续只存放数据库与 API key 等敏感配置，统一由 `utils/env_utils.py` 加载与导出常量。

---

## 1. 现状基线（为了保持“策略不变”）

### 在线问答（当前）
- 入口：`main.py` 初始化各模块并交互式问答
- 路由：`rag_modules/intelligent_query_router.py`
  - LLM 路由：输出 `QueryAnalysis`（complexity、relationship_intensity、reasoning_required、entity_count、recommended_strategy、confidence、reasoning）
  - 降级：关键词规则（complexity_keywords / relation_keywords）→ strategy
  - Combined：Round-robin 合并两侧结果，GraphRAG 结果优先
- 检索：
  - Hybrid：`rag_modules/hybrid_retrieval.py`（BM25 + Milvus similarity + 图邻居增强 + Round-robin 合并）
  - GraphRAG：`rag_modules/graph_rag_retrieval.py`（意图理解→多跳遍历/子图提取→文档化）
- 生成：`rag_modules/generation_integration.py`（OpenAI SDK chat.completions）

### 离线图谱构建（当前）
- 入口：`agent/run_ai_agent.py`、`agent/batch_manager.py`
- 核心：`agent/recipe_ai_agent.py`（Kimi 解析 + 批处理 + 导出 CSV/Neo4j 导入）

---

## 2. 目标架构（LangGraph 多智能体）

整体建议拆为 3 张图（3 个 Graph / workflow），分别独立可运行、可复用组件一致：

1) **OnlineQA Graph（在线问答）**
- 多智能体：RouterAgent、HybridRetrievalAgent、GraphRetrievalAgent、FusionAgent、AnswerAgent、(可选) Critic/EvaluatorAgent

2) **OfflineIngestion Graph（离线图谱构建）**
- 多智能体：IngestionSupervisor、RecipeParseAgent、NormalizeAgent、BatchWriterAgent、ExporterAgent、(可选) ValidatorAgent

3) **Evaluation Graph（回归评估）**
- 用于“策略不变”回归：对一组固定 queries 对比旧系统与新系统的 strategy 分布、召回结构、答案一致性指标

---

## 3. 标准 LangGraph 项目组织（建议）

建议把核心重构到 `src/`，保留现有代码作为“legacy adapters”，逐步迁移：

```
C9/
├─ src/
│  ├─ app/
│  │  ├─ online_qa/                 # 在线问答子系统
│  │  │  ├─ graphs/                 # LangGraph graphs（入口在此）
│  │  │  ├─ agents/                 # Agent/Node 实现
│  │  │  ├─ tools/                  # 工具封装（Milvus/Neo4j/LLM/IO）
│  │  │  └─ state.py                # Graph state 定义
│  │  ├─ offline_ingestion/         # 离线构建子系统
│  │  │  ├─ graphs/
│  │  │  ├─ agents/
│  │  │  ├─ tools/
│  │  │  └─ state.py
│  │  └─ evaluation/
│  │     ├─ graphs/
│  │     └─ datasets/
│  └─ legacy/
│     ├─ rag_modules/               # 现有 rag_modules 适配器（尽量不改内部）
│     └─ agent/                     # 现有 agent 适配器
├─ rag_modules/                     # 现存（阶段 1 继续保留）
├─ agent/                           # 现存（阶段 1 继续保留）
├─ utils/env_utils.py
└─ main.py                          # 先作为 legacy 入口，后期替换为 src/app/... 的 CLI
```

阶段性策略：
- 阶段 1：不动 DB 模块内部实现，仅新增 `src/` + 适配器。
- 阶段 2：逐步把“提示词、路由、融合”迁移到新目录，但 Cypher/Milvus schema 保持不变。

---

## 4. 节点（Node）与工具（Tool）的划分原则

### 适合做“节点”的模块
节点负责**决策/编排/状态流转**，典型特征：
- 需要读写 Graph State（如分析结果、路由选择、候选文档、得分、trace 元信息）
- 需要选择下一步分支（routing / conditional edges）
- 需要并发组织多个工具调用并做融合

### 适合做“工具”的模块
工具负责**单一职责、可复用、可替换**的“动作”，典型特征：
- 只接受参数、返回结果，不直接改写全局状态
- 对外暴露稳定接口（例如 `search(query, k)->docs`）
- DB/外部系统调用（Neo4j/Milvus/文件系统）优先保持为工具

---

## 5. OnlineQA Graph 规划（异步、多智能体）

### 5.1 State（建议字段）

`OnlineQAState`（示例，不强制一次性全上）：
- `query: str`
- `analysis: QueryAnalysis | None`
- `route: Literal["hybrid_traditional","graph_rag","combined"] | None`
- `docs_hybrid: list[Document]`
- `docs_graph: list[Document]`
- `docs_final: list[Document]`
- `answer: str | None`
- `metrics: dict`（耗时、token、route_stats、doc_counts）
- `errors: list[str]`

### 5.2 节点与可调用工具约束

#### Node: AnalyzeQuery（RouterAgent）
- 职责：复刻 `IntelligentQueryRouter.analyze_query`（LLM JSON 输出 + 降级规则）
- 工具白名单：
  - `LLMTool.query_analyzer`（仅允许调用“分析 prompt”，不允许调用检索）
- 输出：`analysis`、`route`（recommended_strategy）

#### Node: Route（LangGraph 路由函数 / conditional edges）
- 职责：根据 `analysis.recommended_strategy` 选择分支
- 工具白名单：无（纯控制流）
- 分支：
  - `hybrid_traditional` → HybridRetrieve
  - `graph_rag` → GraphRetrieve
  - `combined` → ParallelRetrieve → Fuse

#### Node: HybridRetrieve（HybridRetrievalAgent）
- 职责：调用 legacy `HybridRetrievalModule.hybrid_search`
- 工具白名单：
  - `HybridSearchTool`（内部可组合 BM25/Milvus/邻居增强，但不暴露额外 DB 写操作）
- 输出：`docs_hybrid`

#### Node: GraphRetrieve（GraphRetrievalAgent）
- 职责：调用 legacy `GraphRAGRetrieval.graph_rag_search`
- 工具白名单：
  - `GraphRAGSearchTool`（内部可调用 Neo4j 只读查询）
- 输出：`docs_graph`

#### Node: ParallelRetrieve（用于 combined）
- 职责：异步并发跑 HybridRetrieve 与 GraphRetrieve（受并发限制）
- 工具白名单：同上，但不允许写库
- 输出：`docs_hybrid`、`docs_graph`

#### Node: Fuse（FusionAgent）
- 职责：复刻当前 combined 的 Round-robin 融合策略（GraphRAG 优先、去重规则一致）
- 工具白名单：无（纯内存融合）
- 输出：`docs_final`

#### Node: GenerateAnswer（AnswerAgent）
- 职责：调用 legacy `GenerationIntegrationModule.generate_*`（流式可选）
- 工具白名单：
  - `LLMTool.answer_generator`（仅允许生成，不允许路由分析）
- 输出：`answer`

#### Node: EvaluateAnswer（可选 Critic/EvaluatorAgent）
- 职责：质量评估、是否需要补检索、是否追加约束
- 工具白名单：
  - `LLMTool.answer_critic`（只读评估）
- 输出：`metrics`（不修改 docs，避免改变原策略效果；仅用于观测）

### 5.3 异步策略（必须实现）

必须实现的异步点：
- `combined` 分支：Hybrid 与 GraphRAG **并发**检索
- 生成阶段：流式生成保持现状；内部可以 async 但需兼容 CLI 打印

并发控制建议：
- 用 `asyncio.Semaphore` 控制：
  - LLM 并发（例如 2-4）
  - Neo4j 会话并发（例如 4-8，视实例）
  - Milvus 查询并发（例如 4-8）
- 对 legacy 同步函数：用 `asyncio.to_thread()` 包装，避免阻塞事件循环

### 5.4 路由策略“效果不变”的落地方式

保持不变的关键点（必须写入验收标准）：
- `analysis_prompt` 文本与字段含义保持一致
- JSON 字段：`query_complexity`、`relationship_intensity`、`reasoning_required`、`entity_count`、`recommended_strategy`、`confidence`、`reasoning`
- 降级规则与阈值保持一致（当前：`>0.3` 判定 GraphRAG）
- Combined 融合的：
  - 结果配额分配（`traditional_k = top_k//2`，`graph_k = top_k-traditional_k`）
  - Round-robin 顺序（先 graph 再 traditional）
  - 去重逻辑（`hash(doc.page_content[:100])`）

在 LangGraph 内实现方式：
- AnalyzeQuery 节点产出 `analysis`
- 用 `add_conditional_edges` 路由函数读取 `analysis.recommended_strategy.value`
- Fuse 节点复刻 `_combined_search`

---

## 6. OfflineIngestion Graph 规划（把 agent 纳入 LangGraph）

离线链路的目标是“标准化成可观测、可并发、可断点恢复”的 ingestion graph，同时尽量不改现有 CSV/Neo4j 导出格式。

### 6.1 State（建议字段）

`IngestionState`：
- `recipe_root: str`
- `file_queue: list[str]`
- `processed_files: set[str]`
- `failed_files: list[{path, error}]`
- `batch_id: int`
- `batch_size: int`
- `concepts_buffer: list[dict]`
- `relationships_buffer: list[dict]`
- `output_dir: str`
- `resume: bool`
- `metrics: dict`

### 6.2 节点与工具约束

#### Node: ScanFiles（IngestionSupervisor）
- 工具白名单：
  - `FileSystemTool.scan_markdown_files`（只读）
- 输出：`file_queue`

#### Node: LoadProgress（可选）
- 工具白名单：
  - `ProgressStoreTool.load`（读 progress.json）
- 输出：`processed_files`、`batch_id`

#### Node: ParseRecipe（RecipeParseAgent，异步并发）
- 工具白名单：
  - `RecipeParserTool`（内部调用 legacy `KimiRecipeAgent.extract_recipe_info`）
- 输出：结构化 `RecipeInfo`（中间态）
- 约束：不允许写文件/写库，只能解析

#### Node: Normalize（NormalizeAgent）
- 工具白名单：
  - `AmountNormalizerTool`（legacy `amount_normalizer.py`）
- 输出：规范化后的 `RecipeInfo`

#### Node: BuildGraphRecords（BatchWriterAgent）
- 工具白名单：
  - `GraphRecordBuilderTool`（把 RecipeInfo 转为 nodes/relationships 记录）
- 输出：追加到 buffer

#### Node: FlushBatch（BatchWriterAgent）
- 工具白名单：
  - `BatchStoreTool.write_batch_csv`（仅写 output_dir/batch_xxx）
  - `ProgressStoreTool.save`（更新 progress）
- 输出：清空 buffer、batch_id++

#### Node: Export（ExporterAgent）
- 工具白名单：
  - `ExporterTool.export_to_neo4j_csv`（复用 legacy 逻辑）
- 输出：nodes.csv / relationships.csv / neo4j_import.cypher

### 6.3 异步策略（必须实现）

解析阶段建议并发：
- `ParseRecipe` 用 `asyncio.gather` 并发执行，但受限于：
  - API QPS / rate limit
  - 单批并发数（例如 3-5）
- 失败重试：在 Tool 层封装（指数退避、最大重试次数沿用 legacy）

断点续传：
- 继续沿用现有 `progress.json` 与 batch 目录结构，LangGraph 仅作为编排器
- 引入 LangGraph checkpoint 仅作为增强（后期可选），不替换现有机制以降低风险

---

## 7. 数据库接口：谨慎修改策略

原则：不重写查询语句、不改变 schema、不引入写入侧改动。

### Neo4j
- 保持 `GraphDataPreparationModule`、`GraphRAGRetrieval` 的 Cypher 查询与连接方式
- 新增 Tool/Adapter 只做：
  - “调用前后计时与错误处理”
  - “session 生命周期管理（复用 driver）”

### Milvus
- 保持 `MilvusIndexConstructionModule` 的 collection schema / index / search_params
- Tool 层只封装：
  - `has_collection/load_collection/build_vector_index/similarity_search`
  - 指标采集与异常转换

---

## 8. 回归与验收（保证策略效果不变）

### 8.1 路由一致性测试（必须）
- 构造 queries 数据集（至少 50-100 条，覆盖：简单做法、搭配关系、多跳推理、因果/比较）
- 旧路由与新路由输出对比：
  - `recommended_strategy` 一致率
  - `confidence`、`complexity` 分布差异在可接受范围（先观测，再定阈值）

### 8.2 检索结构一致性测试（建议）
- 对比 top_k 的 doc 数量、来源（graph/hybrid/combined）、去重后数量
- 对 combined 的融合顺序进行断言（GraphRAG 优先）

### 8.3 端到端冒烟（必须）
- `python main.py` 可启动（本地 Neo4j/Milvus 可用时）
- 提问 1-2 个 query 产出回答
- LangSmith tracing 在启用时可见（触发一次真实 LLM 调用）

---

## 9. 分阶段里程碑（推荐）

### 阶段 A：LangGraph 化但仍单体（低风险）
- 把 OnlineQA 链路整理为 LangGraph（Analyze→Retrieve→Fuse→Generate）
- DB 模块全部走 legacy 适配器
- 通过路由一致性回归

### 阶段 B：多智能体化（中风险）
- 引入 Supervisor + 专家 Agent（Hybrid/Graph/Fusion/Answer）
- 增加工具权限约束与状态校验

### 阶段 C：离线 ingestion Graph（中风险）
- 将 agent 批处理流程搬入 LangGraph
- 异步并发解析 + 断点续传保持兼容

### 阶段 D：评估与运维化（可选）
- 增加 Evaluation Graph、数据集管理、LangSmith dashboards
- 增加更细粒度的可观测指标（每节点耗时、失败率、命中率）

---

## 10. 风险清单与规避

- 路由策略漂移：LLM 输出易抖动  
  - 规避：保留原 prompt、温度、max_tokens；增加 JSON schema 校验；失败走规则降级

- 异步引入的资源竞争：Neo4j session / Milvus client 线程安全差异  
  - 规避：Tool 层做“连接复用但 session 短生命周期”；同步函数用 `to_thread`

- 离线 ingestion 并发导致 API 限速/失败  
  - 规避：并发上限 + 指数退避 + 批次 flush；失败进入 retry 队列

- 过度重构数据库接口导致不可控  
  - 规避：DB 模块保持原封不动；只加适配器与编排层

