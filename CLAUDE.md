# CLAUDE.md

本文件用于让 AI 辅助工具快速对齐当前代码结构、运行方式与约束。内容以当前仓库实现为准。

---

## 1. 项目定位

这是一个菜谱知识图谱问答系统，分为两条主线：

- **离线构建**：从菜谱文件解析结构化数据，写入 Neo4j，再为在线检索构建 Milvus 向量索引。
- **在线问答**：基于 LangGraph 的多节点图执行，按问题动态路由到 hybrid / graph_rag / combined 检索策略并生成答案。

---

## 2. 当前目录结构（关键）

```text
src/
├── app/
│   ├── config.py
│   ├── online_qa/
│   │   ├── cli.py
│   │   ├── checkpointer.py
│   │   ├── state.py
│   │   ├── graphs/
│   │   │   └── online_qa_graph.py
│   │   └── nodes/
│   │       ├── supervisor_node.py
│   │       ├── hybrid_retrieve_node.py
│   │       ├── graph_retrieve_node.py
│   │       ├── fuse_node.py
│   │       └── answer_node.py
│   └── offline_ingestion/
│       ├── cli.py
│       ├── state.py
│       ├── graphs/
│       │   └── ingestion_graph.py
│       ├── nodes/
│       │   ├── neo4j_write_node.py
│       │   └── milvus_index_node.py
│       └── tools/
│           ├── parse_tool.py
│           ├── build_tool.py
│           ├── export_tool.py
│           ├── progress_tool.py
│           └── scan_files.py
├── core/
│   ├── schemas/document.py
│   ├── tools/
│   │   ├── db/{neo4j_client.py,milvus_client.py}
│   │   ├── graph/neo4j_tool.py
│   │   ├── retrieval/bm25_tool.py
│   │   ├── vector/milvus_tool.py
│   │   └── llm/{llm_client.py,embedding_client.py,generation_tool.py}
│   └── utils/recipe_loader.py
└── legacy/
    ├── agent/
    └── rag_modules/   # 仅保留空包入口，不再承载实现
```

注意：`src/app/online_qa/tools/` 在当前实现中已移除，在线问答通过节点闭包 + 依赖注入组装工具。

---

## 3. 在线问答架构（现状）

### 3.1 图结构

`START -> supervisor -> (hybrid_retrieve / graph_retrieve / 两路并行) -> fuse -> answer -> END`

实现文件：`src/app/online_qa/graphs/online_qa_graph.py`

### 3.2 节点职责

- `supervisor_node`  
  使用 LLM 输出严格 JSON，填充 `QueryAnalysis`，核心字段：`keywords / query_complexity / relationship_intensity / recommended_strategy`。
- `hybrid_retrieve_node`  
  并发执行 BM25 + Milvus 检索；再对 Milvus 命中的 node_id 做 Neo4j 一跳扩展。
- `graph_retrieve_node`  
  基于分析结果选择深度（1 或 2），调用 Neo4j 图检索。
- `fuse_node`  
  `combined` 路径使用 RRF (`_rrf`) 融合；单路路径直接透传结果。
- `answer_node`  
  拼接最近 `history_window` 轮历史与检索上下文，调用 LLM 生成回答，并追加本轮 user/assistant 历史。

### 3.3 会话与持久化

- checkpointer：`src/app/online_qa/checkpointer.py`
- 默认 SQLite：`.checkpoints/c9.db`
- `thread_id` 作为会话隔离键（由 CLI 的 `session_id` 传入）
- CLI 支持历史会话列表与删除（直接操作 checkpoints/writes 表）

### 3.4 在线入口

- 入口：`python -m src.app.online_qa.cli`
- 单问：`python -m src.app.online_qa.cli --query "红烧肉怎么做？" --stream`
- 可选参数：`--top-k --session-id --show-metrics --log-level`

---

## 4. 离线构建架构（现状）

实现文件：`src/app/offline_ingestion/graphs/ingestion_graph.py`

主流程：

`init -> scan -> parse_batch -> flush(按条件) -> finalize -> neo4j_write -> milvus_index -> export`

### 关键点

- `init` 阶段会构造 `GraphRAGConfig`，并注入 Neo4j/Milvus 连接配置。
- `parse_batch` 使用线程池并发解析文件，失败重试（最多 3 次）。
- `neo4j_write_node` 负责将离线中间结果写入图数据库。
- `milvus_index_node` 通过 `src/core/utils/recipe_loader.py` 从 Neo4j 回读 Recipe 文档并写入向量库。
- `export` 默认导出 CSV 结果。

离线入口：

`python -m src.app.offline_ingestion.cli <recipe_dir> -o ./ai_output --resume`

---

## 5. 核心配置

配置文件：`src/app/config.py`

- `GraphRAGConfig`：图数据库、向量库、模型、检索参数与分块参数。
- `DEFAULT_CONFIG`：默认实例。
- `Config`：从环境变量映射运行时配置。

常用环境变量：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=cooking_knowledge
MOONSHOT_API_KEY=...
KIMI_API_KEY=...
C9_SESSION_ID=optional_session_id
```

---

## 6. 数据模型约定

当前查询逻辑按“节点类型字段优先、标签兜底”处理：

- 节点类型识别：`coalesce(n.conceptType, head(labels(n)), "")`
- 主要类型值：`Recipe / Ingredient / CookingStep`
- 主键：`nodeId`

这使系统兼容两种写法：

- 仅使用标签表示类型
- 使用通用标签 + `conceptType` 属性表示类型

---

## 7. 测试与质量检查

当前测试目录：

- `tests/test_online_qa_graph.py`
- `tests/test_core_tools.py`
- `tests/test_fusion.py`
- `tests/test_scan_files.py`

运行：

```bash
python -m pytest tests -v
```

---

## 8. 开发约束（给 AI/开发者）

- 优先修改 `src/app` 与 `src/core`，不要在 `src/legacy` 增加新实现。
- 在线问答新增能力，优先通过“节点 + 依赖注入”扩展，不引入全局 runtime 单例。
- 任何 Cypher 变更需兼容当前类型识别策略，避免写死已不存在标签或属性。
- 变更后至少执行一次 `python -m pytest tests -v`。
