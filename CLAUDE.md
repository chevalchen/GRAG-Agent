# C9 项目当前状态说明（LangGraph GraphRAG）

本文档描述当前代码的真实形态，用于后续开发与排障时快速对齐。

## 1. 项目目标

项目包含两条主线：

- 在线问答（`src/app/online_qa`）
  - 查询分析与路由（hybrid / graph_rag / combined）
  - 检索、融合、生成回答
  - 支持会话 ID 维度的多轮对话
- 离线构建（`src/app/offline_ingestion`）
  - 扫描菜谱文件、并发解析、批次落盘、导出 CSV

## 2. 当前代码分层

### 2.1 App 层（业务编排）

- 在线编排：
  - `src/app/online_qa/graphs/online_qa_graph.py`
  - `src/app/online_qa/nodes/*.py`（6 个节点）
  - `src/app/online_qa/tools/*.py`
  - `src/app/online_qa/state.py`
  - `src/app/online_qa/checkpointer.py`
  - `src/app/online_qa/cli.py`
- 离线编排：
  - `src/app/offline_ingestion/graphs/ingestion_graph.py`
  - `src/app/offline_ingestion/tools/*.py`
  - `src/app/offline_ingestion/state.py`
  - `src/app/offline_ingestion/cli.py`

### 2.2 Core 层（通用契约与基础工具）

- Schema：
  - `src/core/schemas/document.py`
- DB client：
  - `src/core/tools/db/neo4j_client.py`
  - `src/core/tools/db/milvus_client.py`
- LLM client/tool：
  - `src/core/tools/llm/llm_client.py`
  - `src/core/tools/llm/embedding_client.py`
  - `src/core/tools/llm/generation_tool.py`

### 2.3 Legacy 层（被封装依赖）

- `src/legacy/rag_modules/*`
- `src/legacy/agent/*`

原则：`src/app/**` 负责编排，`src/legacy/**` 负责兼容实现，不直接改写 legacy 核心逻辑。

## 3. 在线问答链路（现状）

### 3.1 图结构

`START -> query_analysis -> route -> (hybrid_retrieve / graph_retrieve) -> fuse -> answer -> END`

### 3.2 节点职责

- `query_analysis_node`：写 `analysis`
- `route_node`：写 `route`
- `hybrid_retrieve_node`：写 `hybrid_docs`
- `graph_retrieve_node`：写 `graph_docs`
- `fuse_node`：写 `fused_docs` 与 `docs_final`
- `answer_node`：写 `answer`，并追加本轮 `history`

### 3.3 多轮对话机制

当前是双层机制：

1) 图内会话（同进程）  
- `build_graph()` 绑定 `get_checkpointer()`（MemorySaver）
- `thread_id` 通过 `config={"configurable":{"thread_id": session_id}}` 传入

2) 跨进程会话（CLI 多次命令）  
- `src/app/online_qa/cli.py` 将历史持久化到 `.session_history/<session-id>.json`
- 每次调用前加载历史并注入 `online_graph.ainvoke(..., history=...)`
- 回答后回写 history，保证两次 `python -m ... --session-id xxx` 可连续对话

说明：MemorySaver 仅进程内有效；跨命令连续对话依赖 `.session_history` 文件。

### 3.4 生成侧策略

- `AnswerGenerationTool` 已支持：
  - 基于 429/过载关键词自动重试
  - 将近几轮 history 拼入有效问题，增强追问理解

## 4. 离线 ingestion（现状）

- 图在 `src/app/offline_ingestion/graphs/ingestion_graph.py`
- 解析/构建/导出/进度分别封装为：
  - `parse_tool.py`
  - `build_tool.py`
  - `export_tool.py`
  - `progress_tool.py`
- `ingestion_graph.py` 不直接 import `src.legacy`，而经由 tools 调用

## 5. 运行方式

### 5.1 安装

```bash
pip install -r requirements.txt
```

### 5.2 在线问答

```bash
python -m src.app.online_qa
python -m src.app.online_qa --query "红烧肉怎么做？"
python -m src.app.online_qa --session-id test-001 --query "红烧肉怎么做？"
python -m src.app.online_qa --session-id test-001 --query "可以用电饭锅吗？"
python -m src.app.online_qa --stream --session-id test-001 --query "继续"
```

### 5.3 离线构建

```bash
python -m src.app.offline_ingestion <recipe_dir> -o ./ai_output --resume
```

### 5.4 测试

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

## 6. 配置与环境变量

- 配置入口：`src/app/config.py`
- 环境变量加载：`src/utils/env_utils.py`

关键变量：

- LLM：`MOONSHOT_API_KEY`、`MOONSHOT_BASE_URL`
- Neo4j：`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`NEO4J_DATABASE`
- Milvus：`MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION_NAME`

## 7. 当前目录要点

- `src/app/online_qa/nodes/` 已替代旧 `agents/` 目录
- `src/core/` 已建立并用于共享 schema 与基础工具
- `.session_history/` 用于 CLI 跨进程会话历史持久化

## 8. 维护注意事项

- 不修改 `src/legacy/**` 的业务语义，优先在 app/core 层做封装与编排
- 多轮问题优先检查：
  - `--session-id` 是否一致
  - `.session_history/<session-id>.json` 是否混入旧上下文
  - `answer_generation.py` 的 history 拼接是否生效
