## 项目概览（GraphRAG 智能菜谱）

本项目包含两条主线：

1) **在线问答（GraphRAG + Hybrid RAG）**
- 入口：[main.py](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/main.py)
- 数据源：Neo4j（菜谱图）+ Milvus（向量索引）
- 关键能力：查询智能路由（传统混合检索 / 图RAG / 组合）+ 生成式回答（Moonshot/Kimi）

2) **离线图谱构建（AI 菜谱解析 Agent）**
- 入口脚本：[agent/run_ai_agent.py](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/agent/run_ai_agent.py)、[agent/batch_manager.py](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/agent/batch_manager.py)、[agent/recipe_ai_agent.py](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/agent/recipe_ai_agent.py)
- 功能：将菜谱 Markdown 批量解析为结构化图谱数据（支持断点续传、批次合并、导出 Neo4j CSV）

---

## 目录结构

```
C9/
├─ main.py                       # 在线问答入口（交互式）
├─ config.py                     # GraphRAGConfig 默认配置
├─ requirements.txt              # Python 依赖
├─ .env                          # 仅存放数据库与 API key 等敏感配置（不要提交）
├─ .env.example                  # 环境变量示例
├─ utils/
│  └─ env_utils.py               # 统一加载 .env 并导出 env 常量
├─ rag_modules/
│  ├─ graph_data_preparation.py  # Neo4j→结构化 Document 构建
│  ├─ milvus_index_construction.py # Milvus 建库/插入/检索
│  ├─ hybrid_retrieval.py        # 混合检索（BM25 + Milvus + 图邻居增强）
│  ├─ graph_rag_retrieval.py     # 图RAG（意图理解→图查询→子图/路径提取）
│  ├─ intelligent_query_router.py # LLM/规则路由与策略选择
│  ├─ generation_integration.py  # 答案生成（Moonshot/Kimi，支持 LangSmith wrap）
│  └─ langgraph_workflow.py      # LangGraph 编排（analyze→retrieve→post_process）
└─ agent/
   ├─ recipe_ai_agent.py         # Kimi 菜谱解析与图谱导出
   ├─ run_ai_agent.py            # 便捷运行脚本（含 test）
   ├─ batch_manager.py           # 批次/断点续传管理
   ├─ amount_normalizer.py       # 用量标准化
   └─ config.json                # agent 侧配置（可作为兜底）
```

---

## 在线问答链路（main.py）

在线问答的核心链路：

1) **初始化系统**
- GraphDataPreparationModule：连接 Neo4j 并读取 Recipe/Ingredient/CookingStep
- MilvusIndexConstructionModule：Milvus 集合与向量索引
- GenerationIntegrationModule：OpenAI SDK（Moonshot base_url）与生成策略
- HybridRetrievalModule：BM25 + Milvus + 图索引（KV）与邻居增强
- GraphRAGRetrieval：图结构推理检索
- IntelligentQueryRouter：LLM/规则分析查询→选策略→返回 Document 列表
- GraphRAGWorkflow：用 LangGraph 把 “分析→检索→后处理” 编排为可扩展 DAG

2) **构建知识库**
- 从 Neo4j 载入图数据→构建菜谱文档→切块→写入 Milvus
- 若 Milvus 集合已存在则尝试直接加载，但仍会加载 Neo4j 图以支持图索引

3) **交互式问答**
- 对用户问题进行路由检索（LangGraph workflow）
- 基于检索到的 Document 生成回答（支持流式/非流式）

---

## 离线图谱构建（agent）

Agent 用于把菜谱 Markdown 解析成结构化数据并导出：
- 解析：OpenAI SDK 调用 Kimi（Moonshot base_url）提取 RecipeInfo/Ingredient/Step 等
- 处理：批量处理、批次保存、断点续传、失败重试
- 输出：nodes.csv / relationships.csv / neo4j_import.cypher（以及部分 CSV/RF2 能力）

---

## 环境变量（.env）

约定：`.env` 只存放数据库与 API Key 等敏感信息；不要把真实密钥提交仓库。

由 [utils/env_utils.py](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/utils/env_utils.py) 统一从项目根目录加载，并导出常量（在代码里以 `MOONSHOT_API_KEY` 这种形式使用）。

常用变量（示例见 `.env.example`）：

- Moonshot/Kimi：
  - `MOONSHOT_API_KEY`
  - `MOONSHOT_BASE_URL`（默认 https://api.moonshot.cn/v1）
  - `KIMI_API_KEY`（可选，部分脚本会优先读取）
- Neo4j：
  - `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE`
- Milvus：
  - `MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_COLLECTION_NAME` / `MILVUS_DIMENSION`
- LangSmith（支持两套命名，env_utils 会做兼容映射）：
  - `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT`
  - 或 `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` / `LANGSMITH_PROJECT`

---

## LangSmith / Tracing 说明

本项目不是纯 LCEL/Runnable 链路，LLM 调用主要走 OpenAI SDK；因此需要对 OpenAI client 做 instrumentation：
- [rag_modules/generation_integration.py](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/rag_modules/generation_integration.py) 使用 `langsmith.wrappers.wrap_openai` 包装 OpenAI client
- [agent/recipe_ai_agent.py](file:///d:/CODE/all_in_rag/all-in-rag/code/C9/agent/recipe_ai_agent.py) 同样对 OpenAI client 做 wrap（用于离线解析链路）

启用方式：
- `.env` 设置 `LANGCHAIN_TRACING_V2=true`（或 `LANGSMITH_TRACING=true`）
- 设置 `LANGCHAIN_API_KEY=...`（或 `LANGSMITH_API_KEY=...`）
- 触发一次真实的模型调用后，LangSmith 才会出现 trace

---

## 运行方式（开发者常用）

在线问答：
```bash
python main.py
```

离线解析（Agent）：
```bash
python agent/run_ai_agent.py test
python agent/run_ai_agent.py /path/to/HowToCook-master
python agent/batch_manager.py status -o ./ai_output
```

---

## Claude 协作规则（在本仓库内必须遵守）

### 工作方式
- 优先理解现有架构与调用链，沿用既有模块边界（rag_modules/agent/utils）。
- 保持功能不变优先；新增能力用“可选开关/可插拔模块”方式集成。
- 变更尽量小步且可回退；每次提交/变更必须能运行通过（至少语法与基本链路）。

### 代码风格与约束
- 不要引入无依据的新依赖；新增依赖必须写入 requirements.txt 并说明用途。
- 复用现有的 OpenAI SDK、Neo4j、Milvus 连接方式；不要硬编码密钥。
- 不要在代码里输出/打印密钥或敏感信息。
- 本仓库默认不要新增注释（除非明确要求）；优先通过清晰命名与模块拆分表达意图。

### 文件与文档
- 除非用户明确要求，不要新增 README/文档类文件。
- `.env` 不可提交真实密钥；只维护 `.env.example` 示例即可。

### 调试与验证
- 任何涉及代码修改的任务，至少运行一次 `python -m py_compile ...` 或等价检查。
- 与 LangSmith 相关变更，必须能在启用 tracing 时产生 trace（触发一次真实 LLM 调用）。

