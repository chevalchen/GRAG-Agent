# GraphRAG 智能菜谱项目（C9）

本目录实现一个“图RAG + 向量RAG + 智能路由”的菜谱问答系统，并包含一个离线 AI Agent，用于把菜谱 Markdown 解析成可导入 Neo4j 的图谱数据。

## 你会用到什么

- Neo4j：存放菜谱知识图谱（Recipe/Ingredient/CookingStep 等节点与关系）
- Milvus：存放向量索引（用于语义检索）
- Moonshot/Kimi（OpenAI SDK 兼容接口）：用于查询分析、图查询意图理解与答案生成
- LangGraph：用于把“分析→检索→后处理”编排成可扩展工作流
- LangSmith：可选，用于 tracing（需要启用并触发一次真实 LLM 调用）

## 目录结构

```
C9/
├─ main.py                       # 在线问答入口（交互式）
├─ config.py                     # GraphRAGConfig 默认配置
├─ requirements.txt              # Python 依赖
├─ .env                          # 仅存放数据库与 API key 等敏感配置（不要提交）
├─ .env.example                  # 环境变量示例
├─ utils/
│  └─ env_utils.py               # 统一加载根目录 .env 并导出 env 常量
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

## 在线问答：如何运行

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 配置环境变量

复制 `.env.example` 为 `.env`，填写必要项：

- Moonshot/Kimi：
  - `MOONSHOT_API_KEY=...`
  - `MOONSHOT_BASE_URL=https://api.moonshot.cn/v1`（一般不需要改）
- Neo4j：
  - `NEO4J_URI=bolt://localhost:7687`
  - `NEO4J_USER=neo4j`
  - `NEO4J_PASSWORD=...`
  - `NEO4J_DATABASE=neo4j`
- Milvus：
  - `MILVUS_HOST=localhost`
  - `MILVUS_PORT=19530`

3) 启动

```bash
python main.py
```

启动后会：
- 初始化 Neo4j/Milvus/LLM 客户端与检索模块
- 检测 Milvus 集合是否已存在；存在则加载，不存在则从 Neo4j 构建并写入 Milvus
- 进入交互式问答

## 在线问答：系统做了什么

- **数据准备**：从 Neo4j 读取图数据，并把每个 Recipe 聚合其 Ingredient/Step 信息形成 Document
- **向量索引**：对 Document 分块并写入 Milvus
- **检索策略**：
  - Hybrid：BM25 + 向量检索 + 图邻居增强
  - GraphRAG：识别查询的图结构意图→生成图查询→多跳遍历/子图抽取→形成上下文
- **智能路由**：用 LLM（或规则降级）分析查询复杂度与关系密集度，自动选择 Hybrid / GraphRAG / Combined
- **生成**：统一生成提示词，把检索结果作为上下文生成最终答案（支持流式）

## 离线 Agent：如何运行（构建图谱数据）

1) 测试单个菜谱解析

```bash
python agent/run_ai_agent.py test
```

2) 批量解析目录

```bash
python agent/run_ai_agent.py /path/to/HowToCook-master
```

3) 断点续传/批次管理

```bash
python agent/batch_manager.py status -o ./ai_output
python agent/batch_manager.py continue /path/to/recipes -o ./ai_output
python agent/batch_manager.py merge -o ./ai_output
```

输出通常包含：
- `ai_output/nodes.csv`
- `ai_output/relationships.csv`
- `ai_output/neo4j_import.cypher`

## LangSmith：如何开启 tracing

本项目的 LLM 调用主要通过 OpenAI SDK（Moonshot base_url），已做 `wrap_openai` instrumentation。

在 `.env` 中设置（两套命名任选其一）：

**方式 A（LangChain 命名）**
- `LANGCHAIN_API_KEY=...`
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=graph-rag-cookbook`（可选）

**方式 B（LangSmith 命名）**
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_TRACING=true`
- `LANGSMITH_PROJECT=graph-rag-cookbook`（可选）

注意：
- 启用 tracing 后必须触发一次真实 LLM 调用（例如运行 `python main.py` 并提问一次）LangSmith 才会出现 trace。

## 常见问题

- **LangSmith 看不到 trace**
  - 确认 `.env` 在本目录根路径（C9/.env）
  - 确认设置了 API_KEY 且 TRACING=true
  - 确认实际触发了 LLM 调用（仅启动不提问不会产生 trace）

- **Milvus/Neo4j 连接失败**
  - 检查服务是否启动、地址端口是否正确
  - 检查 `.env` 中账号密码是否正确

- **首次构建慢**
  - 首次会从 Neo4j 构建文档并写入 Milvus，属于预期行为

## 进一步阅读

- 协作规则与更详细的架构说明：`CLAUDE.md`

