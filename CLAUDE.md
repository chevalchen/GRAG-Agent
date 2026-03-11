## 项目概览（LangGraph GraphRAG 智能菜谱）

本项目是一个以 **LangGraph** 为工作流编排框架的 GraphRAG 系统，包含两条主线：

1) **在线问答（Online QA）**
- 能力：查询路由（Hybrid / GraphRAG / Combined）→ 检索并发 → 融合 → 生成式回答
- 入口：`python -m src.app.online_qa`（或 `python main.py` 作为兼容入口）

2) **离线 Ingestion（菜谱图谱构建）**
- 能力：扫描菜谱 → 并发解析（LLM）→ 批次落盘 → 导出 Neo4j CSV
- 入口：`python -m src.app.offline_ingestion <recipe_dir> -o ./ai_output --resume`

---

## 工程分层与封装边界（哪些是“封装旧模块”）

### 业务编排层（新：LangGraph）

这些文件是当前项目的“主业务逻辑入口”，负责工作流编排、状态流转、并发控制、工具权限约束：
- `src/app/online_qa/graphs/online_qa_graph.py`：在线问答 LangGraph（Supervisor → Route → Retrieve → Fuse → Generate）
- `src/app/offline_ingestion/graphs/ingestion_graph.py`：离线 ingestion LangGraph（init/scan/parse_batch/flush/finalize/export）
- `src/app/online_qa/agents/*`：Supervisor + 专家 Agents（路由/检索/生成），只允许访问白名单工具
- `src/app/online_qa/tools/*`：Tool 封装（Hybrid/GraphRAG/生成），统一 async 适配
- `src/app/online_qa/state.py`、`src/app/offline_ingestion/state.py`：LangGraph state 定义
- `src/app/online_qa/cli.py`、`src/app/offline_ingestion/cli.py`：标准 CLI 入口（`python -m ...`）

### Legacy 实现层（旧代码复制进来，被上层封装调用）

`src/legacy/**` 是从旧项目复制过来的实现代码，用于保持行为一致、降低重构风险。
这些文件本身不是 LangGraph 编排的一部分，它们被 `src/app/**` 的 Tools/Agents “包一层”调用：

- `src/legacy/rag_modules/*`：
  - `hybrid_retrieval.py`：Hybrid 检索核心实现（含 BM25/Milvus/图邻居增强）
  - `graph_rag_retrieval.py`：GraphRAG 检索核心实现（Neo4j 图查询/子图/路径）
  - `intelligent_query_router.py`：LLM/规则路由策略（策略效果保持一致）
  - `graph_data_preparation.py`：Neo4j → 文档构建与切块
  - `milvus_index_construction.py`：Milvus 索引构建与相似检索
  - `generation_integration.py`：LLM 生成（OpenAI SDK / Moonshot）

- `src/legacy/agent/*`：
  - `recipe_ai_agent.py`：菜谱解析与图谱导出（被离线 ingestion graph 编排调用）
  - `batch_manager.py` / `run_ai_agent.py`：legacy 运行脚本（阶段性保留，推荐使用 `src/app/offline_ingestion`）

结论：**`src/app/**` 是新项目的 LangGraph 编排与工程形态；`src/legacy/**` 主要是对旧 Agent/Module 的复制与被封装层。**

---

## 配置与环境变量

### 配置
- 主配置：`src/app/config.py`
- 兼容导出：根目录 `config.py` 仅 re-export（保持旧导入不崩）

### 环境变量
- 主加载：`src/utils/env_utils.py`（从项目根目录 `.env` 加载）
- 兼容导出：根目录 `utils/env_utils.py` 仅 re-export

常用变量：
- Moonshot/Kimi：`MOONSHOT_API_KEY`、`MOONSHOT_BASE_URL`、`KIMI_API_KEY`
- Neo4j：`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`NEO4J_DATABASE`
- Milvus：`MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION_NAME`、`MILVUS_DIMENSION`

---

## 运行方式（开发者常用）

在线问答：
```bash
python -m src.app.online_qa
python -m src.app.online_qa --query "红烧肉怎么做？"
python -m src.app.online_qa --stream --query "推荐几个下饭菜"
```

离线 ingestion：
```bash
python -m src.app.offline_ingestion <recipe_dir> -o ./ai_output --resume
```

单元测试：
```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

---

## 项目结构（当前）

```
C9/
├─ README.md
├─ main.py                        # 兼容入口（转发到 src.app.online_qa）
├─ config.py                      # 兼容导出（re-export src/app/config.py）
├─ requirements.txt
├─ utils/
│  └─ env_utils.py                # 兼容导出（re-export src/utils/env_utils.py）
├─ src/
│  ├─ app/
│  │  ├─ config.py
│  │  ├─ online_qa/
│  │  │  ├─ __main__.py
│  │  │  ├─ cli.py
│  │  │  ├─ state.py
│  │  │  ├─ validators.py
│  │  │  ├─ graphs/
│  │  │  │  └─ online_qa_graph.py
│  │  │  ├─ agents/
│  │  │  │  ├─ supervisor.py
│  │  │  │  ├─ router_agent.py
│  │  │  │  ├─ hybrid_agent.py
│  │  │  │  ├─ graph_agent.py
│  │  │  │  ├─ answer_agent.py
│  │  │  │  └─ fusion.py
│  │  │  └─ tools/
│  │  │     ├─ registry.py
│  │  │     ├─ hybrid_search.py
│  │  │     ├─ graph_rag_search.py
│  │  │     └─ answer_generation.py
│  │  └─ offline_ingestion/
│  │     ├─ __main__.py
│  │     ├─ cli.py
│  │     ├─ state.py
│  │     ├─ graphs/
│  │     │  └─ ingestion_graph.py
│  │     └─ tools/
│  │        └─ scan_files.py
│  ├─ legacy/
│  │  ├─ rag_modules/             # 从旧项目复制的 Module（被封装调用）
│  │  └─ agent/                   # 从旧项目复制的 Agent（被封装调用）
│  └─ utils/
│     └─ env_utils.py
├─ tests/
│  ├─ test_online_qa_graph.py
│  ├─ test_fusion.py
│  ├─ test_scan_files.py
│  └─ test_tool_permissions.py
└─ PROJECT_PLAN_LANGGRAPH.md
```

