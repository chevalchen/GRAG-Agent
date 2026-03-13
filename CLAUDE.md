# CLAUDE.md

本文件为 AI 辅助开发提供项目上下文，帮助 Claude 等工具快速理解代码库结构与约定。

---

## 项目概述

这是一个**菜谱知识图谱问答系统**，基于 GraphRAG 架构，结合 Neo4j 图数据库、Milvus 向量数据库和大语言模型（LLM），支持对菜谱知识进行智能问答。

核心功能：
- **离线 Ingestion**：解析 Markdown 格式菜谱，通过 AI Agent（Kimi/Moonshot）提取结构化知识，构建 Neo4j 知识图谱
- **在线问答**：基于 LangGraph 的多节点流水线，支持混合检索（Hybrid RAG + Graph RAG）和多轮对话

---

## 项目结构
```
src/
├── app/                         # 应用层（LangGraph 流水线）
│   ├── config.py                # 全局配置（GraphRAGConfig, DEFAULT_CONFIG）
│   ├── offline_ingestion/       # 离线知识入库流水线
│   │   ├── cli.py               # 入口：python -m src.app.offline_ingestion.cli
│   │   └── graphs/ingestion_graph.py
│   └── online_qa/               # 在线问答流水线
│       ├── cli.py               # 入口：python -m src.app.online_qa.cli
│       ├── graphs/online_qa_graph.py   # LangGraph 图定义
│       ├── nodes/               # 各节点（query_analysis, route, retrieve, fuse, answer）
│       ├── tools/               # 工具封装（hybrid_search, graph_rag_search 等）
│       ├── state.py             # OnlineQAState TypedDict
│       └── checkpointer.py      # SQLite checkpoint（多轮对话持久化）
└── legacy/                      # 底层模块（被 app 层调用）
    ├── rag_modules/
    │   ├── graph_data_preparation.py    # Neo4j 数据加载与文档构建
    │   ├── milvus_index_construction.py # Milvus 向量索引
    │   ├── hybrid_retrieval.py          # 混合检索（BM25 + Milvus + 图）
    │   ├── graph_rag_retrieval.py       # 图 RAG 检索（多跳推理）
    │   ├── graph_indexing.py            # 实体/关系键值对索引
    │   ├── intelligent_query_router.py  # 智能路由（hybrid / graph_rag / combined）
    │   └── generation_integration.py   # LLM 答案生成
    └── agent/
        ├── recipe_ai_agent.py   # Kimi AI Agent（菜谱解析）
        ├── batch_manager.py     # 批量处理与断点续传
        └── run_ai_agent.py      # 旧版运行入口
```

---

## 常用命令

### 离线入库（解析菜谱 → Neo4j）
```bash
python -m src.app.offline_ingestion.cli ./HowToCook-master \
    -o ./ai_output \
    --output-format neo4j \
    --batch-size 20 \
    --resume
```

### 在线问答（交互式 CLI）
```bash
python -m src.app.online_qa.cli
# 单次提问
python -m src.app.online_qa.cli --query "红烧肉怎么做？" --stream
# 指定会话 ID（多轮对话）
python -m src.app.online_qa.cli --session-id my_session
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 流水线框架 | LangGraph 1.1.0 |
| 图数据库 | Neo4j 6.x |
| 向量数据库 | Milvus（pymilvus 2.5.x）|
| 嵌入模型 | `BAAI/bge-small-zh-v1.5`（HuggingFace，512 维）|
| LLM | OpenAI 兼容接口（默认 Kimi/Moonshot）|
| BM25 检索 | `rank-bm25` + LangChain BM25Retriever |
| 对话持久化 | SQLite（`.checkpoints/c9.db`）|
| 依赖管理 | pip / requirements.txt |

---

## 配置

配置入口：`src/app/config.py`，关键类：

- `Config`：从环境变量读取（`SESSION_ID` 等）
- `GraphRAGConfig`：完整 RAG 配置（Neo4j、Milvus、LLM、检索参数）
- `DEFAULT_CONFIG`：默认配置实例

主要环境变量（建议写入 `.env`）：
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
MILVUS_HOST=localhost
MILVUS_PORT=19530
KIMI_API_KEY=sk-xxx
SESSION_ID=（可选，固定会话 ID）
```

---

## 架构说明

### 在线问答流（LangGraph）
```
START → query_analysis → route → hybrid_retrieve ─┐
                                → graph_retrieve  ─┴→ fuse → answer → END
```

- **query_analysis**：提取关键词、意图、复杂度，决定路由策略
- **route**：根据 `analysis.recommended_strategy` 路由到 `hybrid` / `graph_rag` / `combined`
- **hybrid_retrieve**：BM25 + Milvus 向量检索 + 图一跳扩展
- **graph_retrieve**：多跳图遍历、子图提取、路径推理
- **fuse**：Round-robin 合并去重
- **answer**：LLM 生成最终回答，支持流式输出

### 检索策略路由

| 策略 | 触发条件 |
|------|----------|
| `hybrid` | 普通问答，关键词匹配为主 |
| `graph_rag` | 高关系密集度，需要多跳推理 |
| `combined` | 高复杂度，两路并行检索后融合 |

---

## 开发约定

- 所有底层模块位于 `src/legacy/`，应用层封装在 `src/app/`
- 新增检索节点请参照 `src/app/online_qa/nodes/` 下的现有节点模式
- LangGraph 状态类型定义在 `src/app/online_qa/state.py`（`OnlineQAState` TypedDict）
- 日志级别通过 `--log-level` 参数控制，默认 `WARN`
- 安装依赖：`pip install -r requirements.txt --break-system-packages`
- PyTorch 使用 CUDA 版本（`torch==2.6.0+cu126`），确保 CUDA 环境匹配

---

## 数据格式

菜谱 Markdown 文件位于 `./HowToCook-master/dishes/` 下，按菜系分子目录。

Neo4j 图节点类型：
- `Recipe`：菜谱节点（nodeId 前缀 `1xxx`）
- `Ingredient`：食材节点
- `CookingStep`：烹饪步骤节点（nodeId ≥ `200000000`）

关系类型：`REQUIRES`（菜谱→食材）、`CONTAINS_STEP`（菜谱→步骤）