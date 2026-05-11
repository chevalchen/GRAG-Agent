# GRAG-Agent

中医药知识图谱增强问答系统（Graph RAG for Traditional Chinese Medicine）。

---

## 项目简介

本项目是一个基于 GraphRAG 架构的中医药智能问诊系统，包含两条核心主线：

- **离线构建**：解析药品 NER 实体与中医文献数据，写入 Neo4j 知识图谱并构建 Milvus 向量索引
- **在线问答**：基于 LangGraph 的多路检索（图谱检索 + 药品向量 + 文献向量 + 健康科普向量 + RRF 融合 + CrossEncoder 重排）生成智能回答

---

## 核心能力

- **知识图谱检索**：药品、成分、功效、症状、证候、人群等实体关系查询
- **混合召回**：Milvus 向量检索 + BM25 稀疏检索 + Neo4j 图谱检索
- **智能融合**：RRF（倒数排名融合）整合多路检索结果
- **精排优化**：CrossEncoder 重排序提升结果质量
- **会话管理**：SQLite 持久化会话状态，支持多轮对话
- **MCP 接入**：封装为 MCP Server，可在 Trae 等客户端作为本地工具使用

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 工作流引擎 | LangGraph 1.x |
| 知识图谱 | Neo4j |
| 向量数据库 | Milvus |
| 稀疏检索 | BM25 (rank-bm25) |
| 重排序 | CrossEncoder (BAAI/bge-reranker-base) |
| 大语言模型 | Moonshot / Kimi API |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 会话持久化 | SQLite |
| MCP 框架 | FastMCP |

---

## 目录结构

```text
src/
├── app/
│   ├── config.py                    # 配置管理
│   ├── mcp_server.py                # MCP 服务入口
│   ├── online_qa/                   # 在线问答模块
│   │   ├── cli.py                   # 命令行入口
│   │   ├── checkpointer.py          # 会话持久化
│   │   ├── state.py                 # 状态定义
│   │   ├── graphs/
│   │   │   └── online_qa_graph.py   # 问答工作流图
│   │   └── nodes/                   # 节点实现
│   │       ├── supervisor_node.py           # 意图路由
│   │       ├── drug_entity_resolve_node.py  # 药品实体解析
│   │       ├── graph_retrieve_node.py       # 图谱检索
│   │       ├── drug_retrieve_node.py        # 药品向量检索
│   │       ├── lit_retrieve_node.py         # 文献检索
│   │       ├── health_retrieve_node.py      # 健康科普检索
│   │       ├── fuse_node.py                 # RRF 融合
│   │       ├── rerank_node.py               # 精排
│   │       └── answer_node.py               # 答案生成
│   └── offline_ingestion/           # 离线构建模块
│       ├── cli.py                   # 命令行入口
│       ├── state.py                 # 状态定义
│       ├── graphs/
│       │   └── ingestion_graph.py   # 构建工作流图
│       ├── nodes/                   # 节点实现
│       │   ├── parse_node.py        # NER 解析
│       │   ├── chunk_node.py        # 文档分块
│       │   ├── health_chunk_node.py # 健康科普分块
│       │   ├── normalize_node.py    # 实体归一化
│       │   ├── neo4j_write_node.py  # 写入 Neo4j
│       │   ├── milvus_index_node.py # 写入 Milvus（tcm_knowledge）
│       │   └── health_milvus_index_node.py # 写入 Milvus（health_knowledge）
│       └── tools/                   # 工具函数
├── core/
│   ├── schemas/document.py          # 文档模型
│   └── tools/
│       ├── db/                      # 数据库客户端
│       │   ├── neo4j_client.py
│       │   └── milvus_client.py
│       ├── graph/neo4j_tool.py      # 图谱操作
│       ├── retrieval/bm25_tool.py   # BM25 检索
│       ├── vector/milvus_tool.py    # 向量检索
│       └── llm/                     # LLM 相关
│           ├── llm_client.py
│           ├── embedding_client.py
│           └── generation_tool.py
│   └── utils/
│       ├── drug_name_normalizer.py  # 药品名归一化
│       └── recipe_loader.py
├── legacy/                          # 历史代码（保留参考）
│   ├── agent/
│   └── rag_modules/
├── data/                            # 数据目录
├── tests/                           # 测试用例
└── scripts/                         # 脚本工具
```

---

## 环境准备

### 1. 安装依赖

建议使用 `uv`：

```bash
uv sync
```

### 2. 配置环境变量

复制环境变量模板：

```bash
cp example.env .env
```

按实际环境填写以下配置：

```bash
# Neo4j 连接
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Milvus 连接
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION_NAME=tcm_knowledge
MILVUS_HEALTH_COLLECTION_NAME=health_knowledge

# API 密钥（二选一）
MOONSHOT_API_KEY=your_moonshot_key
KIMI_API_KEY=your_kimi_key

# 可选：性能调优
FAST_MODE=false
BM25_TOP_K=10
TOP_K=3
CHECKPOINTER_PATH=.checkpoints/tcm.db
```

### 3. 启动依赖服务

确保 Neo4j 和 Milvus 服务已启动并可访问。

---

## 运行方式

### 1. 离线构建

解析数据文件并构建知识图谱与向量索引（支持 `health/*.jsonl`）：

```bash
uv run python -m src.app.offline_ingestion.cli data -o ./ai_output --resume
```

构建结束会输出：
- Neo4j 节点数 / 关系数
- Milvus 主集合文档数（`milvus_docs`，即 `tcm_knowledge`）
- Milvus 健康集合文档数（`health_milvus_docs`，即 `health_knowledge`）

### 2. 在线问答（CLI）

交互模式：

```bash
uv run python -m src.app.online_qa.cli
```

单问模式：

```bash
uv run python -m src.app.online_qa.cli \
  --query "乌鸡白凤丸的适应症是什么？" \
  --stream
```

### 3. MCP Server（Trae 本地接入）

启动 MCP 服务：

```bash
uv run python -m src.app.mcp_server
```

Trae MCP 配置示例：

```json
{
  "mcpServers": {
    "tcm-assistant": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "src.app.mcp_server"
      ],
      "cwd": "D:\\CODE\\MedicalRAG\\GRAG-Agent"
    }
  }
}
```

### 4. REST API（MVP 最小调用）

启动 REST 服务：

```bash
uv run python -m src.app.api_server
```

健康检查：

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/health"
```

问答接口：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/qa" \
  -H "Content-Type: application/json" \
  -H "x-trace-id: demo-trace-001" \
  -d "{\"question\":\"乌鸡白凤丸的适应症是什么？\",\"session_id\":\"demo-session-1\"}"
```

会话列表与删除：

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/sessions"
curl -X DELETE "http://127.0.0.1:8000/api/v1/sessions/demo-session-1"
```

> 注：`api_server` 已内置 CORS，默认允许 `http://127.0.0.1:5500` 与 `http://localhost:5500`，便于本地静态前端直接联调。

### 5. 前端页面（frontend-minimal）

启动静态服务（保留 `-d frontend-minimal`）：

```bash
python -m http.server 5500 -d frontend-minimal
```

页面入口：
- 控制台版：`http://127.0.0.1:5500/index.html`
- 人群咨询版：`http://127.0.0.1:5500/audience.html`

人群咨询版特性：
- 面向病人、咨询人员、中医药爱好者三类用户
- 错误采用弹窗展示（含 `code/status/trace_id/detail`）
- 回答自动封装为摘要结论、可执行建议、风险与注意、追问方向

---

## 测试

运行全部测试：

```bash
uv run python -m pytest tests -v
```

运行特定测试：

```bash
uv run python -m pytest tests/test_online_qa_graph.py -v
```

---

## 性能基准

测试命令：

```bash
python -m scripts.benchmark_online_qa \
  --queries-file scripts/benchmark_queries_task5.txt \
  --runs 1 \
  --p95-target 6.0 \
  --output benchmark/task5_benchmark_result.json
```

最新结果（2026-04-16）：

| 指标 | 数值 |
|------|------|
| 样本数 | 6 |
| 平均耗时 | 6.082s |
| P50 | 5.507s |
| P95 | 8.715s |
| 目标 | 6.000s（未达标） |
| 瓶颈 | supervisor_node（意图分析） |

---

## 文档索引

| 文档 | 说明 |
|------|------|
| `CLAUDE.md` | 详细架构设计与开发规范（英文） |
| `README.md` | 项目简介与使用指南（本文档） |

---

## 设计原则

1. **节点化扩展**：在线能力通过新增节点实现，避免全局单例
2. **依赖注入**：工具通过工厂函数注入节点，便于测试和替换
3. **统一归一化**：药品名称归一化逻辑离线/在线共享
4. **渐进式降级**：图谱检索失败时自动回退关键词链路
5. **复杂度感知**：简单查询可跳过重排，限制上下文预算

---

## 许可证

MIT License
