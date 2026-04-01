# GRAG-Agent

中医药知识图谱增强问答系统（Graph RAG for TCM）。

## 项目简介

本项目包含两条主线：

- 离线构建：解析药品实体与中医文献数据，写入 Neo4j 并构建 Milvus 向量索引
- 在线问答：基于 LangGraph 的多路检索（图检索 + 向量检索 + 融合 + 重排）生成回答

## 主要能力

- 药品、成分、功效、症状、证候、人群等图谱关系检索
- 药品与中医文献混合召回（Milvus + BM25 + 图谱）
- 多路融合（RRF）与重排（CrossEncoder）
- 会话持久化（SQLite checkpointer）
- MCP Server 封装，可在 Trae 中作为本地 MCP 使用

## 目录结构（关键）

```text
src/app/offline_ingestion   # 离线构建
src/app/online_qa           # 在线问答
src/app/mcp_server.py       # MCP 服务入口
src/core/tools              # 数据库/检索/模型工具
data/                       # 训练与知识数据
```

## 环境准备

建议使用 `uv`：

```bash
uv sync
```

复制环境变量模板：

```bash
cp example.env .env
```

按实际环境填写 `NEO4J_*`、`MILVUS_*`、`KIMI_API_KEY`/`MOONSHOT_API_KEY` 等配置。

## 运行方式

### 1) 离线构建

```bash
uv run python -m src.app.offline_ingestion.cli data -o ./ai_output
```

构建结束会输出：

- Neo4j 节点数 / 关系数
- Milvus 文档数

### 2) 在线问答

```bash
uv run python -m src.app.online_qa.cli --query "乌鸡白凤丸的适应症是什么？" --stream
```

### 3) MCP Server（Trae 本地接入）

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

## 测试

```bash
uv run python -m pytest tests/test_online_qa_graph.py -q
```

## 仓库说明

- 当前实现细节与架构约定见 `CLAUDE.md`
- 仅在本地依赖（Neo4j/Milvus）可用时进行全链路验证
