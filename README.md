# C9 LangGraph GraphRAG 项目

本项目是一个以 LangGraph 为编排框架的 GraphRAG 系统，包含：
- 在线问答（Hybrid 检索 + GraphRAG 检索 + 路由 + 融合 + 生成）
- 离线 ingestion（并发解析菜谱 → 批次落盘 → 导出 Neo4j CSV）

代码主入口与业务编排均在 `src/app/**`，旧逻辑已复制到 `src/legacy/**` 作为可控依赖。

## 1) 安装

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 2) 配置 .env

在根目录创建 `.env`（会被 `src/utils/env_utils.py` 加载）。

最小必需（LLM）：

```env
MOONSHOT_API_KEY=YOUR_KEY
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
```

在线问答还需要（Neo4j + Milvus）：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_PASSWORD
NEO4J_DATABASE=neo4j

MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## 3) 运行在线问答（LangGraph）

```bash
python -m src.app.online_qa
```

单次提问：

```bash
python -m src.app.online_qa --query "红烧肉怎么做？"
```

流式输出：

```bash
python -m src.app.online_qa --stream --query "推荐几个下饭的家常菜"
```

## 4) 运行离线 ingestion（LangGraph）

```bash
python -m src.app.offline_ingestion <recipe_dir> -o ./ai_output --resume
```

可选参数：
- `--batch-size 20`
- `--parse-concurrency 3`
- `--output-format neo4j|csv`

## 5) 运行单元测试

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

## 目录说明

- `src/app/online_qa/`：在线问答 LangGraph（Supervisor + 专家 Agents + Tools + State）
- `src/app/offline_ingestion/`：离线 ingestion LangGraph（scan/parse/flush/export）
- `src/legacy/`：从旧项目复制的可复用模块（DB 接口与检索实现）
- `tests/`：不依赖 Neo4j/Milvus 的单元测试

