from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.app.config import Config, DEFAULT_CONFIG, GraphRAGConfig
from src.app.online_qa.graphs.online_qa_graph import build_graph
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import build_bm25_from_milvus
from src.core.tools.vector.milvus_tool import MilvusVectorTool


@dataclass
class ServerRuntime:
    config: GraphRAGConfig
    graph: object


def _build_config() -> GraphRAGConfig:
    return GraphRAGConfig(
        **{
            **DEFAULT_CONFIG.to_dict(),
            "neo4j_uri": Config.NEO4J_URI,
            "neo4j_user": Config.NEO4J_USER,
            "neo4j_password": Config.NEO4J_PASSWORD,
            "neo4j_database": Config.NEO4J_DATABASE,
            "milvus_host": Config.MILVUS_HOST,
            "milvus_port": Config.MILVUS_PORT,
            "milvus_uri": Config.MILVUS_URI,
            "milvus_collection_name": Config.MILVUS_COLLECTION_NAME,
            "bm25_top_k": Config.BM25_TOP_K,
            "reranker_model": Config.RERANKER_MODEL,
            "checkpointer_path": Config.CHECKPOINTER_PATH,
        }
    )


def _build_runtime() -> ServerRuntime:
    config = _build_config()
    neo4j_tool = Neo4jGraphTool(config)
    milvus_tool = MilvusVectorTool(config)
    _ = milvus_tool.load_collection()
    bm25_tool = build_bm25_from_milvus(
        milvus_tool.client,
        top_k=config.bm25_top_k,
        collection_name=milvus_tool.collection_name,
    )
    llm_tool = LLMGenerationTool(config)
    graph = build_graph(
        config,
        bm25_tool=bm25_tool,
        milvus_tool=milvus_tool,
        neo4j_tool=neo4j_tool,
        llm_tool=llm_tool,
    )
    return ServerRuntime(config=config, graph=graph)


_runtime: ServerRuntime | None = None


@asynccontextmanager
async def app_lifespan(_: FastMCP):
    global _runtime
    _runtime = _build_runtime()
    try:
        yield {"ready": True}
    finally:
        _runtime = None


def _get_runtime() -> ServerRuntime:
    global _runtime
    if _runtime is None:
        _runtime = _build_runtime()
    return _runtime


def _list_history_sessions(db_path: str) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.thread_id, c.checkpoint_id
            FROM checkpoints c
            JOIN (
                SELECT thread_id, MAX(rowid) AS last_rowid
                FROM checkpoints
                GROUP BY thread_id
            ) last ON last.thread_id = c.thread_id AND last.last_rowid = c.rowid
            ORDER BY c.rowid DESC
            """
        )
        rows = cur.fetchall() or []
        conn.close()
    except Exception:
        return []
    return [{"session_id": str(thread_id), "checkpoint_id": str(checkpoint_id)} for thread_id, checkpoint_id in rows]


def _delete_session(db_path: str, session_id: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


mcp = FastMCP(name="tcm-assistant", lifespan=app_lifespan)


@mcp.tool(name="tcm_query")
def tcm_query(question: str, session_id: str = "") -> str:
    runtime = _get_runtime()
    query = (question or "").strip()
    if not query:
        return "请提供问题。"
    sid = (session_id or "").strip() or str(uuid.uuid4())
    state = runtime.graph.invoke(
        {"query": query, "metrics": {"via": "mcp"}},
        config={"configurable": {"thread_id": sid}},
    )
    answer = (state or {}).get("answer") or ""
    return answer or "未生成回答。"


@mcp.tool(name="session_list")
def session_list() -> str:
    runtime = _get_runtime()
    db_path = runtime.config.checkpointer_path
    sessions = _list_history_sessions(db_path)
    return json.dumps({"sessions": sessions}, ensure_ascii=False)


@mcp.tool(name="session_delete")
def session_delete(session_id: str) -> str:
    runtime = _get_runtime()
    sid = (session_id or "").strip()
    if not sid:
        return "missing_session_id"
    db_path = runtime.config.checkpointer_path
    ok = _delete_session(db_path, sid)
    return "ok" if ok else "failed"


def main():
    Path(".checkpoints").mkdir(parents=True, exist_ok=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
