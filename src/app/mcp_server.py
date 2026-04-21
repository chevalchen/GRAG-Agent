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
            "embedding_model": Config.FAST_EMBEDDING_MODEL if Config.FAST_MODE else Config.EMBEDDING_MODEL,
            "llm_model": Config.FAST_LLM_MODEL if Config.FAST_MODE else Config.LLM_MODEL,
            "bm25_top_k": Config.BM25_TOP_K,
            "reranker_model": Config.FAST_RERANKER_MODEL if Config.FAST_MODE else Config.RERANKER_MODEL,
            "retrieval_balance_strategy": Config.RETRIEVAL_BALANCE_STRATEGY,
            "retrieval_timeout_seconds": Config.RETRIEVAL_TIMEOUT_SECONDS,
            "retrieve_expand_factor": Config.RETRIEVE_EXPAND_FACTOR,
            "lit_expand_factor": Config.LIT_EXPAND_FACTOR,
            "graph_expand_factor": Config.GRAPH_EXPAND_FACTOR,
            "max_retrieval_top_k": Config.MAX_RETRIEVAL_TOP_K,
            "max_graph_rows": Config.MAX_GRAPH_ROWS,
            "graph_fallback_max_nodes": Config.GRAPH_FALLBACK_MAX_NODES,
            "rerank_simple_skip_threshold": Config.RERANK_SIMPLE_SKIP_THRESHOLD,
            "rerank_simple_candidate_limit": Config.RERANK_SIMPLE_CANDIDATE_LIMIT,
            "rerank_complex_candidate_limit": Config.RERANK_COMPLEX_CANDIDATE_LIMIT,
            "simple_context_budget_chars": Config.SIMPLE_CONTEXT_BUDGET_CHARS,
            "complex_context_budget_chars": Config.COMPLEX_CONTEXT_BUDGET_CHARS,
            "simple_per_doc_chars": Config.SIMPLE_PER_DOC_CHARS,
            "complex_per_doc_chars": Config.COMPLEX_PER_DOC_CHARS,
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
        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        table_names = {str(name) for (name,) in (cur.fetchall() or [])}
        has_checkpoints = "checkpoints" in table_names
        has_writes = "writes" in table_names
        sessions: list[dict] = []
        visible_session_ids: set[str] = set()
        if has_checkpoints:
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
            for thread_id, checkpoint_id in rows:
                session_id = str(thread_id)
                sessions.append(
                    {"session_id": session_id, "checkpoint_id": str(checkpoint_id)}
                )
                visible_session_ids.add(session_id)
        if has_writes:
            cur.execute(
                """
                SELECT thread_id
                FROM writes
                GROUP BY thread_id
                ORDER BY MAX(rowid) DESC
                """
            )
            write_rows = cur.fetchall() or []
            for (thread_id,) in write_rows:
                session_id = str(thread_id)
                if session_id in visible_session_ids:
                    continue
                sessions.append({"session_id": session_id, "checkpoint_id": ""})
                visible_session_ids.add(session_id)
        conn.close()
    except Exception:
        return []
    return sessions


def _delete_session(db_path: str, session_id: str) -> dict:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        writes_deleted = cur.rowcount if cur.rowcount is not None else 0
        cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        checkpoints_deleted = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
        conn.close()
        total_deleted = writes_deleted + checkpoints_deleted
        if total_deleted <= 0:
            return {
                "code": "SESSION_NOT_FOUND",
                "message": "会话不存在或已删除。",
                "detail": {"session_id": session_id},
            }
        return {
            "code": "OK",
            "message": "会话删除成功。",
            "detail": {
                "session_id": session_id,
                "deleted": {
                    "writes": writes_deleted,
                    "checkpoints": checkpoints_deleted,
                },
            },
        }
    except Exception as ex:
        return {
            "code": "INTERNAL_ERROR",
            "message": "删除会话时发生内部错误。",
            "detail": {"session_id": session_id, "error": str(ex)},
        }


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
        return json.dumps(
            {
                "code": "INVALID_ARGUMENT",
                "message": "参数 session_id 不能为空。",
                "detail": {"session_id": session_id},
            },
            ensure_ascii=False,
        )
    db_path = runtime.config.checkpointer_path
    result = _delete_session(db_path, sid)
    return json.dumps(result, ensure_ascii=False)


def main():
    Path(".checkpoints").mkdir(parents=True, exist_ok=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
