from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.app.qa_service import (
    ServerRuntime,
    ask_question,
    delete_session,
    get_runtime,
    list_history_sessions,
    reset_runtime,
)


@asynccontextmanager
async def app_lifespan(_: FastMCP):
    _ = get_runtime()
    try:
        yield {"ready": True}
    finally:
        reset_runtime()


def _get_runtime() -> ServerRuntime:
    return get_runtime()


def _list_history_sessions(db_path: str) -> list[dict]:
    return list_history_sessions(db_path)


def _delete_session(db_path: str, session_id: str) -> dict:
    return delete_session(db_path, session_id)


mcp = FastMCP(name="tcm-assistant", lifespan=app_lifespan)


@mcp.tool(name="tcm_query")
def tcm_query(question: str, session_id: str = "") -> str:
    runtime = _get_runtime()
    try:
        result = ask_question(runtime, question=question, session_id=session_id, via="mcp")
    except ValueError as ex:
        return str(ex)
    return result.get("answer") or "未生成回答。"


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
