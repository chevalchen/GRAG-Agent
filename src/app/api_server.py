from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.app.config import Config
from src.app.qa_service import (
    ServerRuntime,
    ask_question,
    delete_session,
    get_runtime,
    list_history_sessions,
    reset_runtime,
)

class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(default="", description="可选会话ID")


def _build_success(data: Any, trace_id: str) -> dict:
    return {"ok": True, "data": data, "trace_id": trace_id}


def _build_error(code: str, message: str, trace_id: str, detail: Any | None = None) -> dict:
    body = {
        "ok": False,
        "error": {"code": code, "message": message},
        "trace_id": trace_id,
    }
    if detail is not None:
        body["error"]["detail"] = detail
    return body


def _trace_id_from_request(request: Request) -> str:
    header_trace_id = request.headers.get("x-trace-id", "").strip()
    return header_trace_id or str(uuid.uuid4())


def _error_response(status_code: int, code: str, message: str, trace_id: str, detail: Any | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_build_error(code=code, message=message, trace_id=trace_id, detail=detail),
        headers={"x-trace-id": trace_id},
    )


def _map_delete_error_status(code: str) -> int:
    mapping = {
        "INVALID_ARGUMENT": 400,
        "SESSION_NOT_FOUND": 404,
        "INTERNAL_ERROR": 500,
    }
    return mapping.get(code, 500)


def _get_runtime() -> ServerRuntime:
    return get_runtime()


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    _ = _get_runtime()
    try:
        yield
    finally:
        reset_runtime()


app = FastAPI(title="TCM QA REST API", version="0.1.0", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = _trace_id_from_request(request)
    return _error_response(
        status_code=422,
        code="VALIDATION_ERROR",
        message="请求参数校验失败。",
        trace_id=trace_id,
        detail=exc.errors(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    trace_id = _trace_id_from_request(request)
    return _error_response(
        status_code=exc.status_code,
        code="HTTP_ERROR",
        message=str(exc.detail),
        trace_id=trace_id,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    trace_id = _trace_id_from_request(request)
    return _error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="服务内部错误。",
        trace_id=trace_id,
        detail=str(exc),
    )


@app.get("/api/v1/health")
def health(request: Request):
    trace_id = _trace_id_from_request(request)
    return JSONResponse(
        content=_build_success(data={"status": "ok"}, trace_id=trace_id),
        headers={"x-trace-id": trace_id},
    )


@app.post("/api/v1/qa")
def qa(payload: QARequest, request: Request):
    trace_id = _trace_id_from_request(request)
    runtime = _get_runtime()
    question = payload.question.strip()
    if not question:
        return _error_response(
            status_code=400,
            code="INVALID_ARGUMENT",
            message="参数 question 不能为空。",
            trace_id=trace_id,
        )
    result = ask_question(runtime, question=question, session_id=payload.session_id, via="rest")
    return JSONResponse(
        content=_build_success(data=result, trace_id=trace_id),
        headers={"x-trace-id": trace_id},
    )


def _sse_line(event: str, data: dict) -> str:
    """Format a single SSE event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _qa_stream_generator(question: str, session_id: str, trace_id: str) -> AsyncIterator[str]:
    """
    Stream QA via LangGraph astream_events (v2).

    Emits:
      event:token   per LLM chunk from the answer node
      event:suggest with next_questions after the graph completes
      event:done    with session_id and trace_id
      event:error   on failure
    """
    runtime = _get_runtime()
    sid = (session_id or "").strip() or str(uuid.uuid4())
    input_state = {"query": question, "metrics": {"via": "stream"}, "iteration_count": 0}
    graph_config = {"configurable": {"thread_id": sid}}

    try:
        async for event in runtime.graph.astream_events(input_state, config=graph_config, version="v2"):
            if event.get("event") != "on_chat_model_stream":
                continue
            metadata = event.get("metadata") or {}
            if metadata.get("langgraph_node") != "answer":
                continue
            chunk = (event.get("data") or {}).get("chunk")
            if chunk is None:
                continue
            content = getattr(chunk, "content", None)
            if content:
                yield _sse_line("token", {"chunk": content})

        # Retrieve next_questions from the persisted final state
        next_questions: list[str] = []
        try:
            snapshot = runtime.graph.get_state(config=graph_config)
            next_questions = list((snapshot.values or {}).get("next_questions") or [])
        except Exception:
            pass

        yield _sse_line("suggest", {"questions": next_questions})
        yield _sse_line("done", {"session_id": sid, "trace_id": trace_id})

    except Exception as exc:
        yield _sse_line("error", {"code": "INTERNAL_ERROR", "message": str(exc)})


@app.post("/api/v1/qa/stream")
async def qa_stream(payload: QARequest, request: Request):
    trace_id = _trace_id_from_request(request)
    question = payload.question.strip()
    if not question:
        return _error_response(
            status_code=400,
            code="INVALID_ARGUMENT",
            message="参数 question 不能为空。",
            trace_id=trace_id,
        )
    return StreamingResponse(
        _qa_stream_generator(question, payload.session_id, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "x-trace-id": trace_id,
        },
    )


@app.get("/api/v1/sessions")
def session_list(request: Request):
    trace_id = _trace_id_from_request(request)
    runtime = _get_runtime()
    sessions = list_history_sessions(runtime.config.checkpointer_path)
    return JSONResponse(
        content=_build_success(data={"sessions": sessions}, trace_id=trace_id),
        headers={"x-trace-id": trace_id},
    )


@app.delete("/api/v1/sessions/{session_id}")
def session_delete(session_id: str, request: Request):
    trace_id = _trace_id_from_request(request)
    runtime = _get_runtime()
    sid = (session_id or "").strip()
    if not sid:
        return _error_response(
            status_code=400,
            code="INVALID_ARGUMENT",
            message="参数 session_id 不能为空。",
            trace_id=trace_id,
            detail={"session_id": session_id},
        )
    result = delete_session(runtime.config.checkpointer_path, sid)
    if result.get("code") != "OK":
        return _error_response(
            status_code=_map_delete_error_status(result.get("code", "")),
            code=result.get("code", "INTERNAL_ERROR"),
            message=result.get("message", "删除会话失败。"),
            trace_id=trace_id,
            detail=result.get("detail"),
        )
    return JSONResponse(
        content=_build_success(data=result.get("detail", {}), trace_id=trace_id),
        headers={"x-trace-id": trace_id},
    )


def main():
    import uvicorn

    uvicorn.run("src.app.api_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
