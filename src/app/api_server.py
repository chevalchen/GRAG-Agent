from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
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
