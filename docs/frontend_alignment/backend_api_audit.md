# backend_api_audit

## 审计范围

- 后端实现：`src/app/api_server.py`
- 测试用例：`tests/test_api_server.py`
- 结论时间：2026-05-11

---

## 统一响应协议

### 成功响应

```json
{
  "ok": true,
  "data": {},
  "trace_id": "..."
}
```

### 失败响应

```json
{
  "ok": false,
  "error": {
    "code": "...",
    "message": "...",
    "detail": {}
  },
  "trace_id": "..."
}
```

### Header

- 所有接口都回传 `x-trace-id` 响应头（成功/失败一致）。

---

## 4 个接口请求/响应审计

## 1) 健康检查 `GET /api/v1/health`

### 请求示例

```http
GET /api/v1/health HTTP/1.1
Host: localhost:8000
x-trace-id: trace-health-001
```

### 响应示例（200）

```json
{
  "ok": true,
  "data": {
    "status": "ok"
  },
  "trace_id": "trace-health-001"
}
```

### 对齐结论

- 与测试 `test_health_endpoint` 一致：`status_code=200`，`ok=true`，`data.status=ok`，且存在 `trace_id`。

---

## 2) 问答接口 `POST /api/v1/qa`

### 请求示例（成功）

```http
POST /api/v1/qa HTTP/1.1
Host: localhost:8000
Content-Type: application/json
x-trace-id: trace-qa-001

{
  "question": "测试问题",
  "session_id": "sid-qa"
}
```

### 响应示例（200）

```json
{
  "ok": true,
  "data": {
    "answer": "ok:测试问题:sid-qa",
    "session_id": "sid-qa"
  },
  "trace_id": "trace-qa-001"
}
```

### 失败示例 A（422 参数校验失败）

请求体缺少 `question`：

```json
{
  "session_id": "sid-qa"
}
```

响应：

```json
{
  "ok": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数校验失败。",
    "detail": [
      {
        "loc": ["body", "question"],
        "msg": "Field required",
        "type": "missing"
      }
    ]
  },
  "trace_id": "..."
}
```

### 失败示例 B（400 业务参数无效）

请求 `question` 仅空格，后端 `strip()` 后为空：

```json
{
  "question": "   ",
  "session_id": "sid-qa"
}
```

响应：

```json
{
  "ok": false,
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "参数 question 不能为空。"
  },
  "trace_id": "..."
}
```

### 对齐结论

- 与测试 `test_qa_success` / `test_qa_validation_error` / `test_qa_empty_question` 一致。
- 成功时 `data` 至少包含：`answer`、`session_id`。

---

## 3) 会话列表 `GET /api/v1/sessions`

### 请求示例

```http
GET /api/v1/sessions HTTP/1.1
Host: localhost:8000
x-trace-id: trace-sessions-list-001
```

### 响应示例（200）

```json
{
  "ok": true,
  "data": {
    "sessions": [
      {
        "session_id": "sid-a",
        "checkpoint_id": "cp-a"
      }
    ]
  },
  "trace_id": "trace-sessions-list-001"
}
```

### 对齐结论

- 与测试 `test_sessions_list_and_delete` 一致：可返回存在的 `sid-a`。
- 当底层 DB 异常时，当前实现会降级为 `sessions=[]`（`list_history_sessions` 捕获异常并返回空列表），不会直接报错。

---

## 4) 删除会话 `DELETE /api/v1/sessions/{session_id}`

### 请求示例（成功）

```http
DELETE /api/v1/sessions/sid-a HTTP/1.1
Host: localhost:8000
x-trace-id: trace-sessions-del-001
```

### 响应示例（200）

```json
{
  "ok": true,
  "data": {
    "session_id": "sid-a",
    "deleted": {
      "writes": 0,
      "checkpoints": 1
    }
  },
  "trace_id": "trace-sessions-del-001"
}
```

### 响应示例（404，会话不存在）

```json
{
  "ok": false,
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "会话不存在或已删除。",
    "detail": {
      "session_id": "sid-a"
    }
  },
  "trace_id": "..."
}
```

### 对齐结论

- 与测试 `test_sessions_list_and_delete` 一致：首次删除成功，二次删除返回 `404 + SESSION_NOT_FOUND`。

---

## 错误码映射（现状）

| 场景 | code | HTTP 状态码 | 触发位置 |
|---|---|---:|---|
| 请求体/参数校验失败 | `VALIDATION_ERROR` | 422 | `RequestValidationError` 全局异常处理 |
| 显式 HTTP 异常 | `HTTP_ERROR` | 透传 `exc.status_code` | `HTTPException` 全局异常处理 |
| 内部未捕获异常 | `INTERNAL_ERROR` | 500 | `Exception` 全局异常处理 |
| QA 业务参数为空（空白 question） | `INVALID_ARGUMENT` | 400 | `/api/v1/qa` |
| 删除会话时 path 参数为空字符串（理论分支） | `INVALID_ARGUMENT` | 400 | `/api/v1/sessions/{session_id}` |
| 删除会话目标不存在 | `SESSION_NOT_FOUND` | 404 | `/api/v1/sessions/{session_id}` |
| 删除会话内部错误 | `INTERNAL_ERROR` | 500 | `/api/v1/sessions/{session_id}` |

---

## x-trace-id 规则（现状）

1. 服务读取请求头 `x-trace-id`（`strip()` 后非空即采用）。
2. 若请求头缺失或为空，服务端生成 UUID4 作为 `trace_id`。
3. 响应体固定包含 `trace_id` 字段。
4. 响应头固定设置 `x-trace-id: <trace_id>`。
5. 全局异常处理（422/HTTPException/500）同样遵循以上规则。

---

## 可调用性结论

### 结论

- **可调用：是（通过）**。4 个接口均有可执行路径，且关键路径具备测试覆盖。
- **前后端对齐度：较高**。统一 `ok/data/error/trace_id` 包装、错误码语义基本稳定。

### 风险与边界

- `GET /api/v1/sessions` 在 DB 异常时静默返回空列表，前端难以区分“确实无会话”与“查询失败”。
- `DELETE /api/v1/sessions/{session_id}` 的“空 session_id”分支在路由层通常不可达（FastAPI path 参数不会传空字符串），该错误码更多是防御性代码。

---

## 改进建议（面向前后端对齐）

1. **补充 OpenAPI 示例**：为 4 个接口在 schema 中声明成功/失败示例，减少前端对字段猜测。
2. **细化错误码字典**：沉淀为可共享枚举（例如 `VALIDATION_ERROR`、`INVALID_ARGUMENT`、`SESSION_NOT_FOUND`、`INTERNAL_ERROR`），前端按 code 做统一提示映射。
3. **会话列表失败可观测**：`/sessions` 建议区分“空数据”与“后端故障”（例如返回 500 + `INTERNAL_ERROR`，或附带 `degraded=true`）。
4. **trace 透传规范化**：约束 `x-trace-id` 长度和字符集（如 `^[A-Za-z0-9_-]{1,64}$`），避免异常 header 值污染日志。
5. **补充异常回归测试**：增加 `/sessions` DB 异常与 `/qa` 运行期异常的 API 层测试，确保错误协议持续稳定。

