# Frontend E2E Validation（FastAPI 4 接口）

## 1. 验证范围

本次验证聚焦前后端最小联调闭环，覆盖以下接口：

1. `GET /api/v1/health`
2. `POST /api/v1/qa`
3. `GET /api/v1/sessions`
4. `DELETE /api/v1/sessions/{session_id}`

验证目标：

- 前端契约层是否正确消费统一信封（`ok/data/error/trace_id`）
- `x-trace-id` 是否可透传与回显
- 关键交互链路（健康检查→提问→会话查询→会话删除）是否闭环
- 错误模型映射是否符合审计结论

## 2. 验证环境与命令

- 日期：2026-05-11
- 目录：`D:\CODE\MedicalRAG\GRAG-Agent`
- 命令：

```bash
node --test frontend-minimal/__tests__/*.test.mjs
```

测试结论（本地执行）：

- `tests=4`
- `pass=4`
- `fail=0`

## 3. 4 接口验证结果

| 接口 | 验证方式 | 结果 | 说明 |
|---|---|---|---|
| `GET /api/v1/health` | `apiClient.contract` + `app.interaction` | 通过 | 正确解析 `ok/data`，并优先读取响应头 `x-trace-id` |
| `POST /api/v1/qa` | `apiClient.contract` + `app.interaction` | 通过 | 成功回填 `answer` 与 `session_id`，失败时按 `error.code` 分类抛错 |
| `GET /api/v1/sessions` | `apiClient.contract` + `app.interaction` | 通过 | 成功渲染会话列表，点击会话可回填当前 `session_id` |
| `DELETE /api/v1/sessions/{session_id}` | `app.interaction` | 通过 | 删除成功后刷新列表并清空当前会话；缺失会话路径可映射 `SESSION_NOT_FOUND` |

## 4. 测试项清单

- 契约测试：
  - 健康检查成功解析（含 trace 优先级）
  - 后端错误码 `SESSION_NOT_FOUND` 映射为前端 `session` 分类
  - 非法响应体映射 `HTTP_ERROR`
- 交互测试：
  - 完整链路：health → qa → sessions → delete
  - UI 状态更新：`healthText`、`answerText`、`sessionList`、`sessionId`
  - 异常状态检查：无异常时 `错误：-`

## 5. 风险与遗留项

1. **浏览器与 Node 测试环境差异**
   - 当前交互测试使用最小 fake DOM，不覆盖真实浏览器事件模型（如焦点、键盘、复杂冒泡）。
   - 建议后续补充 Playwright/Cypress 浏览器级别联调。

2. **网络异常覆盖不足**
   - 已覆盖基础 `NETWORK_ERROR`/`HTTP_ERROR` 契约，但未覆盖超时重试与抖动策略。
   - 建议增加超时注入测试（请求取消、重试退避）。

3. **后端真实实例联调证据缺失**
   - 当前测试通过 mock fetch 验证前端契约与交互，不等价于真实后端运行时联调。
   - 建议增加 CI 中的“启动 FastAPI 后执行前端联调用例”阶段。

4. **会话并发操作场景未覆盖**
   - 未覆盖并发删除、跨标签页 session 竞争更新等场景。
   - 建议补充冲突与幂等性用例。
