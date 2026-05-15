import test from "node:test";
import assert from "node:assert/strict";

import { createApiClient, ApiClientError } from "../apiClient.js";

function mockJsonResponse(status, payload, headers = {}) {
  return {
    status,
    headers: {
      get(name) {
        return headers[name.toLowerCase()] || null;
      },
    },
    async json() {
      return payload;
    },
  };
}

test("health: parses envelope and returns trace_id", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async (url, options) => {
    assert.equal(url, "http://127.0.0.1:8000/api/v1/health");
    assert.equal(options.method, "GET");
    assert.ok(options.headers["x-trace-id"]);
    return mockJsonResponse(
      200,
      { ok: true, data: { status: "ok" }, trace_id: "body-trace" },
      { "x-trace-id": "header-trace" }
    );
  };

  const api = createApiClient(() => "http://127.0.0.1:8000");
  const result = await api.health();
  assert.deepEqual(result.data, { status: "ok" });
  assert.equal(result.traceId, "header-trace");
});

test("qa: maps backend error code to ApiClientError", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async () =>
    mockJsonResponse(404, {
      ok: false,
      error: { code: "SESSION_NOT_FOUND", message: "not found", detail: { session_id: "sid-missing" } },
      trace_id: "trace-missing",
    });

  const api = createApiClient(() => "http://127.0.0.1:8000");

  await assert.rejects(
    () => api.ask({ question: "你好", session_id: "sid-missing" }),
    (error) => {
      assert.ok(error instanceof ApiClientError);
      assert.equal(error.code, "SESSION_NOT_FOUND");
      assert.equal(error.category, "session");
      assert.equal(error.status, 404);
      assert.equal(error.traceId, "trace-missing");
      return true;
    }
  );
});

test("sessions: invalid payload is mapped as HTTP_ERROR", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async () => mockJsonResponse(500, "bad-payload", { "x-trace-id": "trace-http-500" });

  const api = createApiClient(() => "http://127.0.0.1:8000");

  await assert.rejects(
    () => api.listSessions(),
    (error) => {
      assert.ok(error instanceof ApiClientError);
      assert.equal(error.code, "HTTP_ERROR");
      assert.equal(error.category, "http");
      assert.equal(error.status, 500);
      assert.equal(error.traceId, "trace-http-500");
      return true;
    }
  );
});
