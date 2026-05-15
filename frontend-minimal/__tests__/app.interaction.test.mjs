import test from "node:test";
import assert from "node:assert/strict";

class FakeElement {
  constructor(id = "") {
    this.id = id;
    this.value = "";
    this.textContent = "";
    this.disabled = false;
    this.onclick = null;
    this.listeners = {};
    this.children = [];
    this._innerHTML = "";
  }

  addEventListener(type, handler) {
    this.listeners[type] = handler;
  }

  dispatchInput(value) {
    this.value = value;
    const handler = this.listeners.input;
    if (handler) {
      handler({ target: this });
    }
  }

  appendChild(child) {
    this.children.push(child);
  }

  set innerHTML(value) {
    this._innerHTML = value;
    if (value === "") {
      this.children = [];
    }
  }

  get innerHTML() {
    return this._innerHTML;
  }
}

function createMockDocument() {
  const ids = [
    "apiBase",
    "btnHealth",
    "question",
    "sessionId",
    "btnAsk",
    "btnRefreshSessions",
    "btnDeleteSession",
    "healthText",
    "answerText",
    "sessionList",
    "traceText",
    "errorText",
  ];
  const map = new Map(ids.map((id) => [id, new FakeElement(id)]));
  return {
    getElementById(id) {
      return map.get(id) || null;
    },
    createElement(tag) {
      return new FakeElement(tag);
    },
    __map: map,
  };
}

function createResponse(status, body, traceId) {
  return {
    status,
    headers: {
      get(name) {
        if (name.toLowerCase() === "x-trace-id") {
          return traceId || null;
        }
        return null;
      },
    },
    async json() {
      return body;
    },
  };
}

test("frontend interaction flow: health -> qa -> sessions -> delete", async (t) => {
  const originalDocument = globalThis.document;
  const originalFetch = globalThis.fetch;

  const sessionStore = [];
  let traceSeed = 0;
  const trace = () => `trace-e2e-${++traceSeed}`;

  globalThis.document = createMockDocument();

  globalThis.fetch = async (url, options = {}) => {
    const u = new URL(url);
    const path = u.pathname;

    if (path === "/api/v1/health" && options.method === "GET") {
      return createResponse(200, { ok: true, data: { status: "ok" }, trace_id: trace() }, trace());
    }

    if (path === "/api/v1/qa" && options.method === "POST") {
      const payload = JSON.parse(options.body || "{}");
      const sid = payload.session_id || "sid-generated";
      if (!sessionStore.includes(sid)) {
        sessionStore.push(sid);
      }
      return createResponse(
        200,
        {
          ok: true,
          data: { answer: `answer:${payload.question}`, session_id: sid },
          trace_id: trace(),
        },
        trace()
      );
    }

    if (path === "/api/v1/sessions" && options.method === "GET") {
      return createResponse(
        200,
        {
          ok: true,
          data: { sessions: sessionStore.map((sid) => ({ session_id: sid, updated_at: "2026-05-11T00:00:00Z" })) },
          trace_id: trace(),
        },
        trace()
      );
    }

    if (path.startsWith("/api/v1/sessions/") && options.method === "DELETE") {
      const sid = decodeURIComponent(path.replace("/api/v1/sessions/", ""));
      const idx = sessionStore.indexOf(sid);
      if (idx >= 0) {
        sessionStore.splice(idx, 1);
        return createResponse(200, { ok: true, data: { session_id: sid }, trace_id: trace() }, trace());
      }
      return createResponse(
        404,
        { ok: false, error: { code: "SESSION_NOT_FOUND", message: "missing" }, trace_id: trace() },
        trace()
      );
    }

    throw new Error(`Unexpected request: ${options.method} ${url}`);
  };

  t.after(() => {
    globalThis.document = originalDocument;
    globalThis.fetch = originalFetch;
  });

  await import("../app.js");

  const els = globalThis.document.__map;
  const apiBase = els.get("apiBase");
  const btnHealth = els.get("btnHealth");
  const question = els.get("question");
  const sessionId = els.get("sessionId");
  const btnAsk = els.get("btnAsk");
  const btnDeleteSession = els.get("btnDeleteSession");
  const healthText = els.get("healthText");
  const answerText = els.get("answerText");
  const sessionList = els.get("sessionList");
  const errorText = els.get("errorText");

  apiBase.dispatchInput("http://127.0.0.1:8000");
  await btnHealth.onclick();
  assert.match(healthText.textContent, /健康状态：ok/);

  question.dispatchInput("四君子汤适合什么人群？");
  sessionId.dispatchInput("sid-e2e-1");
  await btnAsk.onclick();

  assert.match(answerText.textContent, /answer:四君子汤适合什么人群？/);
  assert.equal(sessionList.children.length, 1);
  assert.match(sessionList.children[0].textContent, /sid-e2e-1/);

  await btnDeleteSession.onclick();
  assert.equal(sessionId.value, "");
  assert.equal(sessionList.children.length, 0);
  assert.equal(errorText.textContent, "错误：-");
});
