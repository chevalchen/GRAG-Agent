import { mapError } from "./errorMap.js";

/**
 * @typedef {{ ok: true, data: unknown, trace_id?: string }} SuccessEnvelope
 * @typedef {{ code?: string, message?: string, detail?: unknown }} EnvelopeError
 * @typedef {{ ok: false, error?: EnvelopeError, trace_id?: string }} ErrorEnvelope
 * @typedef {SuccessEnvelope | ErrorEnvelope} ApiEnvelope
 * @typedef {{ data: unknown, traceId: string }} ParsedSuccess
 */

class ApiClientError extends Error {
  constructor({ code, message, category, status, traceId, detail }) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.category = category;
    this.status = status;
    this.traceId = traceId || "";
    this.detail = detail;
  }
}

function generateTraceId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `trace-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/**
 * @param {unknown} payload
 * @returns {payload is ApiEnvelope}
 */
function isApiEnvelope(payload) {
  return !!payload && typeof payload === "object" && Object.prototype.hasOwnProperty.call(payload, "ok");
}

async function parseEnvelope(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }

  const traceId = response.headers.get("x-trace-id") || payload?.trace_id || "";

  if (!isApiEnvelope(payload)) {
    const mapped = mapError("HTTP_ERROR", `响应体格式非法（status=${response.status}）`);
    throw new ApiClientError({
      ...mapped,
      status: response.status,
      traceId,
      detail: null,
    });
  }

  if (payload.ok === true) {
    return { data: payload.data, traceId };
  }

  const backendCode = payload?.error?.code || "UNKNOWN_ERROR";
  const backendMessage = payload?.error?.message || "接口调用失败";
  const mapped = mapError(backendCode, backendMessage);

  throw new ApiClientError({
    ...mapped,
    status: response.status,
    traceId,
    detail: payload?.error?.detail,
  });
}

export function createApiClient(getBaseUrl) {
  async function request(path, { method = "GET", body, traceId } = {}) {
    const baseUrl = (getBaseUrl() || "").replace(/\/+$/, "");
    const finalTraceId = traceId || generateTraceId();

    let response;
    try {
      response = await fetch(`${baseUrl}${path}`, {
        method,
        headers: {
          "Content-Type": "application/json",
          "x-trace-id": finalTraceId,
        },
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (error) {
      const mapped = mapError("NETWORK_ERROR", error?.message || "网络请求失败");
      throw new ApiClientError({
        ...mapped,
        status: 0,
        traceId: finalTraceId,
        detail: null,
      });
    }

    return parseEnvelope(response);
  }

  return {
    health: (traceId) => request("/api/v1/health", { method: "GET", traceId }),
    ask: ({ question, session_id }, traceId) =>
      request("/api/v1/qa", { method: "POST", body: { question, session_id }, traceId }),
    listSessions: (traceId) => request("/api/v1/sessions", { method: "GET", traceId }),
    deleteSession: (sessionId, traceId) =>
      request(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE", traceId }),
  };
}

export { ApiClientError };
