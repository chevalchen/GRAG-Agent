import { createApiClient, ApiClientError } from "./apiClient.js";

const state = {
  apiBase: "http://127.0.0.1:8000",
  currentQuestion: "",
  currentSessionId: "",
  answerText: "",
  sessions: [],
  healthStatus: "unknown",
  loading: { health: false, qa: false, sessions: false, deleteSession: false },
  error: null,
  lastTraceId: "",
};

const els = {
  apiBase: document.getElementById("apiBase"),
  btnHealth: document.getElementById("btnHealth"),
  question: document.getElementById("question"),
  sessionId: document.getElementById("sessionId"),
  btnAsk: document.getElementById("btnAsk"),
  btnRefreshSessions: document.getElementById("btnRefreshSessions"),
  btnDeleteSession: document.getElementById("btnDeleteSession"),
  healthText: document.getElementById("healthText"),
  answerText: document.getElementById("answerText"),
  sessionList: document.getElementById("sessionList"),
  traceText: document.getElementById("traceText"),
  errorText: document.getElementById("errorText"),
};

const api = createApiClient(() => state.apiBase);

function setError(error) {
  state.error = error
    ? {
        category: error.category || "unknown",
        code: error.code || "UNKNOWN_ERROR",
        message: error.message || "未知错误",
        traceId: error.traceId || "",
        status: error.status || 0,
      }
    : null;
}

function renderSessions() {
  els.sessionList.innerHTML = "";
  state.sessions.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.session_id}${item.updated_at ? ` | ${item.updated_at}` : ""}`;
    if (item.session_id === state.currentSessionId && li.style) {
      li.style.borderColor = "rgba(204, 171, 110, 0.78)";
    }
    li.onclick = () => {
      state.currentSessionId = item.session_id;
      els.sessionId.value = item.session_id;
      render();
    };
    els.sessionList.appendChild(li);
  });
}

function render() {
  els.apiBase.value = state.apiBase;
  els.question.value = state.currentQuestion;
  els.sessionId.value = state.currentSessionId;
  const healthLabel = state.loading.health ? "checking..." : state.healthStatus;
  els.healthText.textContent = `健康状态：${healthLabel}`;
  els.answerText.textContent = `回答：${state.answerText || "-"}`;
  els.traceText.textContent = `trace_id：${state.lastTraceId || "-"}`;
  if (state.error) {
    els.errorText.textContent = `错误：${state.error.category}/${state.error.code} | ${state.error.message} | trace_id=${state.error.traceId || "-"}`;
  } else {
    els.errorText.textContent = "错误：-";
  }

  const anyLoading =
    state.loading.health || state.loading.qa || state.loading.sessions || state.loading.deleteSession;
  els.btnHealth.disabled = state.loading.health;
  els.btnAsk.disabled = state.loading.qa;
  els.btnRefreshSessions.disabled = state.loading.sessions;
  els.btnDeleteSession.disabled = state.loading.deleteSession || !state.currentSessionId;
  if (anyLoading) {
    els.errorText.textContent = `${els.errorText.textContent}（请求中...）`;
  }
  renderSessions();
}

async function runAction(type, fn) {
  state.loading[type] = true;
  setError(null);
  render();
  try {
    const result = await fn();
    state.lastTraceId = result.traceId || state.lastTraceId;
  } catch (error) {
    if (error instanceof ApiClientError) {
      setError(error);
      state.lastTraceId = error.traceId || state.lastTraceId;
      if (type === "health") {
        state.healthStatus = "error";
      }
    } else {
      setError({
        category: "unknown",
        code: "UNKNOWN_ERROR",
        message: error?.message || "未知异常",
        traceId: "",
        status: 0,
      });
    }
  } finally {
    state.loading[type] = false;
    render();
  }
}

async function refreshSessions() {
  await runAction("sessions", async () => {
    const result = await api.listSessions();
    state.sessions = result.data?.sessions || [];
    return result;
  });
}

els.apiBase.addEventListener("input", (e) => {
  state.apiBase = e.target.value.trim();
});

els.question.addEventListener("input", (e) => {
  state.currentQuestion = e.target.value;
});

els.sessionId.addEventListener("input", (e) => {
  state.currentSessionId = e.target.value;
});

els.btnHealth.onclick = async () => {
  await runAction("health", async () => {
    const result = await api.health();
    state.healthStatus = result.data?.status || "unknown";
    return result;
  });
};

els.btnAsk.onclick = async () => {
  await runAction("qa", async () => {
    const result = await api.ask({
      question: state.currentQuestion,
      session_id: state.currentSessionId,
    });
    state.answerText = result.data?.answer || "";
    state.currentSessionId = result.data?.session_id || state.currentSessionId;
    await refreshSessions();
    return result;
  });
};

els.btnRefreshSessions.onclick = refreshSessions;

els.btnDeleteSession.onclick = async () => {
  await runAction("deleteSession", async () => {
    const sid = state.currentSessionId;
    const result = await api.deleteSession(sid);
    if ((result.data?.session_id || "") === sid) {
      state.currentSessionId = "";
    }
    await refreshSessions();
    return result;
  });
};

render();
