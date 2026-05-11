import { createApiClient, ApiClientError } from "./apiClient.js";

const ROLE_PRESETS = {
  patient: {
    label: "病人",
    hint: "聚焦个人可执行建议与风险提醒，回答会优先给出日常可落地做法。",
    followups: ["我目前的症状持续多久了？", "是否有用药禁忌或就医红线？"],
  },
  consultant: {
    label: "咨询人员",
    hint: "聚焦沟通话术与转诊边界，便于咨询场景做风险分层。",
    followups: ["面对该问题应先问哪些关键信息？", "哪些情况需要建议用户立即就医？"],
  },
  enthusiast: {
    label: "中医药爱好者",
    hint: "聚焦原理、辨证思路与知识点解释，适合学习交流。",
    followups: ["这个建议对应的中医理论依据是什么？", "可进一步学习哪些经典条目？"],
  },
};

const state = {
  role: "patient",
  apiBase: "http://127.0.0.1:8000",
  sessionId: "",
  question: "",
  sessions: [],
  lastTraceId: "",
  loading: false,
};

const els = {
  apiBase: document.getElementById("apiBase"),
  sessionId: document.getElementById("sessionId"),
  question: document.getElementById("question"),
  btnHealth: document.getElementById("btnHealth"),
  btnAsk: document.getElementById("btnAsk"),
  btnRefreshSessions: document.getElementById("btnRefreshSessions"),
  btnDeleteSession: document.getElementById("btnDeleteSession"),
  rolePatient: document.getElementById("rolePatient"),
  roleConsultant: document.getElementById("roleConsultant"),
  roleEnthusiast: document.getElementById("roleEnthusiast"),
  roleHint: document.getElementById("roleHint"),
  summaryText: document.getElementById("summaryText"),
  actionsList: document.getElementById("actionsList"),
  warningsList: document.getElementById("warningsList"),
  followupList: document.getElementById("followupList"),
  traceText: document.getElementById("traceText"),
  answerText: document.getElementById("answerText"),
  sessionList: document.getElementById("sessionList"),
  errorModal: document.getElementById("errorModal"),
  errorMain: document.getElementById("errorMain"),
  errorDetail: document.getElementById("errorDetail"),
  closeErrorModal: document.getElementById("closeErrorModal"),
};

const api = createApiClient(() => state.apiBase);

function setList(ul, items, emptyText = "暂无信息") {
  ul.innerHTML = "";
  const values = items.length ? items : [emptyText];
  values.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    ul.appendChild(li);
  });
}

function splitSentences(text) {
  return (text || "")
    .split(/[\n。！？!?\r]+/g)
    .map((x) => x.trim())
    .filter(Boolean);
}

function packAnswerInfo(answer) {
  const lines = splitSentences(answer);
  const summary = lines[0] || "暂无结论。";
  const actions = lines.filter((x) => /(建议|应|可以|保持|注意|宜|规律|作息|饮食)/.test(x)).slice(0, 5);
  const warnings = lines.filter((x) => /(避免|禁|慎|不要|警惕|风险|就医|严重)/.test(x)).slice(0, 5);
  const roleFollowups = ROLE_PRESETS[state.role].followups;
  const extraFollowups = lines.filter((x) => /(追问|进一步|补充|还可了解)/.test(x)).slice(0, 3);
  return {
    summary,
    actions,
    warnings,
    followups: [...extraFollowups, ...roleFollowups].slice(0, 5),
  };
}

function applyRole(role) {
  state.role = role;
  els.rolePatient.classList.toggle("active", role === "patient");
  els.roleConsultant.classList.toggle("active", role === "consultant");
  els.roleEnthusiast.classList.toggle("active", role === "enthusiast");
  els.roleHint.textContent = `当前对象：${ROLE_PRESETS[role].label}。${ROLE_PRESETS[role].hint}`;
}

function setLoading(loading) {
  state.loading = loading;
  els.btnAsk.disabled = loading;
  els.btnHealth.disabled = loading;
  els.btnRefreshSessions.disabled = loading;
  els.btnDeleteSession.disabled = loading || !state.sessionId;
}

function showErrorModal(err) {
  const title = err instanceof ApiClientError ? `${err.category}/${err.code}` : "UNKNOWN_ERROR";
  const message = err?.message || "未知异常";
  const detailObj =
    err instanceof ApiClientError
      ? { status: err.status, trace_id: err.traceId, detail: err.detail }
      : { raw: String(err) };
  els.errorMain.textContent = `${title}：${message}`;
  els.errorDetail.textContent = JSON.stringify(detailObj, null, 2);
  els.errorModal.classList.remove("hidden");
  els.errorModal.setAttribute("aria-hidden", "false");
}

function hideErrorModal() {
  els.errorModal.classList.add("hidden");
  els.errorModal.setAttribute("aria-hidden", "true");
}

function renderSessions() {
  els.sessionList.innerHTML = "";
  state.sessions.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.session_id}${item.updated_at ? ` | ${item.updated_at}` : ""}`;
    li.onclick = () => {
      state.sessionId = item.session_id;
      els.sessionId.value = item.session_id;
      setLoading(false);
    };
    els.sessionList.appendChild(li);
  });
}

function roleWrappedQuestion(question) {
  const profile = ROLE_PRESETS[state.role];
  return `【咨询对象】${profile.label}\n【提问】${question}`;
}

async function refreshSessions() {
  const res = await api.listSessions();
  state.lastTraceId = res.traceId || state.lastTraceId;
  state.sessions = res.data?.sessions || [];
  renderSessions();
  els.traceText.textContent = `trace_id：${state.lastTraceId || "-"}`;
}

async function askQuestion() {
  const q = state.question.trim();
  if (!q) {
    showErrorModal(new Error("请输入问题后再发起咨询。"));
    return;
  }
  setLoading(true);
  try {
    const payload = {
      question: roleWrappedQuestion(q),
      session_id: state.sessionId.trim(),
    };
    const res = await api.ask(payload);
    state.lastTraceId = res.traceId || state.lastTraceId;
    state.sessionId = res.data?.session_id || state.sessionId;
    els.sessionId.value = state.sessionId;
    els.traceText.textContent = `trace_id：${state.lastTraceId || "-"}`;
    const answer = res.data?.answer || "";
    els.answerText.textContent = `原始回答：${answer || "-"}`;
    const packed = packAnswerInfo(answer);
    els.summaryText.textContent = packed.summary;
    setList(els.actionsList, packed.actions, "暂无明确可执行建议");
    setList(els.warningsList, packed.warnings, "暂无明显风险提示");
    setList(els.followupList, packed.followups, "暂无追问建议");
    await refreshSessions();
  } catch (err) {
    showErrorModal(err);
  } finally {
    setLoading(false);
  }
}

async function healthCheck() {
  setLoading(true);
  try {
    const res = await api.health();
    state.lastTraceId = res.traceId || state.lastTraceId;
    els.traceText.textContent = `trace_id：${state.lastTraceId || "-"}`;
    els.summaryText.textContent = `后端健康状态：${res.data?.status || "unknown"}`;
  } catch (err) {
    showErrorModal(err);
  } finally {
    setLoading(false);
  }
}

els.apiBase.addEventListener("input", (e) => {
  state.apiBase = e.target.value.trim();
});
els.sessionId.addEventListener("input", (e) => {
  state.sessionId = e.target.value;
  setLoading(false);
});
els.question.addEventListener("input", (e) => {
  state.question = e.target.value;
});
els.btnAsk.onclick = askQuestion;
els.btnHealth.onclick = healthCheck;
els.btnRefreshSessions.onclick = async () => {
  setLoading(true);
  try {
    await refreshSessions();
  } catch (err) {
    showErrorModal(err);
  } finally {
    setLoading(false);
  }
};
els.btnDeleteSession.onclick = async () => {
  if (!state.sessionId.trim()) {
    showErrorModal(new Error("请先选择或输入会话ID。"));
    return;
  }
  setLoading(true);
  try {
    const res = await api.deleteSession(state.sessionId.trim());
    state.lastTraceId = res.traceId || state.lastTraceId;
    els.traceText.textContent = `trace_id：${state.lastTraceId || "-"}`;
    state.sessionId = "";
    els.sessionId.value = "";
    await refreshSessions();
  } catch (err) {
    showErrorModal(err);
  } finally {
    setLoading(false);
  }
};

els.rolePatient.onclick = () => applyRole("patient");
els.roleConsultant.onclick = () => applyRole("consultant");
els.roleEnthusiast.onclick = () => applyRole("enthusiast");
els.closeErrorModal.onclick = hideErrorModal;
els.errorModal.onclick = (e) => {
  if (e.target === els.errorModal) {
    hideErrorModal();
  }
};

applyRole("patient");
setList(els.actionsList, [], "等待回答后自动提取");
setList(els.warningsList, [], "等待回答后自动提取");
setList(els.followupList, ROLE_PRESETS.patient.followups);
renderSessions();
