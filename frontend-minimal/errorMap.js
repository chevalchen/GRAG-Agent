const ERROR_MAP = {
  VALIDATION_ERROR: {
    category: "validation",
    message: "请求参数校验失败，请检查输入格式。",
  },
  INVALID_ARGUMENT: {
    category: "argument",
    message: "参数不合法，请检查 question 或 session_id。",
  },
  SESSION_NOT_FOUND: {
    category: "session",
    message: "会话不存在，可能已被删除。",
  },
  INTERNAL_ERROR: {
    category: "server",
    message: "服务内部错误，请稍后重试。",
  },
  HTTP_ERROR: {
    category: "http",
    message: "HTTP 调用异常，请检查接口地址与方法。",
  },
  NETWORK_ERROR: {
    category: "network",
    message: "网络异常，请检查服务是否启动。",
  },
};

export function mapError(code, fallbackMessage = "未知错误，请联系维护者。") {
  const mapped = ERROR_MAP[code];
  if (mapped) {
    return { code, category: mapped.category, message: mapped.message };
  }
  return { code: code || "UNKNOWN_ERROR", category: "unknown", message: fallbackMessage };
}
