// Maps backend error codes to human-readable Chinese messages
const ERROR_MAP: Record<string, string> = {
  VALIDATION_ERROR: "输入内容格式有误，请检查后重试。",
  INVALID_ARGUMENT: "请求参数无效，请重新输入。",
  SESSION_NOT_FOUND: "会话已过期或不存在，请新建对话。",
  HTTP_ERROR: "网络请求失败，请检查连接后重试。",
  INTERNAL_ERROR: "服务器内部错误，请稍后再试。",
  STREAM_INTERRUPTED: "回答传输中断，请重新发送问题。",
};

export function getErrorMessage(code: string, fallback?: string): string {
  return ERROR_MAP[code] ?? fallback ?? "发生未知错误，请稍后重试。";
}
