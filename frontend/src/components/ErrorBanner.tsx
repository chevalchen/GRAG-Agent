// Dismissible top-of-chat error banner
import { useChatStore } from "../store/chatStore";
import { getErrorMessage } from "../lib/errorMap";

export function ErrorBanner() {
  const error = useChatStore((s) => s.error);
  const clearError = useChatStore((s) => s.clearError);

  if (!error) return null;

  const codeMatch = error.match(/^\[([A-Z_]+)\]/);
  const code = codeMatch ? codeMatch[1] : "INTERNAL_ERROR";
  const humanMessage = getErrorMessage(code, error);

  return (
    <div className="flex items-center gap-3 mx-4 mt-2 px-4 py-2.5 rounded-xl text-sm
      bg-vermillion/10 border border-vermillion/30 text-paper/80">
      <span className="text-vermillion shrink-0">⚠</span>
      <span className="flex-1">{humanMessage}</span>
      <button
        onClick={clearError}
        className="text-muted hover:text-paper transition-colors shrink-0 text-xs"
        aria-label="关闭错误提示"
      >
        ✕
      </button>
    </div>
  );
}
