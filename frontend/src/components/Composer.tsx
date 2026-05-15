// Input area: textarea + send / stop button
import { useState, useRef, useCallback } from "react";
import { useChatStore } from "../store/chatStore";

export function Composer() {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const sendMessage = useChatStore((s) => s.sendMessage);
  const stopStreaming = useChatStore((s) => s.stopStreaming);
  const isStreaming = useChatStore((s) => s.isStreaming);

  const handleSend = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) return;
    setText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    await sendMessage(trimmed);
  }, [text, isStreaming, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts newline
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // auto-resize up to ~6 lines
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 144) + "px";
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div
        className="
          flex items-end gap-2
          bg-surface-2 rounded-2xl
          border border-surface-3
          focus-within:border-accent/50
          px-4 py-3 transition-all duration-200
          gold-ring
        "
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="向中医助手提问…（Enter 发送，Shift+Enter 换行）"
          rows={1}
          disabled={isStreaming}
          className="
            flex-1 bg-transparent resize-none outline-none
            text-sm text-paper placeholder-muted
            leading-relaxed disabled:opacity-50
          "
          style={{ maxHeight: "144px" }}
        />

        {isStreaming ? (
          <button
            onClick={stopStreaming}
            className="
              shrink-0 w-9 h-9 rounded-xl
              bg-vermillion/20 hover:bg-vermillion/40
              border border-vermillion/30
              flex items-center justify-center
              text-vermillion transition-all duration-150
            "
            aria-label="停止生成"
          >
            {/* stop square */}
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
              <rect x="2" y="2" width="10" height="10" rx="2" />
            </svg>
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!text.trim()}
            className="
              shrink-0 w-9 h-9 rounded-xl
              bg-accent hover:bg-accent-light
              flex items-center justify-center
              text-base transition-all duration-150
              disabled:opacity-25 disabled:cursor-not-allowed
              shadow-sm
            "
            aria-label="发送"
            style={{ color: "#0f2018" }}
          >
            {/* upward arrow */}
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path d="M8 13V4M4 8l4-4 4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />
            </svg>
          </button>
        )}
      </div>

      <p className="text-center text-[10px] text-muted/45 mt-1.5 tracking-wide">
        ✦ 内容仅供参考，不构成医疗建议 ✦
      </p>
    </div>
  );
}
