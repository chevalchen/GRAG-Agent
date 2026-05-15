// Single chat message — user or assistant, with markdown rendering and suggest chips
import ReactMarkdown from "react-markdown";
import { MARKDOWN_PLUGINS, MARKDOWN_CLASS } from "../lib/markdown";
import { SuggestChips } from "./SuggestChips";
import type { Message } from "../store/chatStore";

interface Props {
  message: Message;
}

// Small herb leaf for the assistant avatar
function AssistantAvatar() {
  return (
    <div className="
      w-8 h-8 rounded-full
      bg-accent/15 border border-accent/25
      flex items-center justify-center
      text-accent shrink-0
    ">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path
          d="M12 22V8"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
        />
        <path d="M12 16C9 15 5 12 5 8c3 0 7 4 7 8z" opacity="0.9" />
        <path d="M12 11c3-2 7-5 6-8-3 0-7 4-6 8z" opacity="0.9" />
      </svg>
    </div>
  );
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-5`}>
      {/* assistant avatar */}
      {!isUser && (
        <div className="mr-3 mt-1">
          <AssistantAvatar />
        </div>
      )}

      <div className={`max-w-[75%] ${isUser ? "order-1" : ""}`}>
        <div
          className={`
            px-4 py-3 text-sm leading-relaxed
            ${
              isUser
                ? "bg-bubble-user text-paper rounded-2xl rounded-br-sm"
                : "bg-bubble-assistant text-paper rounded-2xl rounded-bl-sm border-l-2 border-accent/40"
            }
          `}
        >
          {isUser ? (
            <span className="whitespace-pre-wrap">{message.content}</span>
          ) : (
            <div className={MARKDOWN_CLASS}>
              <ReactMarkdown remarkPlugins={MARKDOWN_PLUGINS}>
                {message.content || (message.isStreaming ? "▍" : "")}
              </ReactMarkdown>
            </div>
          )}

          {/* streaming cursor */}
          {!isUser && message.isStreaming && message.content && (
            <span className="inline-block w-0.5 h-4 bg-accent animate-pulse ml-0.5 align-text-bottom rounded-full" />
          )}
        </div>

        {/* follow-up chips (only for completed assistant messages) */}
        {!isUser && !message.isStreaming && message.suggestions && (
          <SuggestChips questions={message.suggestions} />
        )}
      </div>

      {/* user avatar */}
      {isUser && (
        <div className="
          w-8 h-8 rounded-full ml-3 mt-1 shrink-0
          bg-bubble-user border border-jade/20
          flex items-center justify-center
          text-jade/80 text-xs font-semibold font-serif
        ">
          你
        </div>
      )}
    </div>
  );
}
