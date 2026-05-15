// Follow-up suggestion chips rendered after an assistant message
import { useChatStore } from "../store/chatStore";

interface Props {
  questions: string[];
}

export function SuggestChips({ questions }: Props) {
  const sendMessage = useChatStore((s) => s.sendMessage);
  const isStreaming = useChatStore((s) => s.isStreaming);

  if (!questions.length) return null;

  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {/* "你可能想问" label */}
      <span className="w-full text-[10px] text-muted/70 tracking-widest mb-0.5">
        猜你想问
      </span>
      {questions.map((q) => (
        <button
          key={q}
          disabled={isStreaming}
          onClick={() => sendMessage(q)}
          className="
            px-3 py-1.5 text-xs rounded-full
            border border-accent/35 text-accent/90
            hover:bg-accent/10 hover:border-accent/60 hover:text-accent
            transition-all duration-150
            disabled:opacity-35 disabled:cursor-not-allowed
            whitespace-nowrap max-w-[260px] truncate
          "
          title={q}
        >
          {q}
        </button>
      ))}
    </div>
  );
}
