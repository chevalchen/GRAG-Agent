// Main content area: HomeHero or MessageList + ErrorBanner + Composer
import { useChatStore } from "../store/chatStore";
import { HomeHero } from "./HomeHero";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { ErrorBanner } from "./ErrorBanner";

export function ChatArea() {
  const messages = useChatStore((s) => s.messages);
  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full">
      {/* header bar */}
      <div className="shrink-0 px-6 py-3 border-b border-surface-3 flex items-center gap-3">
        {/* decorative herb dot */}
        <span className="w-1.5 h-1.5 rounded-full bg-accent/60 shrink-0" />
        <span className="text-sm font-medium text-paper/80 tracking-wide">
          中医药智能问诊助手
        </span>
        {/* right spacer with subtle decorative element */}
        <span className="ml-auto text-xs text-muted/40 tracking-widest hidden sm:block">
          本草·知识图谱
        </span>
      </div>

      {/* error banner */}
      <ErrorBanner />

      {/* messages or hero */}
      {isEmpty ? (
        <div className="flex-1 overflow-y-auto">
          <HomeHero />
        </div>
      ) : (
        <MessageList />
      )}

      {/* composer always at bottom */}
      <Composer />
    </div>
  );
}
