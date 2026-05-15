// Left sidebar: session list, new-chat button, delete session
import { useEffect, useState } from "react";
import { useChatStore } from "../store/chatStore";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

// Herb leaf SVG icon — traditional botanical illustration style
function HerbIcon({ size = 18, className = "" }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
    >
      {/* stem */}
      <path
        d="M12 22V8"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        fill="none"
      />
      {/* left leaf */}
      <path d="M12 16C9 15 5 12 5 8c3 0 7 4 7 8z" opacity="0.9" />
      {/* right leaf */}
      <path d="M12 11c3-2 7-5 6-8-3 0-7 4-6 8z" opacity="0.9" />
    </svg>
  );
}

export function Sidebar({ isOpen, onClose }: Props) {
  const sessions = useChatStore((s) => s.sessions);
  const activeSessionId = useChatStore((s) => s.activeSessionId);
  const loadSessions = useChatStore((s) => s.loadSessions);
  const selectSession = useChatStore((s) => s.selectSession);
  const startNewSession = useChatStore((s) => s.startNewSession);
  const removeSession = useChatStore((s) => s.deleteSession);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("确认删除此会话？")) return;
    setDeletingId(id);
    try {
      await removeSession(id);
    } finally {
      setDeletingId(null);
    }
  };

  const handleSelect = (id: string) => {
    selectSession(id);
    onClose();
  };

  return (
    <>
      {/* mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-10 md:hidden backdrop-blur-sm"
          onClick={onClose}
        />
      )}

      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-20
          w-64 bg-surface flex flex-col transition-transform duration-200
          border-r border-surface-3
          ${isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
        `}
      >
        {/* brand */}
        <div className="px-5 py-5">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-accent/15 flex items-center justify-center text-accent">
              <HerbIcon size={16} />
            </div>
            <div>
              <div className="text-accent font-serif font-semibold text-sm leading-tight tracking-wide">
                中医问诊
              </div>
              <div className="text-muted text-[10px] tracking-wider">TCM·AI·助手</div>
            </div>
          </div>
          {/* gold divider */}
          <div className="gold-divider mt-4" />
        </div>

        {/* new chat */}
        <div className="px-3 pb-3">
          <button
            onClick={() => { startNewSession(); onClose(); }}
            className="
              w-full flex items-center gap-2 px-3 py-2.5 rounded-xl
              bg-accent/10 hover:bg-accent/18 text-accent text-sm
              border border-accent/25 hover:border-accent/45
              transition-all duration-150 font-medium
            "
          >
            <span className="text-base leading-none">＋</span>
            新建对话
          </button>
        </div>

        {/* session list */}
        <div className="flex-1 overflow-y-auto px-3 py-1 space-y-0.5">
          {sessions.length === 0 ? (
            <p className="text-muted text-xs px-3 py-4 text-center italic">暂无历史对话</p>
          ) : (
            sessions.map((s) => (
              <div
                key={s.sessionId}
                onClick={() => handleSelect(s.sessionId)}
                className={`
                  group flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer
                  transition-all duration-150 text-sm
                  ${
                    activeSessionId === s.sessionId
                      ? "bg-accent/12 text-paper border border-accent/20"
                      : "hover:bg-surface-2 text-muted hover:text-paper border border-transparent"
                  }
                `}
              >
                {/* active indicator dot */}
                {activeSessionId === s.sessionId && (
                  <span className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
                )}
                <span className="flex-1 truncate">{s.title}</span>
                <button
                  onClick={(e) => handleDelete(e, s.sessionId)}
                  disabled={deletingId === s.sessionId}
                  className="
                    opacity-0 group-hover:opacity-60 hover:!opacity-100
                    text-muted hover:text-vermillion
                    transition-all disabled:opacity-20 text-xs
                  "
                  aria-label="删除会话"
                >
                  ✕
                </button>
              </div>
            ))
          )}
        </div>

        {/* footer */}
        <div className="px-4 py-3">
          <div className="gold-divider mb-3" />
          <p className="text-[10px] text-muted/50 text-center tracking-wide">
            ✦ 仅供参考，不构成医疗建议 ✦
          </p>
        </div>
      </aside>
    </>
  );
}
