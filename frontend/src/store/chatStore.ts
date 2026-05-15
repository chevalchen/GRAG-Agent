// Zustand store — manages sessions, messages, and streaming state
import { create } from "zustand";
import { nanoid } from "nanoid";
import { streamQA } from "../api/stream";
import { fetchSessions, deleteSession, type SessionEntry } from "../api/sessions";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  suggestions?: string[];
  isStreaming?: boolean;
  error?: string;
}

export interface Session {
  sessionId: string;
  title: string;
}

interface ChatStore {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Message[];
  isStreaming: boolean;
  error: string | null;
  // abort controller for the active stream
  _abortController: AbortController | null;

  loadSessions: () => Promise<void>;
  selectSession: (id: string) => void;
  startNewSession: () => void;
  deleteSession: (id: string) => Promise<void>;
  sendMessage: (text: string) => Promise<void>;
  stopStreaming: () => void;
  clearError: () => void;
  _appendToken: (msgId: string, chunk: string) => void;
  _setSuggestions: (msgId: string, questions: string[]) => void;
  _finalizeMessage: (msgId: string) => void;
  _setMessageError: (msgId: string, err: string) => void;
}

export const useChatStore = create<ChatStore>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  isStreaming: false,
  error: null,
  _abortController: null,

  loadSessions: async () => {
    try {
      const entries: SessionEntry[] = await fetchSessions();
      const sessions: Session[] = entries.map((e) => ({
        sessionId: e.session_id,
        title: e.session_id.slice(0, 8) + "…",
      }));
      set({ sessions });
    } catch {
      // non-fatal: sidebar just stays empty
    }
  },

  selectSession: (id) => {
    // Clear local messages when switching — history is server-side
    set({ activeSessionId: id, messages: [], error: null });
  },

  startNewSession: () => {
    set({ activeSessionId: null, messages: [], error: null });
  },

  deleteSession: async (id) => {
    await deleteSession(id);
    set((s) => ({
      sessions: s.sessions.filter((ss) => ss.sessionId !== id),
      activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
      messages: s.activeSessionId === id ? [] : s.messages,
    }));
  },

  sendMessage: async (text) => {
    const trimmed = text.trim();
    if (!trimmed || get().isStreaming) return;

    const userMsg: Message = { id: nanoid(), role: "user", content: trimmed };
    const assistantMsg: Message = {
      id: nanoid(),
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    set((s) => ({
      messages: [...s.messages, userMsg, assistantMsg],
      isStreaming: true,
      error: null,
    }));

    const controller = new AbortController();
    set({ _abortController: controller });

    const { activeSessionId } = get();
    const assistantId = assistantMsg.id;

    await streamQA(
      trimmed,
      activeSessionId,
      {
        onToken: (chunk) => get()._appendToken(assistantId, chunk),
        onSuggest: (questions) => get()._setSuggestions(assistantId, questions),
        onDone: (sessionId) => {
          // update session id and finalize message
          set((s) => {
            const alreadyExists = s.sessions.some((ss) => ss.sessionId === sessionId);
            // derive title from first user message in this session
            const firstUser = s.messages.find((m) => m.role === "user");
            const title = firstUser
              ? firstUser.content.slice(0, 20) + (firstUser.content.length > 20 ? "…" : "")
              : sessionId.slice(0, 8) + "…";
            return {
              activeSessionId: sessionId,
              sessions: alreadyExists
                ? s.sessions
                : [{ sessionId, title }, ...s.sessions],
            };
          });
          get()._finalizeMessage(assistantId);
        },
        onError: (code, message) => {
          get()._setMessageError(assistantId, message);
          set({ error: `[${code}] ${message}`, isStreaming: false, _abortController: null });
        },
      },
      controller.signal,
    );

    // ensure streaming flag is cleared even if onDone fires before this
    set((s) => ({
      isStreaming: s.messages.some((m) => m.isStreaming) ? false : s.isStreaming,
      _abortController: null,
    }));
  },

  stopStreaming: () => {
    const ctrl = get()._abortController;
    if (ctrl) {
      ctrl.abort();
      set({ isStreaming: false, _abortController: null });
    }
    // finalize any still-streaming message
    set((s) => ({
      messages: s.messages.map((m) =>
        m.isStreaming ? { ...m, isStreaming: false } : m,
      ),
    }));
  },

  clearError: () => set({ error: null }),

  _appendToken: (msgId, chunk) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === msgId ? { ...m, content: m.content + chunk } : m,
      ),
    })),

  _setSuggestions: (msgId, questions) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === msgId ? { ...m, suggestions: questions } : m,
      ),
    })),

  _finalizeMessage: (msgId) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === msgId ? { ...m, isStreaming: false } : m,
      ),
      isStreaming: false,
    })),

  _setMessageError: (msgId, err) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === msgId
          ? { ...m, isStreaming: false, error: err, content: m.content + "\n\n⚠️ " + err }
          : m,
      ),
      isStreaming: false,
    })),
}));
