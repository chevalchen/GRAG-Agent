# Stage 1 Frontend Design — TCM Chat UI

## 1. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Bundler | Vite 5 | Zero-config, instant HMR, native ESM |
| UI framework | React 18 + TypeScript | Component model ideal for streaming chat |
| Styling | Tailwind CSS v3 | Utility-first; no custom CSS overhead |
| Markdown | react-markdown + remark-gfm | Safe render with GFM tables/links |
| State | Zustand | Minimal boilerplate; avoids Redux for 2-screen app |
| Testing | Vitest + @testing-library/react | Co-located with Vite, fast |
| HTTP | native fetch + EventSource | No external HTTP lib needed |

## 2. Directory Layout

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
└── src/
    ├── main.tsx           # ReactDOM.createRoot
    ├── App.tsx            # root layout: Sidebar + ChatArea
    ├── api/
    │   ├── client.ts      # fetch wrapper, envelope {ok,data,error,trace_id}
    │   ├── stream.ts      # SSE consumer: token | suggest | done | error events
    │   └── sessions.ts    # GET /sessions, DELETE /sessions/:id
    ├── store/
    │   └── chatStore.ts   # Zustand: sessions[], activeSessionId, messages[], streaming
    ├── components/
    │   ├── Sidebar.tsx
    │   ├── ChatArea.tsx   # scroll container for MessageList + Composer
    │   ├── MessageList.tsx
    │   ├── MessageBubble.tsx
    │   ├── Composer.tsx
    │   ├── SuggestChips.tsx
    │   ├── HomeHero.tsx
    │   └── ErrorBanner.tsx
    ├── lib/
    │   ├── markdown.ts    # ReactMarkdown component config
    │   ├── errorMap.ts    # backend code → human-readable string
    │   └── topics.ts      # static preset TCM questions
    └── styles/
        └── globals.css    # Tailwind base + custom scrollbar
```

## 3. Data Models (TypeScript)

```typescript
// All messages in a chat session
interface Message {
  id: string;          // nanoid
  role: "user" | "assistant";
  content: string;     // markdown text (may be partial during streaming)
  suggestions?: string[];  // post-response follow-up chips
  isStreaming?: boolean;
  error?: string;      // inline error suffix
}

// A chat session entry (sidebar)
interface Session {
  sessionId: string;
  title: string;       // first 30 chars of first user message
  createdAt?: number;
}

// Zustand store shape
interface ChatStore {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Message[];           // messages for active session
  isStreaming: boolean;
  error: string | null;

  // actions
  loadSessions(): Promise<void>;
  selectSession(id: string): void;
  startNewSession(): void;
  deleteSession(id: string): Promise<void>;
  sendMessage(text: string): Promise<void>;
  stopStreaming(): void;
  clearError(): void;
}
```

## 4. SSE Protocol (frontend ↔ backend)

`POST /api/v1/qa/stream` — request body: `{question, session_id?}`

Event stream format (server-sent events):
```
event: token
data: {"chunk": "某个"}

event: suggest
data: {"questions": ["哪些人群慎用？", "有无副作用？", "如何服用？"]}

event: done
data: {"session_id": "uuid", "trace_id": "uuid"}

event: error
data: {"code": "INTERNAL_ERROR", "message": "..."}
```

`stream.ts` wraps `fetch` (not `EventSource`) so it can send a POST body and supports `AbortController`.

## 5. Component Interaction

```
App
├── Sidebar                  ← session list, new-chat button
│   └── SessionItem[]
└── ChatArea
    ├── HomeHero             ← shown when messages=[] && !activeSessionId
    ├── MessageList
    │   └── MessageBubble[]  ← markdown + role avatar
    │       └── SuggestChips ← shown after assistant message finishes
    ├── ErrorBanner          ← dismissible; shown when store.error != null
    └── Composer             ← textarea + Send / Stop button
```

## 6. Streaming Flow

1. User types → presses Enter or clicks Send.
2. `chatStore.sendMessage(text)` appends a `user` message + placeholder `assistant` message with `isStreaming: true`.
3. `stream.ts` opens `fetch` POST to `/api/v1/qa/stream`.
4. On each `token` event: append `chunk` to placeholder message content.
5. On `suggest` event: store `questions` in the message.
6. On `done` event: set `isStreaming: false`; save `session_id` returned; update sidebar title if new session.
7. On `error` event or fetch rejection: set `message.error` + `store.error`.
8. Stop button → `AbortController.abort()` → stream closes, `isStreaming` = false.

## 7. Color Scheme (dark default)

| Token | Value |
|---|---|
| bg-base | `#1a1a2e` (deep navy) |
| bg-surface | `#16213e` |
| bg-bubble-user | `#0f3460` |
| bg-bubble-assistant | `#1e2a3a` |
| accent | `#e94560` (TCM red) |
| text-primary | `#e0e0e0` |
| text-muted | `#8a9bb0` |

## 8. Backend Changes Required (Stage 1 only)

| File | Change |
|---|---|
| `src/app/config.py` | Add `frontend_origins: list[str]`, default `["http://localhost:5173"]` |
| `src/app/api_server.py` | Read CORS origins from config; add `POST /api/v1/qa/stream` (StreamingResponse + SSE); keep old `/qa` intact |

The SSE route in this stage streams the **current synchronous answer** as a single `token` event followed by `done`, so the frontend SSE consumer works end-to-end even before Stage 2's astream refactor.
