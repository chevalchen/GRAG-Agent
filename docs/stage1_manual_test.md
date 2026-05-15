# Stage 1 Manual Test Checklist

Run both services before starting:
```bash
# Terminal 1 — backend
cd GRAG-Agent
uv run uvicorn src.app.api_server:app --reload

# Terminal 2 — frontend
cd GRAG-Agent/frontend
npm run dev
# visit http://localhost:5173
```

## Test Cases

### TC-01 Cold Start
- [ ] Homepage shows the "中医药智能问诊助手" title
- [ ] 6–8 preset topic chips are visible
- [ ] Disclaimer text is shown below chips

### TC-02 Click Topic → Streaming Response
- [ ] Click any preset chip
- [ ] Message appears in chat view (hero is replaced)
- [ ] Assistant response streams token by token (text appears progressively)
- [ ] Streaming cursor (blinking line) visible during generation
- [ ] 3 follow-up suggestion chips appear below the answer

### TC-03 Manual Question
- [ ] Type a question in the composer and press Enter
- [ ] Response streams as above
- [ ] Pressing Shift+Enter inserts newline instead of sending

### TC-04 Stop Streaming
- [ ] While streaming: Stop button (⏹) is visible
- [ ] Click Stop → streaming halts, partial response retained
- [ ] Composer is ready for new input after stop

### TC-05 Click Suggestion Chip
- [ ] After an answer, click one of the 3 suggestion chips
- [ ] That text appears as next user message, response starts streaming

### TC-06 Session Management
- [ ] After first reply, session appears in sidebar with title from first message
- [ ] Click "新建对话" → blank chat, hero shows
- [ ] Click the previous session → loads session id (history display: server-side; local messages clear)
- [ ] Hover over session row → delete icon (🗑) visible
- [ ] Click delete → confirmation dialog → session removed from list

### TC-07 Error Handling
- [ ] Stop backend, send a message → ErrorBanner appears with human-readable message
- [ ] Click ✕ on ErrorBanner → dismissed
- [ ] Restart backend → next message works normally

### TC-08 Mobile Layout
- [ ] Resize window to mobile width
- [ ] Sidebar hidden behind hamburger (☰) button
- [ ] Click ☰ → sidebar slides in
- [ ] Click overlay → sidebar closes
- [ ] Chat and composer remain usable at mobile width
