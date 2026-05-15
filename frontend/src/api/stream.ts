// SSE consumer for POST /api/v1/qa/stream
// Uses fetch + ReadableStream so we can send a POST body and use AbortController.
// Server emits: token | suggest | done | error

export interface StreamCallbacks {
  onToken: (chunk: string) => void;
  onSuggest: (questions: string[]) => void;
  onDone: (sessionId: string, traceId: string) => void;
  onError: (code: string, message: string) => void;
}

export async function streamQA(
  question: string,
  sessionId: string | null,
  callbacks: StreamCallbacks,
  signal: AbortSignal,
): Promise<void> {
  let response: Response;

  try {
    response = await fetch("/api/v1/qa/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId ?? "" }),
      signal,
    });
  } catch (err: unknown) {
    if ((err as Error).name === "AbortError") return;
    callbacks.onError("HTTP_ERROR", (err as Error).message);
    return;
  }

  if (!response.ok || !response.body) {
    callbacks.onError("HTTP_ERROR", `HTTP ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    let done: boolean;
    let value: Uint8Array | undefined;

    try {
      ({ done, value } = await reader.read());
    } catch {
      // aborted
      break;
    }

    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE lines are separated by \n\n
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const lines = part.split("\n");
      let eventType = "message";
      let dataLine = "";

      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLine = line.slice(5).trim();
        }
      }

      if (!dataLine) continue;

      try {
        const payload = JSON.parse(dataLine);
        switch (eventType) {
          case "token":
            callbacks.onToken(payload.chunk ?? "");
            break;
          case "suggest":
            callbacks.onSuggest(payload.questions ?? []);
            break;
          case "done":
            callbacks.onDone(payload.session_id ?? "", payload.trace_id ?? "");
            break;
          case "error":
            callbacks.onError(payload.code ?? "INTERNAL_ERROR", payload.message ?? "");
            break;
        }
      } catch {
        // malformed JSON — skip
      }
    }
  }
}
