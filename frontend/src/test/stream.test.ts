// Tests for the SSE stream consumer
import { describe, it, expect, vi, beforeEach } from "vitest";
import { streamQA } from "../api/stream";

function makeSseBody(events: { event: string; data: unknown }[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const text = events
    .map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`)
    .join("");
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
}

describe("streamQA", () => {
  const callbacks = {
    onToken: vi.fn(),
    onSuggest: vi.fn(),
    onDone: vi.fn(),
    onError: vi.fn(),
  };

  beforeEach(() => vi.clearAllMocks());

  it("delivers token, suggest, and done events", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: makeSseBody([
          { event: "token", data: { chunk: "你" } },
          { event: "token", data: { chunk: "好" } },
          { event: "suggest", data: { questions: ["问一", "问二"] } },
          { event: "done", data: { session_id: "sess-1", trace_id: "t-1" } },
        ]),
      }),
    );

    const ctrl = new AbortController();
    await streamQA("test query", null, callbacks, ctrl.signal);

    expect(callbacks.onToken).toHaveBeenCalledWith("你");
    expect(callbacks.onToken).toHaveBeenCalledWith("好");
    expect(callbacks.onSuggest).toHaveBeenCalledWith(["问一", "问二"]);
    expect(callbacks.onDone).toHaveBeenCalledWith("sess-1", "t-1");
    expect(callbacks.onError).not.toHaveBeenCalled();
  });

  it("calls onError when server returns non-ok status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, body: null }),
    );

    const ctrl = new AbortController();
    await streamQA("test", null, callbacks, ctrl.signal);

    expect(callbacks.onError).toHaveBeenCalledWith("HTTP_ERROR", "HTTP 500");
  });
});
