import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBubble } from "../components/MessageBubble";

// Minimal stub so SuggestChips doesn't need the real store
vi.mock("../store/chatStore", () => ({
  useChatStore: (sel: (s: { isStreaming: boolean; sendMessage: () => void }) => unknown) =>
    sel({ isStreaming: false, sendMessage: () => {} }),
}));

describe("MessageBubble", () => {
  it("renders user message on the right", () => {
    const msg = { id: "1", role: "user" as const, content: "你好" };
    const { container } = render(<MessageBubble message={msg} />);
    expect(screen.getByText("你好")).toBeTruthy();
    // user bubble has justify-end class
    expect(container.firstChild).toHaveClass("justify-end");
  });

  it("renders assistant message on the left", () => {
    const msg = { id: "2", role: "assistant" as const, content: "您好，请问有什么可以帮您？" };
    const { container } = render(<MessageBubble message={msg} />);
    expect(container.firstChild).toHaveClass("justify-start");
  });

  it("shows streaming cursor when isStreaming=true and content non-empty", () => {
    const msg = { id: "3", role: "assistant" as const, content: "正在回答", isStreaming: true };
    const { container } = render(<MessageBubble message={msg} />);
    // cursor span should be in DOM
    const cursor = container.querySelector(".animate-pulse");
    expect(cursor).not.toBeNull();
  });

  it("renders suggest chips when message has suggestions", () => {
    const msg = {
      id: "4",
      role: "assistant" as const,
      content: "完整回答",
      isStreaming: false,
      suggestions: ["追问一", "追问二"],
    };
    render(<MessageBubble message={msg} />);
    expect(screen.getByText("追问一")).toBeTruthy();
    expect(screen.getByText("追问二")).toBeTruthy();
  });
});
